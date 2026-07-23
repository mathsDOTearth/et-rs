//! [`Transport`] backed by the SDK software emulator via the C++ device-layer.
//!
//! Enabled by the `emu` cargo feature. This lets a host without ET-SoC-1
//! hardware develop and test against the bundled software emulator: the crate
//! builds exactly the same device-ops command bytes it sends to real hardware
//! and pushes them through `dev::IDeviceLayer`'s SysEmu backend (via the
//! `emu-shim` C ABI), so the `proto`, `device` and `trace` code paths are
//! exercised identically to the ioctl path. Only the byte in/out differs.
//!
//! The emulator boots firmware in a spawned `sys_emu` process when the transport
//! is created, which takes appreciable wall-clock time.

use super::{DmaHostBuffer, DramInfo, PoppedResponse, Transport};
use crate::error::{Error, Result};
use std::ffi::{CString, c_char, c_int, c_long, c_void};
use std::path::Path;
use std::time::Duration;

#[repr(C)]
struct EtEmuDev {
    _private: [u8; 0],
}

// SAFETY: these are the C ABI exports of emu-shim/et_emu_shim.h.
unsafe extern "C" {
    fn et_emu_create(
        sdk_prefix: *const c_char,
        run_dir: *const c_char,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> *mut EtEmuDev;
    fn et_emu_destroy(dev: *mut EtEmuDev);
    fn et_emu_last_error() -> *const c_char;
    fn et_emu_dram_base(dev: *mut EtEmuDev) -> u64;
    fn et_emu_dram_size(dev: *mut EtEmuDev) -> u64;
    fn et_emu_dma_max_elem_size(dev: *mut EtEmuDev) -> u64;
    fn et_emu_dma_max_elem_count(dev: *mut EtEmuDev) -> u32;
    fn et_emu_dma_alignment_bits(dev: *mut EtEmuDev) -> c_int;
    fn et_emu_sq_count(dev: *mut EtEmuDev) -> u32;
    fn et_emu_sq_max_msg(dev: *mut EtEmuDev) -> u32;
    fn et_emu_push_sq(
        dev: *mut EtEmuDev,
        sq_index: u16,
        cmd: *const u8,
        size: usize,
        is_dma: u8,
    ) -> c_int;
    fn et_emu_pop_cq(
        dev: *mut EtEmuDev,
        out: *mut u8,
        out_cap: usize,
        out_len: *mut usize,
    ) -> c_int;
    fn et_emu_wait_cq(dev: *mut EtEmuDev, timeout_ms: u32) -> c_int;
    fn et_emu_wait_sq(dev: *mut EtEmuDev, timeout_ms: u32) -> c_int;
    fn et_emu_alloc_dma(dev: *mut EtEmuDev, size: usize, writeable: c_int) -> *mut c_void;
    fn et_emu_free_dma(dev: *mut EtEmuDev, buf: *mut c_void);
    fn et_emu_fw_update(dev: *mut EtEmuDev, img: *const u8, size: usize) -> c_long;
    fn et_emu_trace_size(dev: *mut EtEmuDev, trace_type: u8) -> c_long;
    fn et_emu_extract_trace(
        dev: *mut EtEmuDev,
        trace_type: u8,
        out: *mut u8,
        out_cap: usize,
    ) -> c_long;
}

/// The driver-descriptor DMA flag bit, mirrored so the shim can be told when a
/// command carries host addresses. Matches [`crate::proto::desc_flags::DMA`].
const DESC_FLAG_DMA: u8 = crate::proto::desc_flags::DMA;

/// A [`Transport`] over the software emulator's device-layer backend.
///
/// Not `Send`/`Sync`: it owns a pointer into the single-threaded C++ device
/// layer, consistent with the crate's single-thread command model.
pub struct FfiTransport {
    dev: *mut EtEmuDev,
    max_msg: std::cell::Cell<Option<usize>>,
}

impl FfiTransport {
    /// Boot a single-device software emulator. `sdk_prefix` is the SDK install
    /// root (e.g. `/opt/et`) and `run_dir` a writable directory for emulator
    /// logs. This blocks while the emulator boots firmware.
    pub fn new<P: AsRef<Path>, Q: AsRef<Path>>(sdk_prefix: P, run_dir: Q) -> Result<Self> {
        let prefix = path_to_cstring(sdk_prefix.as_ref())?;
        let run = path_to_cstring(run_dir.as_ref())?;
        let mut err = vec![0i8; 512];
        // SAFETY: valid C strings and a writable error buffer are passed.
        let dev =
            unsafe { et_emu_create(prefix.as_ptr(), run.as_ptr(), err.as_mut_ptr(), err.len()) };
        if dev.is_null() {
            return Err(Error::Protocol(format!(
                "emulator create failed: {}",
                cstr_message(&err)
            )));
        }
        Ok(FfiTransport {
            dev,
            max_msg: std::cell::Cell::new(None),
        })
    }

    fn cached_max_msg(&self) -> Result<usize> {
        if let Some(v) = self.max_msg.get() {
            return Ok(v);
        }
        let v = self.sq_max_msg_size()? as usize;
        self.max_msg.set(Some(v));
        Ok(v)
    }
}

impl Drop for FfiTransport {
    fn drop(&mut self) {
        // SAFETY: `dev` was returned by et_emu_create and is destroyed once.
        unsafe { et_emu_destroy(self.dev) };
    }
}

impl Transport for FfiTransport {
    fn dram_info(&self) -> Result<DramInfo> {
        // SAFETY: `self.dev` is a live device-layer handle.
        unsafe {
            Ok(DramInfo {
                base: et_emu_dram_base(self.dev),
                size: et_emu_dram_size(self.dev),
                // Clamp rather than truncate: the device-layer reports these as
                // 64-/32-bit quantities that may exceed the descriptor widths.
                dma_max_elem_size: et_emu_dma_max_elem_size(self.dev).min(u32::MAX as u64) as u32,
                dma_max_elem_count: et_emu_dma_max_elem_count(self.dev).min(u16::MAX as u32) as u16,
                // The device-layer reports the alignment as a byte quantum too.
                dma_alignment: (et_emu_dma_alignment_bits(self.dev).max(1)).min(u16::MAX as i32)
                    as u16,
            })
        }
    }

    fn fw_update(&self, image: &[u8]) -> Result<()> {
        // SAFETY: `image` is a valid slice for `image.len()` bytes.
        let rc = unsafe { et_emu_fw_update(self.dev, image.as_ptr(), image.len()) };
        if rc < 0 {
            Err(ffi_err("fw_update"))
        } else {
            Ok(())
        }
    }

    fn sq_count(&self) -> Result<u16> {
        // SAFETY: live handle.
        Ok(unsafe { et_emu_sq_count(self.dev) } as u16)
    }

    fn sq_max_msg_size(&self) -> Result<u16> {
        // SAFETY: live handle.
        Ok(unsafe { et_emu_sq_max_msg(self.dev) } as u16)
    }

    fn push_sq(&self, sq_index: u16, cmd: &[u8], flags: u8) -> Result<bool> {
        let is_dma = u8::from(flags & DESC_FLAG_DMA != 0);
        // SAFETY: `cmd` is valid for `cmd.len()` bytes.
        let rc = unsafe { et_emu_push_sq(self.dev, sq_index, cmd.as_ptr(), cmd.len(), is_dma) };
        match rc {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(ffi_err("push_sq")),
        }
    }

    fn pop_cq(&self) -> Result<Option<PoppedResponse>> {
        let cap = self.cached_max_msg()?.max(8);
        let mut buf = vec![0u8; cap];
        let mut len = 0usize;
        // SAFETY: `buf` is writable for `cap` bytes; `len` is a live usize.
        let rc = unsafe { et_emu_pop_cq(self.dev, buf.as_mut_ptr(), cap, &mut len) };
        match rc {
            1 => {
                buf.truncate(len.min(cap));
                Ok(Some(PoppedResponse {
                    bytes: buf,
                    cq_index: 0,
                }))
            }
            0 => Ok(None),
            _ => Err(ffi_err("pop_cq")),
        }
    }

    fn extract_trace(&self, trace_type: u8) -> Result<Vec<u8>> {
        // SAFETY: live handle.
        let size = unsafe { et_emu_trace_size(self.dev, trace_type) };
        if size < 0 {
            return Err(ffi_err("trace_size"));
        }
        let mut buf = vec![0u8; size as usize];
        // SAFETY: `buf` is writable for `size` bytes.
        let written =
            unsafe { et_emu_extract_trace(self.dev, trace_type, buf.as_mut_ptr(), buf.len()) };
        if written < 0 {
            return Err(ffi_err("extract_trace"));
        }
        buf.truncate(written as usize);
        Ok(buf)
    }

    fn wait_cq(&self, timeout: Duration) -> Result<bool> {
        let ms = timeout.as_millis().min(u32::MAX as u128) as u32;
        // SAFETY: live handle.
        let rc = unsafe { et_emu_wait_cq(self.dev, ms) };
        if rc < 0 {
            Err(ffi_err("wait_cq"))
        } else {
            Ok(rc == 1)
        }
    }

    fn wait_sq(&self, timeout: Duration) -> Result<bool> {
        let ms = timeout.as_millis().min(u32::MAX as u128) as u32;
        // SAFETY: live handle.
        let rc = unsafe { et_emu_wait_sq(self.dev, ms) };
        if rc < 0 {
            Err(ffi_err("wait_sq"))
        } else {
            Ok(rc == 1)
        }
    }

    fn dma_host_buffer(&self, size: usize) -> Result<Box<dyn DmaHostBuffer>> {
        // SAFETY: live handle; a positive size is requested writeable.
        let ptr = unsafe { et_emu_alloc_dma(self.dev, size.max(1), 1) } as *mut u8;
        if ptr.is_null() {
            return Err(ffi_err("alloc_dma"));
        }
        Ok(Box::new(FfiDmaBuffer {
            dev: self.dev,
            ptr,
            len: size,
        }))
    }
}

/// A DMA host buffer allocated by the emulator device-layer. Its address is
/// valid both for host access and as the DMA endpoint the emulator dereferences,
/// so it is reported as both the virtual and physical node address.
struct FfiDmaBuffer {
    dev: *mut EtEmuDev,
    ptr: *mut u8,
    len: usize,
}

impl DmaHostBuffer for FfiDmaBuffer {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` owns `len` writable bytes for the buffer's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` owns `len` readable bytes for the buffer's lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn virt_addr(&self) -> u64 {
        self.ptr as u64
    }
    fn phys_addr(&self) -> u64 {
        self.ptr as u64
    }
}

impl Drop for FfiDmaBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from et_emu_alloc_dma on `dev`, freed once.
        unsafe { et_emu_free_dma(self.dev, self.ptr.cast()) };
    }
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::Protocol("path contains an interior NUL".into()))
}

fn cstr_message(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The most recent C++-side failure message, for enriching errors.
fn last_error() -> String {
    // SAFETY: the shim returns a valid NUL-terminated, thread-local C string.
    let ptr = unsafe { et_emu_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Build a transport protocol error carrying the shim's last message.
fn ffi_err(op: &str) -> Error {
    Error::Protocol(format!("emulator {op} failed: {}", last_error()))
}
