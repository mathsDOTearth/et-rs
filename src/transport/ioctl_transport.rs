//! [`Transport`] backed by the PCIe kernel driver's operations character device.
//!
//! This backend targets real ET-SoC-1 hardware exposed at `/dev/etN_ops`. Every
//! method is a thin wrapper over one `ETSOC1_IOCTL_*` call. Where the driver's
//! precise return convention is not pinned down by the uapi header (the exact
//! encoding of "queue full" and "queue empty", and whether byte counts arrive
//! via the ioctl return value or the descriptor), the interpretation is called
//! out in comments and kept in one place so it can be reconciled against the
//! driver on the hardware host.

use super::{DmaHostBuffer, DramInfo, PoppedResponse, Transport};
use crate::error::{Error, Result};
use crate::ffi::ops;
use crate::ioctl;
use std::cell::Cell;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::time::Duration;

/// A [`Transport`] over one device's operations node (`/dev/etN_ops`).
pub struct IoctlTransport {
    fd: OwnedFd,
    /// Cached `GET_SQ_MAX_MSG_SIZE`, used to size completion-queue read buffers.
    max_msg: Cell<Option<usize>>,
}

impl IoctlTransport {
    /// Open device `index`, i.e. `/dev/et{index}_ops`.
    pub fn open(index: u32) -> Result<Self> {
        Self::open_path(format!("/dev/et{index}_ops"))
    }

    /// Open an operations node at an explicit filesystem path.
    pub fn open_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::os::unix::ffi::OsStrExt;
        let mut bytes = path.as_ref().as_os_str().as_bytes().to_vec();
        bytes.push(0);
        // SAFETY: `bytes` is a valid NUL-terminated C string for the call.
        let raw = unsafe { libc::open(bytes.as_ptr().cast(), libc::O_RDWR | libc::O_CLOEXEC) };
        if raw < 0 {
            return Err(Error::last_os("open(/dev/etN_ops)"));
        }
        // SAFETY: `raw` is a freshly opened, owned file descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        Ok(Self {
            fd,
            max_msg: Cell::new(None),
        })
    }

    /// Wrap an already-open operations node file descriptor.
    ///
    /// Ownership of `fd` is transferred to the returned transport.
    pub fn from_owned_fd(fd: OwnedFd) -> Self {
        Self {
            fd,
            max_msg: Cell::new(None),
        }
    }

    fn raw(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    fn cached_max_msg(&self) -> Result<usize> {
        if let Some(v) = self.max_msg.get() {
            return Ok(v);
        }
        let v = self.sq_max_msg_size()? as usize;
        self.max_msg.set(Some(v));
        Ok(v)
    }

    /// Wait for `events` (a `poll(2)` event mask) on the device fd.
    fn poll(&self, events: libc::c_short, timeout: Duration) -> Result<bool> {
        let mut pfd = libc::pollfd {
            fd: self.raw(),
            events,
            revents: 0,
        };
        let millis = timeout.as_millis().min(libc::c_int::MAX as u128) as libc::c_int;
        // SAFETY: single valid pollfd, count 1.
        let rc = unsafe { libc::poll(&mut pfd, 1, millis) };
        if rc < 0 {
            return Err(Error::last_os("poll"));
        }
        Ok(rc > 0 && (pfd.revents & events) != 0)
    }
}

impl Transport for IoctlTransport {
    fn dram_info(&self) -> Result<DramInfo> {
        // SAFETY: zeroed POD filled by a READ ioctl of matching size.
        let mut info: ops::dram_info = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a live `dram_info` for the encoded size.
        unsafe {
            ioctl::ioctl(
                self.raw(),
                ioctl::GET_USER_DRAM_INFO,
                (&raw mut info).cast(),
                "GET_USER_DRAM_INFO",
            )?;
        }
        Ok(DramInfo {
            base: info.base,
            size: info.size,
            dma_max_elem_size: info.dma_max_elem_size,
            dma_max_elem_count: info.dma_max_elem_count,
            // The driver's `align_in_bits` field is a byte quantum (e.g. 64),
            // not a shift amount; carry it through as such.
            dma_alignment: info.align_in_bits,
        })
    }

    fn fw_update(&self, image: &[u8]) -> Result<()> {
        // The driver reads `size` bytes from `ubuf`; the buffer must outlive the
        // call, which it does for the borrow's duration.
        let mut desc = ops::fw_update_desc {
            ubuf: image.as_ptr() as *mut libc::c_void,
            size: image.len() as u64,
        };
        // SAFETY: descriptor points at a live, correctly sized image buffer.
        unsafe {
            ioctl::ioctl(
                self.raw(),
                ioctl::FW_UPDATE,
                (&raw mut desc).cast(),
                "FW_UPDATE",
            )?;
        }
        Ok(())
    }

    fn sq_count(&self) -> Result<u16> {
        ioctl::read_scalar::<u16>(self.raw(), ioctl::GET_SQ_COUNT, "GET_SQ_COUNT")
    }

    fn sq_max_msg_size(&self) -> Result<u16> {
        ioctl::read_scalar::<u16>(
            self.raw(),
            ioctl::GET_SQ_MAX_MSG_SIZE,
            "GET_SQ_MAX_MSG_SIZE",
        )
    }

    fn push_sq(&self, sq_index: u16, cmd: &[u8], flags: u8) -> Result<bool> {
        // Zero-initialise so any tail padding the driver copies is defined.
        let mut desc: ops::cmd_desc = unsafe { std::mem::zeroed() };
        desc.cmd = cmd.as_ptr() as *mut libc::c_void;
        desc.size = cmd.len() as u16;
        desc.sq_index = sq_index;
        desc.flags = flags;
        // SAFETY: descriptor points at the live `cmd` buffer for the call.
        let rc = unsafe {
            libc::ioctl(
                self.raw(),
                ioctl::PUSH_SQ,
                (&raw mut desc).cast::<libc::c_void>(),
            )
        };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            // A full submission queue is reported as a transient errno rather
            // than a hard failure; the caller should retry after `wait_sq`.
            return match err.raw_os_error() {
                Some(libc::EAGAIN) | Some(libc::ENOSPC) | Some(libc::ENOBUFS) => Ok(false),
                _ => Err(Error::Io {
                    op: "PUSH_SQ",
                    source: err,
                }),
            };
        }
        Ok(true)
    }

    fn pop_cq(&self) -> Result<Option<PoppedResponse>> {
        let cap = self.cached_max_msg()?.max(8);
        let mut buf = vec![0u8; cap];
        let mut desc = ops::rsp_desc {
            rsp: buf.as_mut_ptr() as *mut libc::c_void,
            size: cap as u16,
            cq_index: 0,
        };
        // SAFETY: descriptor points at the live, `cap`-byte `buf`.
        let rc = unsafe {
            libc::ioctl(
                self.raw(),
                ioctl::POP_CQ,
                (&raw mut desc).cast::<libc::c_void>(),
            )
        };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            // An empty completion queue is reported as EAGAIN; treat as "no
            // response yet" rather than an error.
            return match err.raw_os_error() {
                Some(libc::EAGAIN) | Some(libc::ENOMSG) => Ok(None),
                _ => Err(Error::Io {
                    op: "POP_CQ",
                    source: err,
                }),
            };
        }
        // The number of bytes popped is reported via the ioctl return value when
        // positive, otherwise via the descriptor's updated `size`. A zero-length
        // result means the queue was empty.
        let len = if rc > 0 {
            rc as usize
        } else {
            desc.size as usize
        };
        if len == 0 {
            return Ok(None);
        }
        buf.truncate(len.min(cap));
        Ok(Some(PoppedResponse {
            bytes: buf,
            cq_index: desc.cq_index,
        }))
    }

    fn extract_trace(&self, trace_type: u8) -> Result<Vec<u8>> {
        // Query the buffer size first; the trace descriptor itself carries no
        // length, so the host must allocate exactly what the driver expects.
        let mut ty = trace_type;
        // SAFETY: `ty` is a live u8; the request encodes sizeof(u8).
        let size = unsafe {
            ioctl::ioctl(
                self.raw(),
                ioctl::GET_TRACE_BUFFER_SIZE,
                (&raw mut ty).cast(),
                "GET_TRACE_BUFFER_SIZE",
            )?
        } as usize;

        let mut buf = vec![0u8; size];
        // Zero-initialise so any tail padding the driver copies is defined.
        let mut desc: ops::trace_desc = unsafe { std::mem::zeroed() };
        desc.trace_type = trace_type;
        desc.buf = buf.as_mut_ptr() as *mut libc::c_void;
        // SAFETY: descriptor points at the live, `size`-byte `buf`.
        unsafe {
            ioctl::ioctl(
                self.raw(),
                ioctl::EXTRACT_TRACE_BUFFER,
                (&raw mut desc).cast(),
                "EXTRACT_TRACE_BUFFER",
            )?;
        }
        Ok(buf)
    }

    fn wait_cq(&self, timeout: Duration) -> Result<bool> {
        self.poll(libc::POLLIN, timeout)
    }

    fn wait_sq(&self, timeout: Duration) -> Result<bool> {
        self.poll(libc::POLLOUT, timeout)
    }

    fn dma_host_buffer(&self, size: usize) -> Result<Box<dyn DmaHostBuffer>> {
        // The driver DMAs only to/from its own CMA region, which is obtained by
        // mmap-ing the operations node (each mapping is a fresh CMA allocation).
        // It resolves the bus address from the mapped virtual address itself, so
        // the DMA node's physical field is left 0. This mirrors DevicePcie's
        // allocDmaBuffer: mmap(NULL, size, RW, MAP_SHARED, fd, 0).
        let len = size.max(1);
        // SAFETY: standard mmap of `len` bytes from the device fd.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.raw(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::last_os("mmap(DMA buffer)"));
        }
        Ok(Box::new(MmapDmaBuffer {
            ptr: ptr.cast(),
            len,
        }))
    }
}

/// A DMA host buffer mapped from the driver's CMA pool. Its mapped virtual
/// address is DMA-capable; the driver resolves the bus address, so the physical
/// field is reported as 0.
struct MmapDmaBuffer {
    ptr: *mut u8,
    len: usize,
}

impl DmaHostBuffer for MmapDmaBuffer {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` maps `len` writable bytes for the buffer's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` maps `len` readable bytes for the buffer's lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn virt_addr(&self) -> u64 {
        self.ptr as u64
    }
    fn phys_addr(&self) -> u64 {
        0
    }
}

impl Drop for MmapDmaBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` came from the mmap above, unmapped once.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}
