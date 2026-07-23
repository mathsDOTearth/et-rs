//! ET-SoC-1 character-device ioctl request codes and thin syscall wrappers.
//!
//! The request codes are reconstructed from `et_ioctl.h` using the standard
//! Linux `asm-generic` ioctl encoding. bindgen cannot evaluate the
//! function-like `_IOR`/`_IOW`/`_IOWR` macros, so the magic number, ordinal and
//! direction of every command are transcribed here (each cross-referenced to its
//! header line) while the *size* field is derived from the bindgen-generated
//! descriptor structs. This guarantees the encoded size always tracks the ABI
//! that bindgen sees, even if a descriptor changes.
//!
//! The full request-code table from `et_ioctl.h` is transcribed here for
//! completeness and future use; some codes are not yet exercised by the safe
//! API, hence the module-level `dead_code` allowance.
#![allow(dead_code)]

use crate::error::{Error, Result};
use crate::ffi::ops;
use core::mem::size_of;
use std::os::fd::RawFd;

// --- asm-generic ioctl encoding (from <asm-generic/ioctl.h>) ---

const IOC_NRBITS: u32 = 8;
const IOC_TYPEBITS: u32 = 8;
const IOC_SIZEBITS: u32 = 14;

const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;

/// Data-transfer direction, from the perspective of userspace.
const IOC_NONE: u32 = 0;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;

/// The Esperanto PCIe ioctl magic (`ESPERANTO_PCIE_IOCTL_MAGIC`, 0xE7).
const MAGIC: u32 = ops::ESPERANTO_PCIE_IOCTL_MAGIC;

/// Encode an ioctl request code, mirroring the kernel `_IOC` macro.
const fn ioc(dir: u32, nr: u32, size: usize) -> libc::c_ulong {
    debug_assert!(size <= (1 << IOC_SIZEBITS));
    ((dir << IOC_DIRSHIFT)
        | (MAGIC << IOC_TYPESHIFT)
        | (nr << IOC_NRSHIFT)
        | ((size as u32) << IOC_SIZESHIFT)) as libc::c_ulong
}

// Request codes. The (direction, ordinal, size-type) of each entry mirrors the
// corresponding `#define` in et_ioctl.h.

/// `_IOR(MAGIC, 1, struct dram_info)`
pub const GET_USER_DRAM_INFO: libc::c_ulong = ioc(IOC_READ, 1, size_of::<ops::dram_info>());
/// `_IOW(MAGIC, 2, struct fw_update_desc)`
pub const FW_UPDATE: libc::c_ulong = ioc(IOC_WRITE, 2, size_of::<ops::fw_update_desc>());
/// `_IOR(MAGIC, 3, __u16)`
pub const GET_SQ_COUNT: libc::c_ulong = ioc(IOC_READ, 3, size_of::<u16>());
/// `_IOR(MAGIC, 4, __u16)`
pub const GET_SQ_MAX_MSG_SIZE: libc::c_ulong = ioc(IOC_READ, 4, size_of::<u16>());
/// `_IOR(MAGIC, 5, struct dev_config)`
pub const GET_DEVICE_CONFIGURATION: libc::c_ulong = ioc(IOC_READ, 5, size_of::<ops::dev_config>());
/// `_IOW(MAGIC, 6, struct cmd_desc)`
pub const PUSH_SQ: libc::c_ulong = ioc(IOC_WRITE, 6, size_of::<ops::cmd_desc>());
/// `_IOWR(MAGIC, 7, struct rsp_desc)`
pub const POP_CQ: libc::c_ulong = ioc(IOC_READ | IOC_WRITE, 7, size_of::<ops::rsp_desc>());
/// `_IOR(MAGIC, 8, __u64)`
pub const GET_SQ_AVAIL_BITMAP: libc::c_ulong = ioc(IOC_READ, 8, size_of::<u64>());
/// `_IOR(MAGIC, 9, __u64)`
pub const GET_CQ_AVAIL_BITMAP: libc::c_ulong = ioc(IOC_READ, 9, size_of::<u64>());
/// `_IOW(MAGIC, 10, struct sq_threshold)`
pub const SET_SQ_THRESHOLD: libc::c_ulong = ioc(IOC_WRITE, 10, size_of::<ops::sq_threshold>());
/// `_IOW(MAGIC, 11, __u8)`
pub const GET_TRACE_BUFFER_SIZE: libc::c_ulong = ioc(IOC_WRITE, 11, size_of::<u8>());
/// `_IOWR(MAGIC, 12, struct trace_desc)`
pub const EXTRACT_TRACE_BUFFER: libc::c_ulong =
    ioc(IOC_READ | IOC_WRITE, 12, size_of::<ops::trace_desc>());
/// `_IOR(MAGIC, 13, __u32)`
pub const GET_DEVICE_STATE: libc::c_ulong = ioc(IOC_READ, 13, size_of::<u32>());

/// Invoke `ioctl(fd, request, argp)`, mapping a negative return to [`Error::Io`].
///
/// # Safety
/// `argp` must point to a region valid for the access implied by `request`'s
/// direction and size, for the duration of the call.
pub(crate) unsafe fn ioctl(
    fd: RawFd,
    request: libc::c_ulong,
    argp: *mut libc::c_void,
    op: &'static str,
) -> Result<libc::c_int> {
    // SAFETY: forwarded to the caller's contract on `argp`.
    let rc = unsafe { libc::ioctl(fd, request, argp) };
    if rc < 0 {
        Err(Error::last_os(op))
    } else {
        Ok(rc)
    }
}

/// Read a plain scalar out of the device via a `_IOR(..., T)` request.
pub(crate) fn read_scalar<T: Copy + Default>(
    fd: RawFd,
    request: libc::c_ulong,
    op: &'static str,
) -> Result<T> {
    let mut value = T::default();
    // SAFETY: `value` is a live `T` and the request encodes `sizeof(T)`.
    unsafe {
        ioctl(fd, request, (&raw mut value).cast(), op)?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values produced by the C preprocessor from `et_ioctl.h` on
    /// x86-64 Linux (`cc -I/opt/et/include`). These pin the encoding so a change
    /// to the descriptor sizes or the request table is caught immediately.
    #[test]
    fn request_codes_match_c_header() {
        assert_eq!(GET_USER_DRAM_INFO, 2149115649);
        assert_eq!(FW_UPDATE, 1074849538);
        assert_eq!(GET_SQ_COUNT, 2147673859);
        assert_eq!(GET_SQ_MAX_MSG_SIZE, 2147673860);
        assert_eq!(GET_DEVICE_CONFIGURATION, 2149639941);
        assert_eq!(PUSH_SQ, 1074849542);
        assert_eq!(POP_CQ, 3222333191);
        assert_eq!(GET_SQ_AVAIL_BITMAP, 2148067080);
        assert_eq!(GET_CQ_AVAIL_BITMAP, 2148067081);
        assert_eq!(SET_SQ_THRESHOLD, 1074063114);
        assert_eq!(GET_TRACE_BUFFER_SIZE, 1073866507);
        assert_eq!(EXTRACT_TRACE_BUFFER, 3222333196);
        assert_eq!(GET_DEVICE_STATE, 2147804941);
    }
}
