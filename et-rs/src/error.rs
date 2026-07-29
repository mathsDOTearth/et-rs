//! Error and result types for the crate.

use std::fmt;

use crate::proto::KernelErrorPtr;

/// The result type returned throughout `et_soc1`.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors surfaced by the host-side interface.
#[derive(Debug)]
pub enum Error {
    /// A `libc` call (open, ioctl, poll, ...) failed. Carries the operation
    /// name and the underlying `errno`.
    Io {
        /// The syscall or ioctl that failed, for diagnostic context.
        op: &'static str,
        /// The underlying operating-system error.
        source: std::io::Error,
    },

    /// The device firmware returned a non-success status code in a command
    /// response. `code` is the raw `dev_ops_api_*_response_e` value.
    Device {
        /// The command family that reported the failure.
        command: &'static str,
        /// The raw device status code.
        code: u32,
    },

    /// A kernel launch did not complete successfully. `status` is the raw
    /// `dev_ops_api_kernel_launch_response_e` code and `status_name` its symbolic
    /// name (e.g. `"EXCEPTION"`). When the firmware appended diagnostics, `detail`
    /// carries the faulting shire mask and the device addresses of the U-mode
    /// exception and trace buffers.
    KernelLaunch {
        /// The raw kernel-launch status code.
        status: u32,
        /// Symbolic name of the status code, or `"UNKNOWN"`.
        status_name: &'static str,
        /// Appended exception/trace pointers, if the device provided them.
        detail: Option<KernelErrorPtr>,
    },

    /// A device response could not be parsed, or did not match the command it
    /// was expected to answer (wrong `msg_id`, short buffer, mismatched tag).
    Protocol(String),

    /// A request exceeded a device-advertised limit (DMA element size/count,
    /// submission-queue message size, DRAM capacity, ...).
    Limit(String),

    /// The supplied kernel image could not be parsed as a RISC-V ELF.
    Elf(String),

    /// The device DRAM bump allocator has been exhausted.
    OutOfMemory {
        /// Bytes requested by the failing allocation.
        requested: u64,
        /// Bytes still available in the region.
        available: u64,
    },
}

impl Error {
    /// Construct an [`Error::Io`] from the current `errno`.
    pub(crate) fn last_os(op: &'static str) -> Self {
        Error::Io {
            op,
            source: std::io::Error::last_os_error(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { op, source } => write!(f, "{op} failed: {source}"),
            Error::Device { command, code } => {
                write!(f, "device rejected {command} with status {code}")
            }
            Error::KernelLaunch {
                status,
                status_name,
                detail,
            } => {
                write!(f, "kernel-launch failed: {status_name} (status {status})")?;
                if let Some(d) = detail {
                    write!(f, "; faulting shires {:#x}", d.shire_mask)?;
                    if d.exception_buffer != 0 {
                        write!(f, "; exception buffer @ {:#x}", d.exception_buffer)?;
                    }
                    if d.trace_buffer != 0 {
                        write!(f, "; trace buffer @ {:#x}", d.trace_buffer)?;
                    }
                }
                Ok(())
            }
            Error::Protocol(msg) => write!(f, "protocol error: {msg}"),
            Error::Limit(msg) => write!(f, "limit exceeded: {msg}"),
            Error::Elf(msg) => write!(f, "invalid kernel ELF: {msg}"),
            Error::OutOfMemory {
                requested,
                available,
            } => write!(
                f,
                "device DRAM exhausted: requested {requested} bytes, {available} available"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
