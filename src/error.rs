//! Error and result types for the crate.

use std::fmt;

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
