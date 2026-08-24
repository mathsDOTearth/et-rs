//! Host-side BLAS launcher for the ET-SoC-1 tensor extension.
//!
//! This module wraps the `sgemm-rs` device kernel with a typed, validated
//! host API. The entry point is [`sgemm`], which validates arguments, stages
//! a [`GemmArgs`] struct in device memory, and launches the kernel via
//! [`Device::launch_spmd`].
//!
//! # Conventions
//!
//! - All matrices are row-major (C order).
//! - Leading dimensions are supplied in bytes (not element counts) to match
//!   the hardware's byte-addressed stride interface.
//! - v0.1 supports only `alpha = 1.0, beta = 0.0`; other values return
//!   [`GemmError::UnsupportedScaling`].
//!
//! # Example
//!
//! ```no_run
//! use et_soc1::{Device, Result};
//! use et_soc1::blas::{GemmError, sgemm};
//!
//! fn main() -> Result<()> {
//!     let dev    = Device::open(0)?;
//!     let kernel = dev.load_kernel(&std::fs::read("sgemm-rs.elf").unwrap())?;
//!
//!     let m = 128_usize;
//!     let n = 64_usize;
//!     let k = 256_usize;
//!     // Row strides in bytes: N columns of f32 each.
//!     let lda = (k * 4).next_multiple_of(64) as u32;
//!     let ldb = (n * 4).next_multiple_of(64) as u32;
//!     let ldc = (n * 4).next_multiple_of(64) as u32;
//!
//!     let a_buf = dev.alloc_array::<f32>(m * k)?;
//!     let b_buf = dev.alloc_array::<f32>(k * n)?;
//!     let c_buf = dev.alloc_array::<f32>(m * n)?;
//!
//!     sgemm(
//!         &dev, &kernel,
//!         m as u32, n as u32, k as u32,
//!         1.0, a_buf.addr(), lda,
//!              b_buf.addr(), ldb,
//!         0.0, c_buf.addr(), ldc,
//!         1,   // n_shires
//!     )?;
//!     Ok(())
//! }
//! ```

use et_abi::{GemmArgs, GEMM_TILE_N, TENSOR_ALIGN};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::transport::Transport;

// Re-export LoadedKernel so callers can use it without importing device directly.
use crate::device::LoadedKernel;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to the BLAS launcher, returned as [`Error::Limit`] strings.
/// These are surfaced via [`sgemm`] and wrapped in the host crate's [`Error`].
///
/// They are listed here for documentation; callers match on [`Error::Limit`].
#[derive(Debug, Clone, PartialEq)]
pub enum GemmError {
    /// At least one matrix pointer is not 64-byte aligned.
    UnalignedPointer {
        matrix: &'static str,
        addr: u64,
    },
    /// A leading dimension is not a multiple of [`TENSOR_ALIGN`] (64 bytes).
    UnalignedStride {
        matrix: &'static str,
        lda: u32,
    },
    /// `N` is not a multiple of [`GEMM_TILE_N`] (16).
    ///
    /// Deprecated in v0.4.0: the kernel now handles arbitrary N. This variant
    /// is retained for source compatibility and will be removed in a future
    /// major release. [`sgemm`] no longer returns this error.
    #[deprecated(since = "0.4.0", note = "sgemm now accepts arbitrary N; \
                                          this error variant is never returned")]
    NNotMultipleOfTileN { n: u32 },
    /// One or more dimensions are zero, which is invalid.
    ZeroDimension,
    /// `alpha != 1.0` or `beta != 0.0`; v0.1 supports only this combination.
    UnsupportedScaling { alpha: f32, beta: f32 },
}

impl GemmError {
    fn into_limit(self) -> Error {
        Error::Limit(match self {
            GemmError::UnalignedPointer { matrix, addr } => format!(
                "sGEMM: matrix {matrix} pointer {addr:#x} is not \
                 {TENSOR_ALIGN}-byte aligned"
            ),
            GemmError::UnalignedStride { matrix, lda } => format!(
                "sGEMM: leading dimension of {matrix} ({lda} bytes) is not a \
                 multiple of {TENSOR_ALIGN}"
            ),
            #[allow(deprecated)]
            GemmError::NNotMultipleOfTileN { n } => format!(
                "sGEMM: N={n} is not a multiple of {GEMM_TILE_N} (deprecated \
                 constraint; sgemm now accepts arbitrary N)"
            ),
            GemmError::ZeroDimension => {
                "sGEMM: M, N, and K must all be >= 1".into()
            }
            GemmError::UnsupportedScaling { alpha, beta } => format!(
                "sGEMM: alpha={alpha}, beta={beta} not supported in v0.1 \
                 (only alpha=1.0, beta=0.0)"
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Matrix allocation helper
// ---------------------------------------------------------------------------

/// Allocate a device buffer for an [rows x cols] row-major f32 matrix whose
/// row stride is padded to a 64-byte boundary.
///
/// Returns `(device_addr, lda_bytes)`. The allocation uses the device bump
/// allocator. For a typical GPU-style sGEMM, allocate A, B, and C with this
/// function, upload host data, call [`sgemm`], then download C.
///
/// # Example
/// ```no_run
/// # use et_soc1::{Device, transport::IoctlTransport};
/// # let dev: Device<IoctlTransport> = Device::open(0).unwrap();
/// let (addr, lda) = et_soc1::blas::alloc_tensor_matrix(&dev, 128, 256).unwrap();
/// // addr is 64-byte aligned; lda = ceil(256 * 4, 64) = 1024 bytes.
/// ```
pub fn alloc_tensor_matrix<Tr: Transport>(
    dev: &Device<Tr>,
    rows: usize,
    cols: usize,
) -> Result<(u64, u32)> {
    // Row stride in bytes, padded up to a 64-byte boundary.
    let row_bytes = (cols * 4).next_multiple_of(TENSOR_ALIGN);
    let total     = (rows * row_bytes) as u64;
    let region    = dev.alloc(total)?;
    let addr      = region.addr;
    if addr % TENSOR_ALIGN as u64 != 0 {
        // The device bump allocator should align to at least TENSOR_ALIGN;
        // if it does not, the allocation is unusable for tensor ops.
        return Err(Error::Limit(format!(
            "device allocator returned address {addr:#x} with insufficient \
             alignment for tensor operations (need {TENSOR_ALIGN} bytes)"
        )));
    }
    Ok((addr, row_bytes as u32))
}

// ---------------------------------------------------------------------------
// sgemm
// ---------------------------------------------------------------------------

/// Launch the `sgemm-rs` device kernel to compute C = alpha*A*B + beta*C.
///
/// # Parameters
/// - `dev`:      open device handle.
/// - `kernel`:   loaded `sgemm-rs` ELF image (from [`Device::load_kernel`]).
/// - `m`, `n`, `k`: matrix dimensions (M rows of A/C, N columns of B/C,
///   K inner dimension). All must be >= 1. N may be any positive integer.
/// - `alpha`:    scaling factor for A*B (must be 1.0 in v0.1).
/// - `a`:        device address of A [M x K] (64-byte aligned).
/// - `lda`:      row stride of A in bytes (multiple of 64).
/// - `b`:        device address of B [K x N] (64-byte aligned).
/// - `ldb`:      row stride of B in bytes (multiple of 64).
/// - `beta`:     scaling factor for C (must be 0.0 in v0.1).
/// - `c`:        device address of C [M x N] (64-byte aligned).
/// - `ldc`:      row stride of C in bytes (multiple of 64).
/// - `n_shires`: number of compute shires to use. Each shire contributes
///   32 Minion workers, so `n_shires * 32` tiles execute concurrently.
///   Use [`topology::Topology::n_shires`] for the full device parallelism,
///   or a smaller value for testing.
///
/// # Errors
/// Returns [`Error::Limit`] for any alignment or dimension constraint
/// violation (see [`GemmError`]), or propagates device/transport errors.
#[allow(clippy::too_many_arguments)]
pub fn sgemm<Tr: Transport>(
    dev:      &Device<Tr>,
    kernel:   &LoadedKernel,
    m:        u32,
    n:        u32,
    k:        u32,
    alpha:    f32,
    a:        u64,
    lda:      u32,
    b:        u64,
    ldb:      u32,
    beta:     f32,
    c:        u64,
    ldc:      u32,
    n_shires: u32,
) -> Result<()> {
    // --- Argument validation ------------------------------------------------

    if m == 0 || n == 0 || k == 0 {
        return Err(GemmError::ZeroDimension.into_limit());
    }

    if alpha != 1.0 || beta != 0.0 {
        return Err(GemmError::UnsupportedScaling { alpha, beta }.into_limit());
    }

    for (name, addr) in [("A", a), ("B", b), ("C", c)] {
        if addr % TENSOR_ALIGN as u64 != 0 {
            return Err(
                GemmError::UnalignedPointer { matrix: name, addr }.into_limit()
            );
        }
    }

    for (name, stride) in [("A", lda), ("B", ldb), ("C", ldc)] {
        if !(stride as usize).is_multiple_of(TENSOR_ALIGN) {
            return Err(
                GemmError::UnalignedStride { matrix: name, lda: stride }
                    .into_limit()
            );
        }
    }

    // --- Launch -------------------------------------------------------------

    let args = GemmArgs {
        a,
        b,
        c,
        n_shires: n_shires as u64,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        alpha,
        beta,
    };

    // The kernel is launched on all harts of the participating shires
    // (n_shires * 64 harts). Inside the kernel, only primary harts
    // (mhartid & 1 == 0) do tensor work; companion harts return early.
    let shire_mask = (1_u64 << n_shires) - 1;
    dev.launch_spmd(kernel, shire_mask, &args)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use et_abi::DeviceArgs;

    fn dummy_args() -> GemmArgs {
        GemmArgs {
            a:        0x0080_0000_0000,
            b:        0x0080_0100_0000,
            c:        0x0080_0200_0000,
            n_shires: 1,
            m:        32,
            n:        16,
            k:        16,
            lda:      64,
            ldb:      64,
            ldc:      64,
            alpha:    1.0,
            beta:     0.0,
        }
    }

    #[test]
    fn gemmargs_size_consistent() {
        // 4 u64 + 8 u32/f32 = 32 + 32 = 64 bytes.
        assert_eq!(core::mem::size_of::<GemmArgs>(), 64);
    }

    #[test]
    fn validates_alpha_beta() {
        // alpha=2.0 should produce a Limit error.
        let err = GemmError::UnsupportedScaling { alpha: 2.0, beta: 0.0 }.into_limit();
        assert!(matches!(err, Error::Limit(_)));
    }

    #[test]
    #[allow(deprecated)]
    fn n_not_multiple_error_formats() {
        // NNotMultipleOfTileN is deprecated (sgemm no longer returns it) but
        // retained for source compatibility; verify the message still formats.
        let err = GemmError::NNotMultipleOfTileN { n: 7 }.into_limit();
        assert!(matches!(err, Error::Limit(ref s) if s.contains("N=7")));
    }

    #[test]
    fn validates_pointer_alignment() {
        let err = GemmError::UnalignedPointer {
            matrix: "A",
            addr:   0x0080_0000_0001,
        }
        .into_limit();
        assert!(matches!(err, Error::Limit(ref s) if s.contains("matrix A")));
    }

    #[test]
    fn validates_stride_alignment() {
        let err = GemmError::UnalignedStride { matrix: "B", lda: 63 }.into_limit();
        assert!(matches!(err, Error::Limit(ref s) if s.contains("B")));
    }

    #[test]
    fn gemmargs_roundtrip() {
        let a = dummy_args();
        let b = unsafe { GemmArgs::from_ptr(a.as_bytes().as_ptr()) };
        assert_eq!(*b, a);
    }
}
