//! Shared host/device ABI for the ET-SoC-1: the kernel-launch argument structs,
//! defined **once** and used by both the host launcher and the device kernel.
//!
//! Kernel arguments are passed by pointer: the host stages an argument struct in
//! device memory and the firmware delivers its address to the kernel (in `a0`).
//! Because both the host (x86-64) and the device (RV64) are little-endian, the
//! in-memory `#[repr(C)]` layout *is* the wire layout, so no explicit
//! serialisation is needed -- the host takes the struct's bytes and the kernel
//! reinterprets the pointer. Defining each struct here keeps the two sides from
//! drifting (mismatched field order, sizes, or padding).

#![no_std]

/// ET-SoC-1 cache-line size, in bytes.
///
/// Per-hart outputs are laid out at this stride on both sides: the host strides
/// its padded arrays by it and the device writes each hart's cell at
/// `base + hart * CACHE_LINE`. Defining it once here keeps the two from drifting,
/// which on this software-coherent part would cause silent false-sharing
/// corruption.
pub const CACHE_LINE: usize = 64;

/// A wrapper that aligns `T` to a cache-line boundary.
///
/// On the ET-SoC-1 (a software-coherent architecture), two values sharing a
/// cache line that are written by distinct harts without explicit cache
/// operations cause false-sharing corruption. Wrapping per-hart output data
/// in `CachePadded` ensures each instance occupies a distinct 64-byte line,
/// making cross-hart false sharing structurally impossible regardless of the
/// surrounding allocation layout.
///
/// The inner value is accessed directly via the public tuple field `0`.
///
/// # Example
///
/// ```
/// use et_abi::CachePadded;
/// let cell: CachePadded<u64> = CachePadded(0);
/// assert_eq!(core::mem::align_of::<CachePadded<u64>>(), 64);
/// ```
#[repr(align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CachePadded<T>(pub T);

/// Harts per compute shire on the ET-SoC-1 (architectural constant).
pub const HARTS_PER_SHIRE: u32 = 64;

/// Harts per neighbourhood on the ET-SoC-1 (architectural constant).
pub const HARTS_PER_NEIGHBOURHOOD: u32 = 16;

/// A plain-old-data kernel-argument struct exchanged between host and device.
///
/// # Safety
/// Implementors must be `#[repr(C)]`, contain only integer fields with no
/// padding, and be valid for any bit pattern. Then [`DeviceArgs::as_bytes`] and
/// [`DeviceArgs::from_ptr`] are a faithful round-trip on little-endian hosts and
/// devices.
pub unsafe trait DeviceArgs: Sized + Copy {
    /// Borrow the struct as its on-wire bytes (host side: stage these in device
    /// memory as the launch arguments).
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `Self` is repr(C) POD (trait contract), so its bytes are a
        // valid representation of length `size_of::<Self>()`.
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const u8,
                core::mem::size_of::<Self>(),
            )
        }
    }

    /// Reinterpret a device-memory pointer as these arguments (device side).
    ///
    /// # Safety
    /// `ptr` must point to at least `size_of::<Self>()` bytes of a valid,
    /// suitably aligned instance -- e.g. the launch-args region the firmware
    /// passed in `a0`.
    unsafe fn from_ptr<'a>(ptr: *const u8) -> &'a Self {
        // SAFETY: forwarded to the caller's contract on `ptr`.
        unsafe { &*(ptr as *const Self) }
    }
}

// ---------------------------------------------------------------------------
// Tensor-extension constants
// ---------------------------------------------------------------------------

/// Required alignment for all matrix pointers and row strides used with the
/// ET-SoC-1 tensor-load/store instructions. TensorLoad and TensorStore each
/// require the source or destination address to be 64-byte aligned.
pub const TENSOR_ALIGN: usize = 64;

/// Number of addressable cache lines in each Minion's L1 scratchpad.
/// TensorLoad START field is 6 bits, spanning lines 0..47 inclusive.
pub const SCP_LINES: usize = 48;

/// Bytes per L1 scratchpad line (one cache line).
pub const SCP_LINE_BYTES: usize = 64;

/// Minion cores per compute shire on the ET-SoC-1.
/// Each shire has 32 dual-threaded Minion cores (64 harts total).
pub const MINIONS_PER_SHIRE: u32 = 32;

// ---------------------------------------------------------------------------
// GEMM tile dimensions
// ---------------------------------------------------------------------------

/// Number of C output rows computed per tile by TensorFMA32.
/// Equals the maximum AROWS+1 value (4-bit field, max 15 -> 16 rows).
pub const GEMM_TILE_M: usize = 16;

/// Inner-dimension (K) slice processed per TensorFMA32 call.
/// Limited to 16 f32 values per A-matrix row fitting in one 64-byte
/// scratchpad line (ACOLS field is 4-bit, max 15 -> 16 columns).
pub const GEMM_TILE_K: usize = 16;

/// Number of C output columns computed per tile.
/// With BCOLS=3 (the maximum 2-bit field value), TensorFMA32 produces
/// 4*(BCOLS+1) = 16 output f32 columns per row, filling two 256-bit FP
/// registers per C row and requiring exactly 64 bytes per row in memory.
/// N must be a multiple of this value.
pub const GEMM_TILE_N: usize = 16;

// ---------------------------------------------------------------------------
// GemmArgs
// ---------------------------------------------------------------------------

/// Arguments for the single-precision general matrix multiplication (sGEMM)
/// kernel (`sgemm-rs`), implementing C = alpha*A*B + beta*C.
///
/// # Layout invariants (v0.1 restrictions)
/// - `alpha` must be `1.0` and `beta` must be `0.0`.
/// - `n` must be a multiple of [`GEMM_TILE_N`] (16).
/// - `a`, `b`, `c` must be [`TENSOR_ALIGN`]-byte aligned device addresses.
/// - `lda`, `ldb`, `ldc` must be multiples of [`TENSOR_ALIGN`] (64 bytes).
///
/// All dimensions are in elements; leading dimensions are in bytes.
///
/// # ABI layout
/// The four 8-byte fields (`a`, `b`, `c`, `n_shires`) are grouped first to
/// give the struct 8-byte alignment with no internal or trailing padding:
/// `4*8 + 8*4 = 64 bytes` total.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmArgs {
    /// Device address of A [M x K], row-major, 64-byte aligned.
    pub a:        u64,
    /// Device address of B [K x N], row-major, 64-byte aligned.
    pub b:        u64,
    /// Device address of C [M x N], row-major, 64-byte aligned.
    pub c:        u64,
    /// Number of participating compute shires. Stored as `u64` to keep
    /// all 8-byte fields contiguous and the total struct size a multiple
    /// of the struct's 8-byte alignment. Effective range: 1..=34.
    pub n_shires: u64,
    /// Number of rows of A and C (M dimension).
    pub m:        u32,
    /// Number of columns of B and C (N dimension; must be a multiple of 16).
    pub n:        u32,
    /// Shared inner dimension (K): columns of A and rows of B.
    pub k:        u32,
    /// Row stride of A in bytes (multiple of 64).
    pub lda:      u32,
    /// Row stride of B in bytes (multiple of 64).
    pub ldb:      u32,
    /// Row stride of C in bytes (multiple of 64).
    pub ldc:      u32,
    /// A*B scaling factor. Must be `1.0` in v0.1.
    pub alpha:    f32,
    /// C scaling factor. Must be `0.0` in v0.1.
    pub beta:     f32,
}

// SAFETY: repr(C); 4 u64 fields followed by 8 u32/f32 fields, ordered by
// decreasing size -> no padding. 4*8 + 8*4 = 64 bytes, a multiple of the
// struct's 8-byte alignment.
unsafe impl DeviceArgs for GemmArgs {}
const _: () = assert!(core::mem::size_of::<GemmArgs>() == 64);

/// Arguments for the data-parallel reduction kernel (`reduce-rs`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReduceArgs {
    /// Device address of the input array (`n` × `u32`).
    pub input: u64,
    /// Device address of the output array (`n_harts` × one `u64` per cache line).
    pub out: u64,
    /// Number of input elements.
    pub n: u32,
    /// Number of participating harts.
    pub n_harts: u32,
}

// SAFETY: repr(C), only u64/u32 fields ordered by decreasing size -> no padding.
unsafe impl DeviceArgs for ReduceArgs {}
const _: () = assert!(core::mem::size_of::<ReduceArgs>() == 24);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_args_size() {
        // 4 u64 + 8 u32/f32 = 32 + 32 = 64 bytes, a multiple of 8.
        assert_eq!(core::mem::size_of::<GemmArgs>(), 64);
    }

    #[test]
    fn gemm_args_roundtrip() {
        let a = GemmArgs {
            a:        0x0080_0100_0000,
            b:        0x0080_0200_0000,
            c:        0x0080_0300_0000,
            n_shires: 4,
            m:        128,
            n:        64,
            k:        256,
            lda:      1024,   // 256 * 4 bytes, 64-byte aligned
            ldb:      256,    // 64 * 4 bytes, 64-byte aligned
            ldc:      256,    // 64 * 4 bytes, 64-byte aligned
            alpha:    1.0,
            beta:     0.0,
        };
        let bytes = a.as_bytes();
        assert_eq!(bytes.len(), 64);
        let b = unsafe { GemmArgs::from_ptr(bytes.as_ptr()) };
        assert_eq!(*b, a);
        // Verify leading dimensions are 64-byte aligned as the kernel requires.
        assert_eq!(a.lda as usize % TENSOR_ALIGN, 0);
        assert_eq!(a.ldb as usize % TENSOR_ALIGN, 0);
        assert_eq!(a.ldc as usize % TENSOR_ALIGN, 0);
        // Verify N is a multiple of GEMM_TILE_N.
        assert_eq!(a.n as usize % GEMM_TILE_N, 0);
    }

    #[test]
    fn reduce_args_roundtrip() {
        let a = ReduceArgs {
            input: 0x0080_0580_1000,
            out: 0x0080_0590_0000,
            n: 262_144,
            n_harts: 64,
        };
        let bytes = a.as_bytes();
        assert_eq!(bytes.len(), 24);
        // The device would do exactly this from the args pointer.
        let b = unsafe { ReduceArgs::from_ptr(bytes.as_ptr()) };
        assert_eq!(*b, a);
    }
}
