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
// DevicePod
// ---------------------------------------------------------------------------

/// Types that may be copied verbatim between host and device memory.
///
/// # Safety
/// An implementor must be "plain old data": a scalar, or a `#[repr(C)]` struct
/// composed entirely of such types, with no padding bytes and valid for every
/// bit pattern. These invariants make it safe to reinterpret a value (or slice)
/// as its raw bytes and back again, which the upload/download DMA paths rely on.
///
/// The provided implementations cover the standard integer and floating-point
/// scalars. Implement this trait for your own `#[repr(C)]` POD structs to store
/// them in a `DeviceBuffer` or pass them via `upload_slice`.
///
/// Because the trait is defined in `et-abi`, external crates can implement it
/// for their own types without running into the orphan rule.
pub unsafe trait DevicePod: Copy + 'static {}

unsafe impl DevicePod for u8 {}
unsafe impl DevicePod for u16 {}
unsafe impl DevicePod for u32 {}
unsafe impl DevicePod for u64 {}
unsafe impl DevicePod for u128 {}
unsafe impl DevicePod for i8 {}
unsafe impl DevicePod for i16 {}
unsafe impl DevicePod for i32 {}
unsafe impl DevicePod for i64 {}
unsafe impl DevicePod for i128 {}
unsafe impl DevicePod for f32 {}
unsafe impl DevicePod for f64 {}
unsafe impl DevicePod for usize {}
unsafe impl DevicePod for isize {}

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

/// Number of f32 output columns produced per TensorFMA32 call (BCOLS=3 gives
/// 4*(3+1) = 16 columns). Each tile row occupies exactly 64 bytes in the FP
/// register file. N need not be a multiple of this value; the last tile column
/// may be partial, with the hardware writing 64 bytes per row regardless --
/// the caller reads only the N valid columns from the 64-byte-aligned allocation.
pub const GEMM_TILE_N: usize = 16;

// ---------------------------------------------------------------------------
// GemmArgs
// ---------------------------------------------------------------------------

/// Arguments for the single-precision general matrix multiplication (sGEMM)
/// kernel (`sgemm-rs`), implementing C = alpha*A*B + beta*C.
///
/// # Layout invariants (v0.1 restrictions)
/// - `alpha` must be `1.0` and `beta` must be `0.0`.
/// - `n` may be any positive integer; partial last-column tiles are handled
///   transparently via 64-byte-aligned row padding.
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
    pub a: u64,
    /// Device address of B [K x N], row-major, 64-byte aligned.
    pub b: u64,
    /// Device address of C [M x N], row-major, 64-byte aligned.
    pub c: u64,
    /// Number of participating compute shires. Stored as `u64` to keep
    /// all 8-byte fields contiguous and the total struct size a multiple
    /// of the struct's 8-byte alignment. Effective range: 1..=34.
    pub n_shires: u64,
    /// Number of rows of A and C (M dimension).
    pub m: u32,
    /// Number of columns of B and C (N dimension). May be any positive integer;
    /// the last output tile column is partial when N is not a multiple of 16.
    pub n: u32,
    /// Shared inner dimension (K): columns of A and rows of B.
    pub k: u32,
    /// Row stride of A in bytes (multiple of 64).
    pub lda: u32,
    /// Row stride of B in bytes (multiple of 64).
    pub ldb: u32,
    /// Row stride of C in bytes (multiple of 64).
    pub ldc: u32,
    /// A*B scaling factor. Must be `1.0` in v0.1.
    pub alpha: f32,
    /// C scaling factor. Must be `0.0` in v0.1.
    pub beta: f32,
}

// SAFETY: repr(C); 4 u64 fields followed by 8 u32/f32 fields, ordered by
// decreasing size -> no padding. 4*8 + 8*4 = 64 bytes, a multiple of the
// struct's 8-byte alignment.
unsafe impl DeviceArgs for GemmArgs {}
const _: () = assert!(core::mem::size_of::<GemmArgs>() == 64);

/// Arguments for the cache-coherence test kernel (`cache-test-rs`).
///
/// Each primary Minion hart writes its global Minion index to
/// `output[minion_idx]` (stride = 64 bytes, one u32 per cache line), then
/// issues the cache operation selected by `op` and `fence`. The host downloads
/// the padded array and verifies `output[i] == i as u32`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheTestArgs {
    /// Device address of the output buffer:
    /// `n_shires * MINIONS_PER_SHIRE` entries of `u32`, each at stride 64.
    pub output: u64,
    /// Number of participating compute shires (1..=32).
    pub n_shires: u64,
    /// Writeback destination level (`CacheDest` discriminant: 1 = L2, 2 = L3,
    /// 3 = Mem/DDR). Selects how far the writeback propagates so the fault can be
    /// narrowed to the DDR path; only `Mem` is host-DMA visible. Ignored by
    /// [`CACHE_TEST_OP_FLUSH`], which always targets `Mem`.
    pub dest: u64,
    /// Cache operation applied to the written cell: one of
    /// [`CACHE_TEST_OP_WRITEBACK`], [`CACHE_TEST_OP_INVALIDATE`] or
    /// [`CACHE_TEST_OP_FLUSH`].
    pub op: u64,
}

// SAFETY: repr(C), four u64 fields, no padding.
unsafe impl DeviceArgs for CacheTestArgs {}
const _: () = assert!(core::mem::size_of::<CacheTestArgs>() == 32);

/// [`CacheTestArgs::op`]: `cache_writeback_to(dest)` (`flush_va`).
pub const CACHE_TEST_OP_WRITEBACK: u64 = 0;
/// [`CacheTestArgs::op`]: `cache_invalidate_to(dest)` (`evict_va`). Passes only
/// if eviction writes dirty lines back rather than discarding them.
pub const CACHE_TEST_OP_INVALIDATE: u64 = 1;
/// [`CacheTestArgs::op`]: `cache_flush` (`evict_va` to `Mem`).
pub const CACHE_TEST_OP_FLUSH: u64 = 2;

/// Arguments for the data-parallel reduction kernel (`reduce-rs`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReduceArgs {
    /// Device address of the input array (`n` x `u32`).
    pub input: u64,
    /// Device address of the output array (`n_harts` x one `u64` per cache line).
    pub out: u64,
    /// Number of input elements.
    pub n: u32,
    /// Number of participating harts.
    pub n_harts: u32,
}

// SAFETY: repr(C), only u64/u32 fields ordered by decreasing size -> no padding.
unsafe impl DeviceArgs for ReduceArgs {}
const _: () = assert!(core::mem::size_of::<ReduceArgs>() == 24);

/// Bytes of output per Minion written by the tensor-extension test kernel: one
/// cache line for each of its three subtests. Shared by the kernel and the host
/// so the two cannot disagree on the layout.
pub const TENSOR_EXT_TEST_OUT_STRIDE: usize = 3 * CACHE_LINE;

/// Arguments for the tensor-extension instruction test kernel (`tensor-ext-test`).
///
/// All Minions read the same shared input buffers and write their results to
/// per-Minion sections of `output` for independent host verification.
///
/// # Output buffer layout (per Minion, stride [`TENSOR_EXT_TEST_OUT_STRIDE`])
/// Minion `shire * 32 + m` writes at byte offset
/// `(shire * 32 + m) * TENSOR_EXT_TEST_OUT_STRIDE`:
/// - `[0..64)`:   TensorFMA16A32 result: 4 x f32, expected `[5.0, 0.0, 0.0, 0.0]`.
/// - `[64..128)`:  TensorIMA8A32 result: 4 x i32 stored as f32 bit patterns,
///   expected `[4, 0, 0, 0]`.
/// - `[128..192)`: TensorStoreFromScp passthrough: 64 bytes copied verbatim from
///   L1 scratchpad line 0, expected to equal the `a_fp16` buffer contents.
///
/// # Input data
/// Inputs encode the simplest non-trivial tile (AROWS=0, ACOLS=0, BCOLS=0):
/// - `a_fp16`: `[2.0_f16, 3.0_f16, 0..0]` (64 bytes; first 4 bytes used).
/// - `b_fp16`: TenB-interleaved fp16 pairs: `[1.0, 1.0, 0.0, ...]`
///   (64 bytes; first 4 bytes = `b[0,0]=1.0` and `b[1,0]=1.0` for output col 0).
/// - `a_int8`: `[1, 1, 1, 1, 0..0]` (64 bytes; first 4 bytes used).
/// - `b_int8`: IMA8A32-interleaved int8 groups: col 0 = `[1,1,1,1]`, cols 1-3 = 0
///   (64 bytes; first 16 bytes used).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorExtTestArgs {
    /// Base address of the output buffer: `n_shires * 32 *
    /// TENSOR_EXT_TEST_OUT_STRIDE` bytes, 64-byte aligned.
    pub output: u64,
    /// 64-byte-aligned device address of the fp16 A input (64 bytes).
    pub a_fp16: u64,
    /// 64-byte-aligned device address of the fp16 B input in TenB interleaved
    /// format (64 bytes). (PRM TensorFMA16A32: `TenB[k].h[j*2+0] = b[2k,j]`.)
    pub b_fp16: u64,
    /// 64-byte-aligned device address of the int8 A input (64 bytes).
    pub a_int8: u64,
    /// 64-byte-aligned device address of the int8 B input in IMA8A32 interleaved
    /// format (64 bytes). (PRM TensorIMA8A32: word j = `[b[0,j]|b[1,j]|b[2,j]|b[3,j]]`.)
    pub b_int8: u64,
    /// Extent of the output array in shires (1..=32). Cells are indexed by
    /// physical shire number, so this is one more than the highest launched
    /// shire, not the number of launched shires; Minions in shires at or
    /// beyond it do not write.
    pub n_shires: u64,
}

// SAFETY: repr(C), six u64 fields, no padding.
unsafe impl DeviceArgs for TensorExtTestArgs {}
const _: () = assert!(core::mem::size_of::<TensorExtTestArgs>() == 48);

/// Arguments for the PS SIMD instruction verification kernel (`simd-test-rs`).
///
/// Each primary Minion computes `(minion_idx + 1) as f32 * 3.0` in all 16
/// lanes of one C-tile row (FP registers f0 and f1) using the `FBCX.PS`
/// (broadcast) and `FMUL.PS` (element-wise multiply) instructions, and stores
/// the row to its output cell with TensorStore. The host verifies all 16
/// values of every cell.
///
/// # Output buffer layout
/// `n_shires * MINIONS_PER_SHIRE` cells of 16 `f32` (one 64-byte cache line
/// each). Minion `i = shire * 32 + m`, indexed by physical shire, writes the
/// cell at byte offset `i * 64`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdTestArgs {
    /// Device address of the output buffer.
    pub output: u64,
    /// Extent of the output array in shires (1..=32): one more than the highest
    /// launched shire. Minions in shires at or beyond it do not write.
    pub n_shires: u64,
}

// SAFETY: repr(C), two u64 fields, no padding.
unsafe impl DeviceArgs for SimdTestArgs {}
const _: () = assert!(core::mem::size_of::<SimdTestArgs>() == 16);

/// Arguments for the PS transcendental accuracy kernel (`ps-math-test-rs`).
///
/// The input is divided into 64-byte units of 16 `f32` (two PS registers).
/// The primary hart of every launched Minion takes units in grid-stride
/// order, loads each with FLQ2, applies [`PsMathTestArgs::operation`] to both
/// registers, stores the result with FSQ2 to the same offset in the output
/// and writes the line back to DDR. The host compares every lane against a
/// reference.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PsMathTestArgs {
    /// 64-byte-aligned device address of the input array (`count` x `f32`).
    pub input: u64,
    /// 64-byte-aligned device address of the output array (`count` x `f32`).
    pub output: u64,
    /// Number of `f32` elements; a multiple of 16.
    pub count: u64,
    /// Launched shires. Minions are ranked densely over the set bits, so a
    /// mask with gaps still covers every unit.
    pub shire_mask: u32,
    /// One of [`PS_MATH_OP_COPY`], [`PS_MATH_OP_EXP2`], [`PS_MATH_OP_LOG2`]
    /// or [`PS_MATH_OP_RECIPROCAL`].
    pub operation: u32,
}

// SAFETY: repr(C), three u64 and two u32 fields, no padding.
unsafe impl DeviceArgs for PsMathTestArgs {}
const _: () = assert!(core::mem::size_of::<PsMathTestArgs>() == 32);

/// [`PsMathTestArgs::operation`]: FLQ2 then FSQ2 with no arithmetic; verifies
/// the load and store paths bit-exactly.
pub const PS_MATH_OP_COPY: u32 = 0;
/// [`PsMathTestArgs::operation`]: `FEXP.PS` (2^x).
pub const PS_MATH_OP_EXP2: u32 = 1;
/// [`PsMathTestArgs::operation`]: `FLOG.PS` (log2 x).
pub const PS_MATH_OP_LOG2: u32 = 2;
/// [`PsMathTestArgs::operation`]: `FRCP.PS` (1/x).
pub const PS_MATH_OP_RECIPROCAL: u32 = 3;

/// Arguments for the `channel-bench-rs` ping-pong kernel.
///
/// Two Minions exchange messages through device memory under explicit cache
/// maintenance. For each iteration `i` in `1..=iterations` the ping Minion
/// writes `message_bytes` of payload to the forward data area, writes it back
/// to [`cache_level`](Self::cache_level), then writes and writes back the
/// forward flag. The pong Minion polls the forward flag (invalidating it
/// before every read), invalidates and reads the payload, writes its
/// complement to the reply data area and raises the reply flag in the same
/// way. The ping Minion records the round-trip time in cycles.
///
/// `buffer` layout, all offsets from [`buffer`](Self::buffer):
///
/// | Offset | Contents |
/// |---|---|
/// | [`CHANNEL_FORWARD_FLAG_OFFSET`] | forward flag (`u64`, own cache line) |
/// | [`CHANNEL_REPLY_FLAG_OFFSET`] | reply flag (`u64`, own cache line) |
/// | [`CHANNEL_FORWARD_DATA_OFFSET`] | forward payload, `message_bytes` |
/// | [`channel_reply_data_offset`] | reply payload, `message_bytes` |
///
/// `results` layout: the ping and pong headers (one cache line each, `u64`
/// fields indexed by `CHANNEL_HEADER_*`), then one sample per iteration of
/// [`CHANNEL_SAMPLE_WORDS`] `u64`: the round-trip time and the send time
/// (from the first payload write to the completion of the flag writeback),
/// both in cycles of `hpmcounter3`.
///
/// A flag holds `(epoch << 32) | i`, so that a stale line left in a cache by
/// an earlier launch cannot satisfy a wait.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelBenchArgs {
    /// 64-byte-aligned device address of the channel buffer, of at least
    /// [`channel_buffer_bytes`]`(message_bytes)` bytes, zeroed by the host.
    pub buffer: u64,
    /// 64-byte-aligned device address of the results area, of at least
    /// [`channel_results_bytes`]`(iterations)` bytes.
    pub results: u64,
    /// Payload bytes per message in each direction; a multiple of 8, and 0
    /// for a flag-only exchange.
    pub message_bytes: u64,
    /// Cycles after which a wait for a flag is abandoned with
    /// [`CHANNEL_STATUS_TIMEOUT`].
    pub timeout_cycles: u64,
    /// Shire of the ping Minion.
    pub ping_shire: u32,
    /// Minion (0..32) of the ping Minion within its shire.
    pub ping_minion: u32,
    /// Shire of the pong Minion.
    pub pong_shire: u32,
    /// Minion (0..32) of the pong Minion within its shire.
    pub pong_minion: u32,
    /// Cache level to which messages are written back and from which they are
    /// re-read: the `CacheDest` discriminant, 1 (L2, same shire only), 2 (L3)
    /// or 3 (DDR).
    pub cache_level: u32,
    /// Messages sent in each direction; below 2^20.
    pub iterations: u32,
    /// Launch identifier placed in the upper half of every flag value.
    pub epoch: u32,
    /// Reserved; must be zero.
    pub reserved: u32,
}

// SAFETY: repr(C), four u64 and eight u32 fields, no padding.
unsafe impl DeviceArgs for ChannelBenchArgs {}
const _: () = assert!(core::mem::size_of::<ChannelBenchArgs>() == 64);

/// Offset of the forward (ping to pong) flag in [`ChannelBenchArgs::buffer`].
pub const CHANNEL_FORWARD_FLAG_OFFSET: u64 = 0;
/// Offset of the reply (pong to ping) flag in [`ChannelBenchArgs::buffer`].
pub const CHANNEL_REPLY_FLAG_OFFSET: u64 = CACHE_LINE as u64;
/// Offset of the forward payload in [`ChannelBenchArgs::buffer`].
pub const CHANNEL_FORWARD_DATA_OFFSET: u64 = 2 * CACHE_LINE as u64;

/// Offset of the reply payload in [`ChannelBenchArgs::buffer`]: the forward
/// payload rounded up to whole cache lines, so the two never share a line.
pub const fn channel_reply_data_offset(message_bytes: u64) -> u64 {
    CHANNEL_FORWARD_DATA_OFFSET + message_bytes.next_multiple_of(CACHE_LINE as u64)
}

/// Bytes required for [`ChannelBenchArgs::buffer`].
pub const fn channel_buffer_bytes(message_bytes: u64) -> u64 {
    channel_reply_data_offset(message_bytes) + message_bytes.next_multiple_of(CACHE_LINE as u64)
}

/// Offset of the ping Minion's header in [`ChannelBenchArgs::results`].
pub const CHANNEL_PING_HEADER_OFFSET: u64 = 0;
/// Offset of the pong Minion's header in [`ChannelBenchArgs::results`].
pub const CHANNEL_PONG_HEADER_OFFSET: u64 = CACHE_LINE as u64;
/// Offset of the first sample in [`ChannelBenchArgs::results`].
pub const CHANNEL_SAMPLES_OFFSET: u64 = 2 * CACHE_LINE as u64;
/// `u64` words per sample: round-trip cycles, then send cycles.
pub const CHANNEL_SAMPLE_WORDS: u64 = 2;

/// Bytes required for [`ChannelBenchArgs::results`].
pub const fn channel_results_bytes(iterations: u32) -> u64 {
    CHANNEL_SAMPLES_OFFSET + iterations as u64 * CHANNEL_SAMPLE_WORDS * 8
}

/// Header word: one of the `CHANNEL_STATUS_*` codes.
pub const CHANNEL_HEADER_STATUS: usize = 0;
/// Header word: received payload words that did not match the expected value.
pub const CHANNEL_HEADER_ERRORS: usize = 1;
/// Header word: iterations completed by this Minion.
pub const CHANNEL_HEADER_COMPLETED: usize = 2;
/// Header word: the iteration whose wait timed out, or 0.
pub const CHANNEL_HEADER_FAILED_ITERATION: usize = 3;

/// Header status: every iteration completed.
pub const CHANNEL_STATUS_OK: u64 = 1;
/// Header status: a wait for a flag exceeded
/// [`ChannelBenchArgs::timeout_cycles`].
pub const CHANNEL_STATUS_TIMEOUT: u64 = 2;
/// Header status: the arguments were rejected and no message was sent.
pub const CHANNEL_STATUS_BAD_ARGUMENTS: u64 = 3;

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
            a: 0x0080_0100_0000,
            b: 0x0080_0200_0000,
            c: 0x0080_0300_0000,
            n_shires: 4,
            m: 128,
            n: 64,
            k: 256,
            lda: 1024, // 256 * 4 bytes, 64-byte aligned
            ldb: 256,  // 64 * 4 bytes, 64-byte aligned
            ldc: 256,  // 64 * 4 bytes, 64-byte aligned
            alpha: 1.0,
            beta: 0.0,
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

    #[test]
    fn channel_bench_layout() {
        // Flag-only: both flags, no payload lines.
        assert_eq!(channel_buffer_bytes(0), 128);
        assert_eq!(channel_reply_data_offset(0), 128);
        // A partial line is rounded up so the payloads never share a line.
        assert_eq!(channel_reply_data_offset(8), 192);
        assert_eq!(channel_buffer_bytes(8), 256);
        assert_eq!(channel_reply_data_offset(4096), 128 + 4096);
        assert_eq!(channel_buffer_bytes(4096), 128 + 2 * 4096);
        assert_eq!(channel_results_bytes(0), 128);
        assert_eq!(channel_results_bytes(10), 128 + 10 * 16);
    }
}
