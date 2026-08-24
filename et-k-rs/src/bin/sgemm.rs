//! Device kernel: single-precision GEMM (sGEMM) using the ET-SoC-1 tensor extension.
//!
//! Computes C = alpha * A * B + beta * C where:
//! - A is [M x K] row-major f32
//! - B is [K x N] row-major f32
//! - C is [M x N] row-major f32
//!
//! # v0.1 restrictions
//! - `alpha` must be `1.0` and `beta` must be `0.0`.
//! - All matrix pointers and leading dimensions must be 64-byte aligned.
//!
//! # Parallelism
//!
//! The kernel partitions the output tile grid across Minion cores. Each
//! compute shire has 32 Minion cores; `GemmArgs::n_shires` shires
//! participate, giving `n_shires * 32` concurrent Minion workers. Only the
//! primary hart (mhartid & 1 == 0) of each Minion issues tensor instructions;
//! the companion hart (mhartid & 1 == 1) returns immediately.
//!
//! Tile assignment is **shire-blocked**: each shire owns a contiguous
//! `ceil(n_tiles / n_shires)` slice of the tile grid. Within that block,
//! the 32 Minions distribute cyclically with step 32. Concentrating all
//! Minions in a shire on the same row-band of C improves A-row reuse in
//! the shire-shared L2 cache relative to the global-cyclic alternative.
//!
//! ```text
//! shire_base = shire * ceil(n_tiles / n_shires)
//! local_idx  = minion_in_shire;  step = 32;
//! while local_idx < shire_block_size { compute tile; local_idx += step; }
//! ```
//!
//! # Tile layout
//!
//! For a full tile:
//! - A sub-tile: 16 rows x 16 cols of f32, loaded into L1 scratchpad lines 0..15
//! - B sub-tile: 16 rows x 16 cols of f32, loaded into TenB register file
//! - C sub-tile: 16 rows x 16 cols of f32, accumulated in FP registers f0..f31
//!   (two 256-bit registers per C row: f[2i] and f[2i+1] for row i)
//!
//! Memory layout of the FP register file after TensorFMA32 with BCOLS=3,
//! AROWS=15, STEP=0, FREG=0:
//! - Row 0: f0 (lanes 0..7), f1 (lanes 8..15)
//! - Row 1: f2 (lanes 0..7), f3 (lanes 8..15)
//! - ...
//! - Row 15: f30 (lanes 0..7), f31 (lanes 8..15)

#![no_std]
#![no_main]

use et_abi::{
    DeviceArgs, GemmArgs, GEMM_TILE_K, GEMM_TILE_M, GEMM_TILE_N, MINIONS_PER_SHIRE,
};
use et_kernel::{
    fence, hart_id, kernel_entry, shire_id,
    tensor::{
        TensorEvent, fma32_xs, tensor_fma32, tensor_load, tensor_load_b,
        tensor_store, tensor_wait,
    },
};

kernel_entry!();

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // SAFETY: firmware staged a valid GemmArgs at args_ptr before launch.
    let args: &GemmArgs = unsafe { GemmArgs::from_ptr(args_ptr as *const u8) };

    // Only the primary hart of each Minion (even mhartid within a shire)
    // performs tensor work; the companion hart returns immediately.
    let h = hart_id();
    if h & 1 != 0 {
        return 0;
    }

    // Compute this Minion's global index and the total number of active
    // Minions. Tile assignment is cyclic with step = total_minions.
    let shire    = shire_id();
    let hart_in_shire = h & 63;            // 6 low bits: 0..63
    let minion_in_shire = hart_in_shire >> 1; // 0..31

    let total_minions = args.n_shires as u32 * MINIONS_PER_SHIRE;
    let my_minion = shire * MINIONS_PER_SHIRE + minion_in_shire;

    // This Minion participates only if its index is within the launched range.
    if my_minion >= total_minions {
        return 0;
    }

    // Tile grid dimensions. N need not be a multiple of GEMM_TILE_N; the last
    // column tile is partial and the hardware stores 64 bytes per C row
    // regardless, writing to the 64-byte-aligned padding already allocated by
    // alloc_tensor_matrix. Only C[row][0..N] is read by the caller.
    let n_tile_m = (args.m as usize).div_ceil(GEMM_TILE_M);
    let n_tile_n = (args.n as usize).div_ceil(GEMM_TILE_N);
    let n_tiles  = n_tile_m * n_tile_n;

    // Shire-blocked distribution: this shire handles a contiguous block.
    // Within the block, Minions distribute cyclically with step = MINIONS_PER_SHIRE.
    // All Minions in a shire work on the same row-band of C, improving A-row
    // reuse in the shire-shared L2 cache.
    let shire_size = n_tiles.div_ceil(args.n_shires as usize);
    let shire_base = (shire as usize) * shire_size;
    // Number of tiles this shire is responsible for (zero if shire_base >= n_tiles).
    let shire_end  = n_tiles.saturating_sub(shire_base).min(shire_size);

    let mut local_idx = minion_in_shire as usize;
    while local_idx < shire_end {
        let tile_idx = shire_base + local_idx;
        let tile_row = tile_idx / n_tile_n;
        let tile_col = tile_idx % n_tile_n;
        // SAFETY: tile coordinates are in-bounds; all pointer arithmetic stays
        // within the device buffers validated by the host before launch.
        unsafe { compute_tile(args, tile_row, tile_col) };
        local_idx += MINIONS_PER_SHIRE as usize;
    }

    0
}

/// Compute one output tile C[tile_row*TM..(tile_row+1)*TM][tile_col*TN..(tile_col+1)*TN].
///
/// The k-loop iterates over the inner dimension in slices of GEMM_TILE_K, calling
/// TensorFMA32 for each. On the first k-iteration `mul_only = true` so the FP
/// register file is initialised with A*B rather than A*B plus uninitialised
/// accumulator values. Subsequent iterations accumulate (mul_only = false).
///
/// # Safety
/// `tile_row` and `tile_col` must be valid indices into the tile grid; all
/// matrix pointer arithmetic must not overflow `usize`.
#[inline(always)]
unsafe fn compute_tile(args: &GemmArgs, tile_row: usize, tile_col: usize) {
    let c_row = tile_row * GEMM_TILE_M;
    let c_col = tile_col * GEMM_TILE_N;

    // Actual M rows in this tile (last tile may be partial).
    let actual_m = GEMM_TILE_M.min(args.m as usize - c_row);
    let arows    = (actual_m - 1) as u8;

    // With N enforced as a multiple of GEMM_TILE_N, BCOLS is always 3.
    let bcols: u8 = 3; // 4*(3+1) = 16 output f32 columns per row

    let a_base = args.a as usize;
    let b_base = args.b as usize;
    let c_base = args.c as usize;
    let lda    = args.lda as usize; // bytes
    let ldb    = args.ldb as usize;
    let ldc    = args.ldc as usize;

    let n_k_tiles = (args.k as usize).div_ceil(GEMM_TILE_K);

    let mut k_tile = 0_usize;
    while k_tile < n_k_tiles {
        let k_start  = k_tile * GEMM_TILE_K;
        let actual_k = GEMM_TILE_K.min(args.k as usize - k_start);
        let acols    = (actual_k - 1) as u8;

        // Device address of the A sub-tile: row c_row, column k_start.
        // lda is 64-byte aligned and k_start is a multiple of 16 (GEMM_TILE_K),
        // so k_start * 4 is a multiple of 64, making addr 64-byte aligned.
        let a_addr = a_base + c_row * lda + k_start * 4;

        // Device address of the B sub-tile: row k_start, column c_col.
        // c_col = tile_col * GEMM_TILE_N; c_col * 4 = tile_col * 64,
        // which is 64-byte aligned when ldb is 64-byte aligned.
        let b_addr = b_base + k_start * ldb + c_col * 4;

        // Load A tile into L1 scratchpad lines 0..arows (ID=0).
        // x31 = lda is set atomically with the CSRRW inside tensor_load.
        // SAFETY: a_addr is 64B-aligned; firmware guarantees buffer bounds.
        unsafe {
            tensor_load(a_addr, 0, arows, false, lda as u64);
            tensor_wait(TensorEvent::Load0);
        }

        // Load B tile into TenB register file (forward-paired with the FMA
        // below; no explicit wait needed before TensorFMA32).
        // Use id=true (Load1) so that tensor_wait(Load0) above waits only
        // for the A tile and not for this B DMA.
        // SAFETY: b_addr is 64B-aligned; acols matches the K-tile size.
        unsafe {
            tensor_load_b(b_addr, acols, false, ldb as u64, true);
        }

        // Issue TensorFMA32. First k-iteration: mul_only=true initialises the
        // FP register file with A*B. Subsequent iterations accumulate.
        let xs = fma32_xs(
            bcols,
            arows,
            acols,
            0,     // AOFFSET = 0 (A starts at byte 0 of each scratchpad line)
            true,  // TENB = 1 (B from TenB register file)
            0,     // BSTART ignored when TENB = 1
            0,     // ASTART = 0 (A at scratchpad lines 0..arows)
            k_tile == 0, // mul_only
            false, // use_mask = false (AROWS field controls row count)
        );
        // SAFETY: TensorWait(Load0) has been issued; TenB is being loaded.
        unsafe {
            tensor_fma32(xs);
            tensor_wait(TensorEvent::Fma);
        }

        k_tile += 1;
    }

    // Store the accumulated C tile from FP registers to memory.
    // ROWS = arows; SIZE = 3 (64B/row = 16 f32); STEP = 0, FREG = 0.
    let c_addr = c_base + c_row * ldc + c_col * 4;
    // SAFETY: c_addr is 64B-aligned; TensorWait(Fma) was issued above.
    unsafe {
        tensor_store(c_addr, arows, ldc as u64);
    }

    // Ensure the tensor store reaches memory before the kernel returns
    // (software coherence: other agents may read C via DMA after ecall).
    fence();
}

// ---------------------------------------------------------------------------
// Minimal runtime support
// ---------------------------------------------------------------------------

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // No unwinding support; spin indefinitely on panic.
    loop {
        unsafe { core::arch::asm!("wfi") };
    }
}
