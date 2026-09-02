//! Tensor extension instruction test kernel.
//!
//! Each primary Minion hart exercises three instructions added in v0.5.0, in sequence:
//!
//! 1. **TensorFMA16A32** (CSR 0x801, bits 3:1 = 001): fp16 GEMM.
//!    A = `[2.0_f16, 3.0_f16]`, B interleaved: `[1.0, 1.0, ...]`.
//!    Expected C[0] = 5.0_f32; C[1..3] = 0.0.
//!
//! 2. **TensorIMA8A32** (CSR 0x801, bits 3:1 = 011): int8 GEMM.
//!    A = `[1, 1, 1, 1]` (int8), B col 0 = `[1, 1, 1, 1]`, cols 1..3 = 0.
//!    Expected C[0] = 4 (int32); C[1..3] = 0. Result stored via FP register
//!    file (DST = 1) and read back as int32 bit patterns.
//!
//! 3. **TensorStoreFromScp** (CSR 0x87F, bit 48 = 1): passthrough.
//!    Writes scratchpad line 0 (populated in subtest 1 with the fp16 A data)
//!    directly to memory. Host verifies the 64 bytes match the input.
//!
//! Tile size for all subtests: AROWS=0 (1 row), ACOLS=0 (1 K-group), BCOLS=0
//! (4 output columns). This is the smallest non-trivial GEMM tile.
//!
//! # Output layout (per Minion, stride = 192 bytes)
//! - bytes `[0..64)`:   FMA16A32 result  (4 × f32).
//! - bytes `[64..128)`:  IMA8A32 result   (4 × i32 as f32 bit patterns).
//! - bytes `[128..192)`: StoreFromScp passthrough (64 bytes from scratchpad line 0).
//!
//! # Usage
//! Build with `--target riscv64imac-unknown-none-elf --release`, then run the
//! host-side `tensor_ext_test` example with the resulting ELF.

#![no_std]
#![no_main]

use et_abi::{DeviceArgs, MINIONS_PER_SHIRE, TensorExtTestArgs};
use et_kernel::{
    fence, hart_id, kernel_entry, shire_id,
    tensor::{
        TensorEvent, fma16a32_xs, ima8a32_xs,
        tensor_fma16a32, tensor_ima8a32, tensor_load, tensor_load_b,
        tensor_store, tensor_store_from_scp, tensor_wait,
    },
};

kernel_entry!();

/// Bytes of output per Minion (3 subtests × 1 tile row × 64 bytes/row).
const OUT_STRIDE: usize = 192;

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // SAFETY: firmware staged a valid TensorExtTestArgs at args_ptr.
    let args: &TensorExtTestArgs =
        unsafe { TensorExtTestArgs::from_ptr(args_ptr as *const u8) };

    // Only the primary hart of each Minion issues tensor instructions.
    let h = hart_id();
    if h & 1 != 0 {
        return 0;
    }

    let shire           = shire_id();
    let minion_in_shire = (h & 63) >> 1;
    let my_minion       = shire * MINIONS_PER_SHIRE + minion_in_shire;
    let total_minions   = args.n_shires as u32 * MINIONS_PER_SHIRE;

    if my_minion >= total_minions {
        return 0;
    }

    let out_base = args.output as usize + my_minion as usize * OUT_STRIDE;

    // SAFETY: all args pointers are 64-byte aligned device memory; out_base is
    // within the allocated output buffer, 64-byte aligned.
    unsafe { run_tests(args, out_base) };

    0
}

/// Execute all three tensor extension subtests for one Minion.
///
/// Scratchpad layout used:
/// - Line 0:  A fp16 data (from subtest 1; reused by subtest 3).
/// - Line 1:  A int8 data (subtest 2).
/// - Line 16: B int8 data (subtest 2).
///
/// # Safety
/// All device pointers in `args` must be 64-byte aligned and valid. `out_base`
/// must be 64-byte aligned and point to at least 192 writable bytes.
#[inline(always)]
unsafe fn run_tests(args: &TensorExtTestArgs, out_base: usize) {
    // -----------------------------------------------------------------------
    // Subtest 1: TensorFMA16A32
    //
    // A (scratchpad line 0): [2.0_f16, 3.0_f16] at bytes 0-3.
    //   fp16 2.0 = 0x4000, fp16 3.0 = 0x4200 (little-endian).
    // B (TenB[0]): interleaved fp16 pairs per output column (PRM Table 9-4).
    //   h[0]=b[0,0]=1.0_f16, h[1]=b[1,0]=1.0_f16, h[2..7]=0.0.
    // Expected C: f[0].e[0]=5.0_f32, f[0].e[1..3]=0.0.
    //   C'[0][j] = A[0][k]*B[k,j] + A[0][k+1]*B[k+1,j] (PRM FMA16A32 pseudocode).
    //   j=0: 2.0*1.0 + 3.0*1.0 = 5.0. j=1..3: 0.0.
    // -----------------------------------------------------------------------
    unsafe {
        // Load A fp16 into scratchpad line 0 (id=Load0).
        tensor_load(args.a_fp16 as usize, 0, 0, false, 64);
        tensor_wait(TensorEvent::Load0);

        // Load B fp16 into TenB register file (id=Load1). Forward-pairs with
        // the TensorFMA16A32 below; the hardware waits internally for TenB.
        // ROWS must equal ACOLS of the FMA (both 0, matching PRM pairing rule).
        tensor_load_b(args.b_fp16 as usize, 0, false, 64, true);

        // TensorFMA16A32: AROWS=0, ACOLS=0, BCOLS=0, TENB=1, mul_only=true.
        tensor_fma16a32(fma16a32_xs(0, 0, 0, 0, true, 0, 0, true, false));
        tensor_wait(TensorEvent::Fma);

        // Store C (64 bytes: f[0] and f[1]) to output section 0.
        tensor_store(out_base, 0, 64);
        tensor_wait(TensorEvent::Store);
    }

    // -----------------------------------------------------------------------
    // Subtest 2: TensorIMA8A32
    //
    // A (scratchpad line 1): [1, 1, 1, 1] at bytes 0-3 (signed int8).
    // B (scratchpad line 16): IMA8A32 interleaved int8 (PRM Table 9-4):
    //   word 0 = [b[0,0]=1, b[1,0]=1, b[2,0]=1, b[3,0]=1] (col 0, K-rows 0-3).
    //   words 1-3 = [0, 0, 0, 0] (cols 1-3).
    // Expected C: f[0].e[0] = 4 (int32), f[0].e[1..3] = 0.
    //   C[0][0] = 1*1+1*1+1*1+1*1 = 4. C[0][1..3] = 0.
    // DST=1 stores int32 result as fp32 bit patterns in f[0]; tensor_store
    // writes those bit patterns directly to memory.
    // -----------------------------------------------------------------------
    unsafe {
        // Load A int8 and B int8 into the scratchpad (both with Load0 id;
        // a single tensor_wait(Load0) drains both).
        tensor_load(args.a_int8 as usize, 1, 0, false, 64);
        tensor_load(args.b_int8 as usize, 16, 0, false, 64);
        tensor_wait(TensorEvent::Load0);

        // TensorIMA8A32: TENB=0 (B from scratchpad BSTART=16), ASTART=1,
        // DST=1 (result to FP register file), UA=0 (signed), UB=0 (signed),
        // mul_only=true (C = A*B, not C += A*B).
        tensor_ima8a32(ima8a32_xs(0, 0, 0, 0, false, 16, 1, true, false, false, true, false));
        tensor_wait(TensorEvent::Fma);

        // Store int32 result (as f32 bit patterns) to output section 1.
        tensor_store(out_base + 64, 0, 64);
        tensor_wait(TensorEvent::Store);
    }

    // -----------------------------------------------------------------------
    // Subtest 3: TensorStoreFromScp
    //
    // Scratchpad line 0 was loaded in subtest 1 and has not been modified
    // since. Write it directly from the L1 scratchpad to output section 2.
    // The host verifies output[128..192] == a_fp16 buffer (64 bytes).
    // -----------------------------------------------------------------------
    unsafe {
        tensor_store_from_scp(out_base + 128, 0, 0, 1, 64);
        tensor_wait(TensorEvent::Store);
    }

    // Tensor stores bypass L1 cache and write directly to DRAM; fence()
    // ensures all DRAM writes are visible to the DMA engine on kernel return.
    fence();
}

// ---------------------------------------------------------------------------
// Minimal runtime support
// ---------------------------------------------------------------------------

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        unsafe { core::arch::asm!("wfi") };
    }
}
