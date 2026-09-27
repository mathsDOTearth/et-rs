//! PS SIMD instruction verification kernel for the ET-SoC-1.
//!
//! Each primary Minion hart (even mhartid) exercises the two PS SIMD
//! instructions used by the `et_kernel::simd` module:
//!
//! - `FBCX.PS` (opcode 0x0b, funct3=3, imm=0): broadcasts the bit pattern of a
//!   GPR scalar to all eight f32 lanes of a PS FP register.
//! - `FMUL.PS` (opcode 0x7b, funct7=8, funct3=0): element-wise f32 multiply of
//!   two PS FP register pairs, writing the result to the first.
//!
//! Each participating Minion calls the library entry points themselves:
//!
//! 1. [`broadcast_ps_bits`] places `(minion + 1) as f32` in all eight lanes of
//!    f0 and of f1 (row 0 of the C tile), and `3.0` in all lanes of f28.
//! 2. [`fmul_ps_row`]`(0, 28)` multiplies f0 and f1 lane-wise by f28.
//! 3. `tensor_store` writes row 0 (f0 then f1: 16 f32, 64 bytes) to the
//!    Minion's cache-line output cell.
//!
//! The host checks all 16 values of every cell, so an instruction that wrote
//! only lane 0, or only one register of the pair, is detected. All values are
//! exactly representable as f32, so the comparison is bit-exact.
//!
//! Cells are indexed by physical Minion number (`shire * 32 + minion`), so a
//! shire mask with gaps leaves the absent shires' cells untouched rather than
//! shifting every later Minion. `n_shires` bounds the cell array.

#![no_std]
#![no_main]

use et_abi::{CACHE_LINE, DeviceArgs, MINIONS_PER_SHIRE, SimdTestArgs};
use et_kernel::simd::{broadcast_ps_bits, fmul_ps_row};
use et_kernel::tensor::{TensorEvent, tensor_store, tensor_wait};
use et_kernel::{fence, hart_id, kernel_entry, shire_id};

kernel_entry!();

/// FP register holding the broadcast multiplier.
const ALPHA_REGISTER: u8 = 28;
/// Multiplier applied to every lane.
const ALPHA: f32 = 3.0;

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // SAFETY: firmware staged a valid SimdTestArgs at args_ptr before launch.
    let args: &SimdTestArgs = unsafe { SimdTestArgs::from_ptr(args_ptr as *const u8) };

    // Tensor stores are issued from the primary hart of each Minion only.
    let h = hart_id();
    if h & 1 != 0 {
        return 0;
    }

    let shire = shire_id();
    if shire as u64 >= args.n_shires {
        return 0;
    }
    let my_minion = shire * MINIONS_PER_SHIRE + ((h & 63) >> 1);
    let cell_addr = args.output as usize + my_minion as usize * CACHE_LINE;

    // Input (minion + 1) avoids a trivial 0.0 for Minion 0. Both operands are
    // passed as integer bit patterns, so no FP register is used by compiled code.
    let input_bits = ((my_minion + 1) as f32).to_bits();

    // SAFETY: no FP values are live (none are computed in this kernel); the
    // simd and tensor functions declare f0..f31 clobbered. `cell_addr` is this
    // Minion's exclusive, 64-byte-aligned output line.
    unsafe {
        broadcast_ps_bits(input_bits, 0);
        broadcast_ps_bits(input_bits, 1);
        broadcast_ps_bits(ALPHA.to_bits(), ALPHA_REGISTER);
        fmul_ps_row(0, ALPHA_REGISTER);
        tensor_store(cell_addr, 0, CACHE_LINE as u64);
        tensor_wait(TensorEvent::Store);
    }
    fence();

    0
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
