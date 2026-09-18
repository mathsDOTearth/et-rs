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
//! Minion `i` computes `(i + 1) as f32 * 3.0` using these instructions and
//! writes the f32 result to its cache-line-padded output cell. The host
//! verifies every cell against the expected scalar value. All values in the
//! range 1..=1024 are exactly representable as f32 (well within the 24-bit
//! mantissa), so a bit-exact comparison is valid regardless of rounding mode.
//!
//! The encodings tested here are identical to those emitted by
//! `broadcast_ps(alpha, dest)` and `fmul_ps_row(row, scratch)` in
//! `et_kernel::simd`.
//!
//! # Build note
//! Requires `target-feature=+f` (enabled in `.cargo/config.toml`) for the
//! `fmv.w.x` and `fmv.x.w` F-extension instructions.

#![no_std]
#![no_main]

use core::mem::size_of;

use et_abi::{CACHE_LINE, DeviceArgs, MINIONS_PER_SHIRE, SimdTestArgs};
use et_kernel::{cache::cache_writeback, fence, hart_id, kernel_entry, shire_id};

kernel_entry!();

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // SAFETY: firmware staged a valid SimdTestArgs at args_ptr before launch.
    let args: &SimdTestArgs = unsafe { SimdTestArgs::from_ptr(args_ptr as *const u8) };

    let h = hart_id();
    if h & 1 != 0 {
        return 0;
    }

    let shire = shire_id();
    let hart_in_shire = h & 63;
    let minion_in_shire = hart_in_shire >> 1;
    let my_minion = shire * MINIONS_PER_SHIRE + minion_in_shire;
    let total_minions = args.n_shires as u32 * MINIONS_PER_SHIRE;

    if my_minion >= total_minions {
        return 0;
    }

    // Input: (my_minion + 1) as f32; avoids a trivial 0.0 case for Minion 0.
    // Alpha: 3.0. Both are injected into the FP register file as GPR bit patterns
    // via fmv.w.x / FBCX.PS so the compiler does not need to allocate FP operands
    // around the inline asm.
    let input_bits = ((my_minion + 1) as f32).to_bits() as u64;
    let alpha_bits = 3.0_f32.to_bits() as u64;
    let result_bits: u64;

    // SAFETY: f2 and f28 are caller-saved FP temporaries on RV64F. No live
    // values are held there at this point; all uses are declared as clobbers.
    unsafe {
        core::arch::asm!(
            // Load input float bits into f2 (lane 0 of the 256-bit PS register).
            // f2 is the low register of the row-1 pair (f2/f3) -- chosen to keep
            // f0/f1 free for any implicit use by the C calling convention.
            "fmv.w.x f2, {input_bits}",
            // FBCX.PS f28 <- broadcast of alpha_bits to all 8 PS lanes.
            // I-type: opcode=0x0b, funct3=3, rd=f28, rs1={alpha_bits}, imm=0.
            // MATCH = 0x0000_300b (imm=0, rs1=0, funct3=3, rd=0, opcode=0x0b).
            ".insn i 0x0b, 3, f28, {alpha_bits}, 0",
            // FMUL.PS f2 = f2 .* f28: element-wise multiply all 8 lanes.
            // R-type: opcode=0x7b, funct3=0, funct7=8, rd=f2, rs1=f2, rs2=f28.
            // MATCH = 0x1000_007b (funct7=8, rs2=0, rs1=0, funct3=0, rd=0, opcode=0x7b).
            ".insn r 0x7b, 0, 8, f2, f2, f28",
            // Extract lane 0 (low 32 bits of f2) back to a GPR for writeback.
            "fmv.x.w {result_bits}, f2",
            input_bits  = in(reg)  input_bits,
            alpha_bits  = in(reg)  alpha_bits,
            result_bits = out(reg) result_bits,
            out("f2")   _,
            out("f28")  _,
            options(nostack, preserves_flags),
        );
    }

    let result = f32::from_bits(result_bits as u32);
    let cell_addr = args.output as usize + my_minion as usize * CACHE_LINE;

    // SAFETY: cell_addr is an exclusive 64-byte cell allocated by the host;
    // no other Minion aliases this address.
    unsafe {
        core::ptr::write_volatile(cell_addr as *mut f32, result);
    }

    fence();
    // SAFETY: cell_addr is valid device memory; size_of::<f32>() bytes lie within the cell.
    unsafe {
        cache_writeback(cell_addr, size_of::<f32>());
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
