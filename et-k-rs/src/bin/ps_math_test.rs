//! PS transcendental accuracy kernel for the ET-SoC-1.
//!
//! Applies one of the native PS transcendental instructions to an input array
//! of `f32`, so that the host can measure the error of every lane against a
//! reference:
//!
//! - `FEXP.PS` (2^x), `FLOG.PS` (log2 x) and `FRCP.PS` (1/x), all R-type with
//!   opcode 0x7b, funct7 0x2c and the function selected by rs2 (4, 3 and 7);
//! - a copy operation with no arithmetic, which verifies the `FLQ2` (load)
//!   and `FSQ2` (store) paths bit-exactly before the arithmetic results are
//!   trusted.
//!
//! The arrays are divided into 64-byte units (16 `f32`, two PS registers).
//! The primary hart of each launched Minion takes units in grid-stride order:
//! unit `rank`, `rank + n_minions`, ..., where `rank` is the Minion's dense
//! index over the launched shires. Each unit is invalidated in the cache
//! before loading, since the host rewrites the input between launches, and
//! written back to DDR after storing.
//!
//! No floating-point value is computed by compiled code; all FP register
//! traffic is through the `et_kernel::simd` wrappers.

#![no_std]
#![no_main]

use et_abi::{
    CACHE_LINE, DeviceArgs, MINIONS_PER_SHIRE, PS_MATH_OP_EXP2, PS_MATH_OP_LOG2,
    PS_MATH_OP_RECIPROCAL, PsMathTestArgs,
};
use et_kernel::cache::{cache_invalidate, cache_writeback};
use et_kernel::simd::{fexp_ps, flog_ps, frcp_ps, load_ps, store_ps};
use et_kernel::{hart_id, kernel_entry, shire_id};

kernel_entry!();

/// Bytes held by one PS register (eight `f32` lanes).
const PS_BYTES: usize = 32;
/// `f32` elements per unit (one cache line).
const UNIT_ELEMENTS: u64 = (CACHE_LINE / size_of::<f32>()) as u64;

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // SAFETY: firmware staged a valid PsMathTestArgs at args_ptr before launch.
    let args: &PsMathTestArgs = unsafe { PsMathTestArgs::from_ptr(args_ptr as *const u8) };

    let h = hart_id();
    if h & 1 != 0 {
        return 0;
    }
    let shire = shire_id();
    let shire_bit = 1u32.checked_shl(shire).unwrap_or(0);
    if args.shire_mask & shire_bit == 0 {
        return 0;
    }

    // Dense Minion rank over the launched shires.
    let shires_below = (args.shire_mask & shire_bit.wrapping_sub(1)).count_ones();
    let rank = (shires_below * MINIONS_PER_SHIRE + ((h & 63) >> 1)) as u64;
    let n_minions = (args.shire_mask.count_ones() * MINIONS_PER_SHIRE) as u64;
    let n_units = args.count / UNIT_ELEMENTS;

    let mut unit = rank;
    while unit < n_units {
        let offset = (unit as usize) * CACHE_LINE;
        let source = args.input as usize + offset;
        let destination = args.output as usize + offset;

        // SAFETY: both addresses lie within the host-allocated arrays and are
        // 64-byte aligned; this hart alone handles the unit. No FP values are
        // live in compiled code; the wrappers declare f0..f31 clobbered.
        unsafe {
            cache_invalidate(source, CACHE_LINE);
            load_ps(0, source);
            load_ps(1, source + PS_BYTES);
            match args.operation {
                PS_MATH_OP_EXP2 => {
                    fexp_ps(0);
                    fexp_ps(1);
                }
                PS_MATH_OP_LOG2 => {
                    flog_ps(0);
                    flog_ps(1);
                }
                PS_MATH_OP_RECIPROCAL => {
                    frcp_ps(0);
                    frcp_ps(1);
                }
                // PS_MATH_OP_COPY; an unrecognised operation also copies,
                // which the host's per-operation checks then report.
                _ => {}
            }
            store_ps(0, destination);
            store_ps(1, destination + PS_BYTES);
            cache_writeback(destination, CACHE_LINE);
        }
        unit += n_minions;
    }

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
