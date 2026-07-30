//! Data-parallel reduction: sum a large array across a whole shire of harts.
//!
//! This is the "fearless data parallelism" demo. The kernel launches on a shire
//! (every hart runs it); each hart reduces its **disjoint** slice of the input
//! and writes its partial to its **own** cache-line-padded output cell. There is
//! no cross-hart sharing during the compute, so it is coherence-clean (passes
//! the emulator's consistency checkers and works on hardware); the host DMAs the
//! partials back and combines them.
//!
//! Data safety: the kernel body is safe Rust built on [`et_kernel::Grid`], which
//! hands each hart only its own slice and its own cell. A kernel author cannot
//! name another hart's data, so a cross-hart data race is unrepresentable. The
//! only `unsafe` is the thin, commented boundary that turns the launch arguments
//! and device addresses into typed slices.
//!
//! Launch arguments: the firmware delivers the launch command's
//! `pointer_to_args` in register `a0` (verified on device -- despite the runtime
//! docs saying `ra`, `ra` is 0 at entry). `a0` flows straight through `_start`
//! into `entry_point`'s first parameter, so no register shuffling is needed.

#![no_std]
#![no_main]

use et_abi::{DeviceArgs, ReduceArgs};
use et_kernel::{Grid, MsgBuf, device_slice, kernel_entry, trace_str};

// The `_start` entry point (naked, in .text.init). `a0` carries the launch-args
// pointer straight through to `entry_point`.
kernel_entry!();

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // --- unsafe boundary: interpret the launch args and device memory ---
    // SAFETY: the firmware passed the launch command's `pointer_to_args` in a0;
    // `ReduceArgs` is the shared host/device definition (et-abi).
    let args = unsafe { ReduceArgs::from_ptr(args_ptr as *const u8) };
    let grid = Grid::new(args.n_harts);
    if !grid.active() {
        return 0;
    }
    // SAFETY: the host allocated `n` u32 of input and `n_harts` cache-lines of
    // output at these device addresses (see the host program).
    let input: &[u32] = unsafe { device_slice(args.input as usize, args.n as usize) };
    let cell: &mut u64 = unsafe { grid.output_cell(args.out as usize) };

    // --- safe from here: partition + reduce, no raw pointers, no aliasing ---
    let partial: u64 = grid.my_slice(input).iter().map(|&x| x as u64).sum();
    *cell = partial;

    // One hart reports the parameters and its partial, to confirm args delivery.
    if grid.hart() == 0 {
        let mut m = MsgBuf::new();
        m.str(b"reduce: n=")
            .u64(args.n as u64)
            .str(b" harts=")
            .u64(args.n_harts as u64)
            .str(b" hart0_partial=")
            .u64(partial);
        trace_str(m.as_slice());
    }
    0
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
