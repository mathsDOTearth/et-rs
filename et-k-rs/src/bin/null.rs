//! Empty compute kernel: every hart returns immediately.
//!
//! Used by the host `bench` example to measure the fixed cost of a kernel
//! launch (command submission, firmware dispatch, completion) independently of
//! any kernel work. The argument pointer is ignored, so the same image serves
//! launches with and without staged arguments.

#![no_std]
#![no_main]

use et_kernel::kernel_entry;

kernel_entry!();

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(_args_ptr: usize) -> i64 {
    0
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
