//! Pure-Rust "hello world" compute kernel: every hart logs its greeting to the
//! U-mode trace buffer and returns. See the package README for the mechanism.

#![no_std]
#![no_main]

use et_kernel::{MsgBuf, hart_id, kernel_entry, trace_str};

// The `_start` entry point (naked, in .text.init). The stack pointer is set by
// firmware before entry.
kernel_entry!();

#[unsafe(no_mangle)]
pub extern "C" fn entry_point() -> i64 {
    let mut m = MsgBuf::new();
    m.str(b"Hello World from hart ").u64(hart_id() as u64);
    trace_str(m.as_slice());
    0
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
