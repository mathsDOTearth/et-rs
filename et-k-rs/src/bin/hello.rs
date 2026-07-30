//! Pure-Rust "hello world" compute kernel: every hart logs its greeting to the
//! U-mode trace buffer and returns. See the package README for the mechanism.

#![no_std]
#![no_main]

use core::arch::naked_asm;
use et_kernel::{MsgBuf, hart_id, trace_str};

/// Startup, placed in `.text.init` (laid down first at the fixed U-mode entry
/// address): init `gp`, run the kernel, return to firmware via `ecall`
/// (`SYSCALL_RETURN_FROM_KERNEL` = 8, `KERNEL_RETURN_SUCCESS` = 0). The stack
/// pointer is set by firmware before entry.
#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.init")]
pub extern "C" fn _start() -> ! {
    naked_asm!(
        ".option push",
        ".option norelax",
        "la gp, __global_pointer$",
        ".option pop",
        "call entry_point",
        "li a2, 0",
        "mv a1, a0",
        "li a0, 8",
        "ecall",
    )
}

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
