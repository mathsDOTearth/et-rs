//! Pure-Rust "hello world" compute kernel for the ET-SoC-1.
//!
//! This is the device-side counterpart to the host `et_soc1` crate: every hart
//! writes `"Hello World from hart N"` into its U-mode trace buffer, then returns
//! to firmware. It is a faithful reimplementation of the SDK's C `hello.c`
//! (`et_printf` + `get_hart_id`) with no C dependency at all.
//!
//! ## How a hart reaches its trace buffer
//!
//! Firmware, before launching the kernel, populates a per-hart trace control
//! block (CB) at the fixed address [`CB_BASE`]. Each CB (64 bytes, one per hart)
//! holds the base address and current write offset of that hart's slice of the
//! trace buffer. Logging a string is exactly what the SDK's `Trace_String`
//! does: reserve `header + aligned_string` bytes by bumping the CB's offset,
//! then write a `trace_string_t` entry (a 16-byte common header followed by the
//! NUL-terminated, 8-byte-aligned string). Firmware finalises the buffer's size
//! headers when the kernel returns, after which the host DMAs it back and
//! decodes it with `et_soc1::trace`.
//!
//! Build: `cargo build --release` (targets RV64IMAC via `.cargo/config.toml`).

#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
use core::ptr::{read_volatile, write_volatile};

// Startup: initialise the global pointer, run the kernel, then return to
// firmware via `ecall` with SYSCALL_RETURN_FROM_KERNEL and KERNEL_RETURN_SUCCESS.
// Firmware has already set the stack pointer before entry.
global_asm!(
    ".section .text.init, \"ax\"",
    ".global _start",
    "_start:",
    ".option push",
    ".option norelax",
    "    la gp, __global_pointer$",
    ".option pop",
    "    call entry_point",
    "    li a2, 0", // KERNEL_RETURN_SUCCESS
    "    mv a1, a0", // kernel return value
    "    li a0, 8", // SYSCALL_RETURN_FROM_KERNEL
    "    ecall",
);

/// Base of the per-hart U-mode trace control-block array
/// (`CM_UMODE_TRACE_CB_BASEADDR`). Each entry is 64 bytes.
const CB_BASE: usize = 0x8004_F23000;
/// Size of one trace control block (`umode_trace_control_block_t`, `aligned(64)`).
const CB_STRIDE: usize = 64;
/// Byte offset of `base_per_hart` (u64) within a control block.
const CB_BASE_PER_HART: usize = 24;
/// Byte offset of `offset_per_hart` (u32) within a control block.
const CB_OFFSET_PER_HART: usize = 36;

/// `trace_type` discriminant for a string entry (`TRACE_TYPE_STRING`).
const TRACE_TYPE_STRING: u16 = 0;
/// Size of the common entry header (`trace_entry_header_t`).
const ENTRY_HEADER_SIZE: usize = 16;
/// Maximum logged string length (`TRACE_STRING_MAX_SIZE`).
const TRACE_STRING_MAX: usize = 512;

/// Read the current hart's ID from the custom `hartid` CSR (`0xCD0`).
#[inline(always)]
fn hart_id() -> u32 {
    let v: u64;
    // SAFETY: reads a U-mode-accessible CSR with no side effects.
    unsafe { asm!("csrr {0}, 0xcd0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v as u32
}

/// Read a cycle timestamp (`hpmcounter3`, CSR `0xC03`), used for the entry's
/// `cycle` field, matching the SDK trace encoder's timestamp source.
#[inline(always)]
fn timestamp() -> u64 {
    let v: u64;
    // SAFETY: reads a U-mode-accessible performance counter CSR.
    unsafe { asm!("csrr {0}, 0xc03", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// Map a hart ID to its control-block index (`GET_CB_INDEX`).
#[inline(always)]
fn cb_index(hart: u32) -> usize {
    if hart < 2048 { hart as usize } else { (hart - 32) as usize }
}

/// Round `n` up to a multiple of 8 (`TRACE_STRING_SIZE_ALIGN`).
#[inline(always)]
fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Append a decimal rendering of `v` to `buf` starting at `pos`; return the new
/// position. `buf` is assumed large enough for the caller's values.
fn write_decimal(buf: &mut [u8], pos: usize, v: u32) -> usize {
    let mut digits = [0u8; 10];
    let mut count = 0;
    let mut x = v;
    loop {
        digits[count] = b'0' + (x % 10) as u8;
        count += 1;
        x /= 10;
        if x == 0 {
            break;
        }
    }
    let mut p = pos;
    let mut i = count;
    while i > 0 {
        i -= 1;
        buf[p] = digits[i];
        p += 1;
    }
    p
}

/// The kernel body: log this hart's greeting into its trace buffer.
#[unsafe(no_mangle)]
pub extern "C" fn entry_point() -> i64 {
    let hid = hart_id();

    // Compose "Hello World from hart <id>\n" with a trailing NUL. The buffer is
    // comfortably larger than the longest possible line.
    let mut msg = [0u8; 64];
    let prefix = b"Hello World from hart ";
    let mut len = 0;
    while len < prefix.len() {
        msg[len] = prefix[len];
        len += 1;
    }
    len = write_decimal(&mut msg, len, hid);
    msg[len] = b'\n';
    len += 1;
    msg[len] = 0; // NUL terminator, included in the logged length
    len += 1;

    // The entry's string field is NUL-terminated and 8-byte aligned.
    let str_len = align8(len).min(TRACE_STRING_MAX);

    // Reserve space in this hart's trace buffer by bumping the CB offset, exactly
    // as trace_buffer_reserve does for the common (below-threshold) case.
    let cb = CB_BASE + cb_index(hid) * CB_STRIDE;
    // SAFETY: firmware populated the CB at this fixed address before launch.
    let base = unsafe { read_volatile((cb + CB_BASE_PER_HART) as *const u64) } as usize;
    let offset = unsafe { read_volatile((cb + CB_OFFSET_PER_HART) as *const u32) };
    let head = base + offset as usize;

    // Write the trace_string_t: 16-byte common header then the string payload.
    // SAFETY: `head` lies within this hart's reserved trace-buffer slice.
    unsafe {
        write_volatile(head as *mut u64, timestamp()); // header.cycle
        write_volatile((head + 8) as *mut u32, str_len as u32); // header.payload_size
        write_volatile((head + 12) as *mut u16, hid as u16); // header.hart_id
        write_volatile((head + 14) as *mut u16, TRACE_TYPE_STRING); // header.type

        let s = (head + ENTRY_HEADER_SIZE) as *mut u8;
        let mut i = 0;
        while i < str_len {
            let byte = if i < len { msg[i] } else { 0 };
            write_volatile(s.add(i), byte);
            i += 1;
        }

        // Commit the reservation.
        write_volatile(
            (cb + CB_OFFSET_PER_HART) as *mut u32,
            offset + (ENTRY_HEADER_SIZE + str_len) as u32,
        );
    }

    0
}

/// A kernel cannot unwind; on the (unexpected) panic path, spin so firmware's
/// watchdog reclaims the hart rather than executing undefined code.
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
