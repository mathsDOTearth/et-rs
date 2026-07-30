//! Single-producer / single-consumer, lock-free, **non-atomic** queue across two
//! harts of one neighbourhood, in shire-local L2 scratchpad.
//!
//! The kernel launches on a whole shire; two harts take the producer/consumer
//! roles (`PRODUCER_HART` / `CONSUMER_HART`, defaulting to the two threads of
//! minion 0), the rest return immediately. They share a ring buffer in the
//! shire's L2 scratchpad and coordinate with a classic SPSC protocol:
//!
//! * only the producer writes `tail` and the slots; only the consumer writes
//!   `head`. Each index therefore has a single writer, so naturally aligned
//!   32-bit loads/stores suffice -- **no atomic read-modify-write, no locks**.
//! * ordering (publish the slot before advancing `tail`; observe `tail` before
//!   reading the slot) is enforced with a hardware `fence rw, rw`, not atomics.
//!
//! Startup race is avoided without atomics: the producer initialises `head`,
//! `tail` and the ring, then publishes a `go` sentinel the consumer spins on.
//!
//! The consumer verifies the received sequence on-device and logs the result to
//! the trace buffer, which the host decodes.

#![no_std]
#![no_main]

use core::arch::naked_asm;
use core::ptr::{read_volatile, write_volatile};
use et_kernel::{MsgBuf, fence, hart_id, scp_shire_base, shire_id, trace_str};

/// Startup, placed in `.text.init` (laid down first at the fixed U-mode entry
/// address): init `gp`, run the kernel, return to firmware via `ecall`.
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

// --- Demo configuration (compile-time; a launch-arg version is a follow-up) ---
/// Producer hart. Default: thread 0 of minion 0.
const PRODUCER_HART: u32 = 0;
/// Consumer hart. Default: thread 1 of minion 0 (same minion, same neighbourhood).
/// Set to 2 for minion 1 (distinct L1, same neighbourhood) to probe cross-L1.
const CONSUMER_HART: u32 = 1;
/// Number of items to pass through the queue.
const N: u32 = 4096;
/// Ring capacity in slots (power of two).
const CAP: u32 = 256;
/// Sentinel the producer publishes once the control words are initialised.
const GO: u32 = 0x600D_600D;
/// Spin bound so a stall reports a timeout instead of hanging the device.
const SPIN_MAX: u64 = 2_000_000;

// Shared-region layout, each control word on its own 64-byte line (no false
// sharing); byte offsets from the region base.
const TAIL_OFF: usize = 0; // producer writes, consumer reads
const HEAD_OFF: usize = 64; // consumer writes, producer reads
const GO_OFF: usize = 128; // producer writes once, consumer reads
const RING_OFF: usize = 192; // CAP u32 slots

/// Where the shared ring lives.
///
/// * `false` (default): a `static` in the kernel image -- a fixed DRAM address
///   both harts resolve identically, coherent through the shared L1 of one
///   minion. Proves the lock-free logic on coherent memory.
/// * `true`: the shire L2 scratchpad. NOTE: the format-0 base
///   (`scp_shire_base`) is a per-hart *local* window, so two harts do NOT share
///   it -- this needs the correct shared-scratchpad base before it will work.
const USE_SCRATCHPAD: bool = false;

/// Backing store for the shared region when not using scratchpad. Non-zero
/// initialiser keeps it in `.data` (a kernel must have no `.bss`). Sized to
/// cover `RING_OFF` + `CAP` u32 slots.
const SHARED_WORDS: usize = RING_OFF / 4 + CAP as usize;
static mut SHARED: [u32; SHARED_WORDS] = [0xDEAD_BEEF; SHARED_WORDS];

#[inline(always)]
fn payload(i: u32) -> u32 {
    i.wrapping_add(1)
}

#[unsafe(no_mangle)]
pub extern "C" fn entry_point() -> i64 {
    let hid = hart_id();
    if hid != PRODUCER_HART && hid != CONSUMER_HART {
        return 0;
    }

    let base: *mut u8 = if USE_SCRATCHPAD {
        scp_shire_base(shire_id()) as *mut u8
    } else {
        &raw mut SHARED as *mut u8
    };
    // SAFETY: all offsets lie within the region (SCP page or the SHARED array).
    let tail_ptr = unsafe { base.add(TAIL_OFF) } as *mut u32;
    let head_ptr = unsafe { base.add(HEAD_OFF) } as *mut u32;
    let go_ptr = unsafe { base.add(GO_OFF) } as *mut u32;
    let ring = unsafe { base.add(RING_OFF) } as *mut u32;

    if hid == PRODUCER_HART {
        produce(tail_ptr, head_ptr, go_ptr, ring);
    } else {
        consume(tail_ptr, head_ptr, go_ptr, ring);
    }
    0
}

fn produce(tail_ptr: *mut u32, head_ptr: *mut u32, go_ptr: *mut u32, ring: *mut u32) {
    // Initialise all control words, then release the consumer.
    // SAFETY: scratchpad addresses within this shire's 2.5 MB region.
    unsafe {
        write_volatile(tail_ptr, 0);
        write_volatile(head_ptr, 0);
    }
    fence();
    unsafe { write_volatile(go_ptr, GO) };

    let mut tail: u32 = 0;
    let mut spins: u64 = 0;
    let mut ok = true;
    let mut i: u32 = 0;
    while i < N {
        // Wait while the ring is full (tail - head == CAP).
        loop {
            let h = unsafe { read_volatile(head_ptr) };
            if tail.wrapping_sub(h) < CAP {
                break;
            }
            spins += 1;
            if spins > SPIN_MAX {
                ok = false;
                break;
            }
            core::hint::spin_loop();
        }
        if !ok {
            break;
        }
        // Publish the slot, then advance tail with release ordering.
        unsafe { write_volatile(ring.add((tail % CAP) as usize), payload(i)) };
        fence();
        tail = tail.wrapping_add(1);
        unsafe { write_volatile(tail_ptr, tail) };
        i += 1;
    }

    let mut m = MsgBuf::new();
    m.str(b"SPSC producer hart ")
        .u64(PRODUCER_HART as u64)
        .str(b" sent ")
        .u64(i as u64)
        .str(if ok { b" items OK" } else { b" items TIMEOUT" });
    trace_str(m.as_slice());
}

fn consume(tail_ptr: *mut u32, head_ptr: *mut u32, go_ptr: *mut u32, ring: *mut u32) {
    // Wait for the producer to finish initialisation.
    let mut spins: u64 = 0;
    let mut ok = true;
    loop {
        if unsafe { read_volatile(go_ptr) } == GO {
            break;
        }
        spins += 1;
        if spins > SPIN_MAX {
            ok = false;
            break;
        }
        core::hint::spin_loop();
    }
    fence();

    let mut head: u32 = 0;
    let mut sum: u64 = 0;
    let mut errors: u32 = 0;
    let mut i: u32 = 0;
    while ok && i < N {
        // Wait while the ring is empty (tail == head).
        loop {
            let t = unsafe { read_volatile(tail_ptr) };
            if t != head {
                break;
            }
            spins += 1;
            if spins > SPIN_MAX {
                ok = false;
                break;
            }
            core::hint::spin_loop();
        }
        if !ok {
            break;
        }
        // Observe the published slot with acquire ordering.
        fence();
        let v = unsafe { read_volatile(ring.add((head % CAP) as usize)) };
        if v != payload(i) {
            errors += 1;
        }
        sum = sum.wrapping_add(v as u64);
        head = head.wrapping_add(1);
        unsafe { write_volatile(head_ptr, head) };
        i += 1;
    }

    // Expected checksum = sum of payload(0..N) = 1 + 2 + ... + N.
    let expected = (N as u64) * (N as u64 + 1) / 2;
    let pass = ok && errors == 0 && i == N && sum == expected;

    let mut m = MsgBuf::new();
    m.str(b"SPSC consumer hart ")
        .u64(CONSUMER_HART as u64)
        .str(b" got ")
        .u64(i as u64)
        .str(b" items, sum ")
        .u64(sum)
        .str(b" (expected ")
        .u64(expected)
        .str(b"), errors ")
        .u64(errors as u64)
        .str(if pass {
            b" -> RESULT PASS"
        } else {
            b" -> RESULT FAIL"
        });
    trace_str(m.as_slice());
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
