//! Cache-coherence test kernel for the ET-SoC-1.
//!
//! Each primary Minion hart (even mhartid) writes its global Minion index as a
//! `u32` to its own cache-line-padded output cell, then calls
//! `cache_writeback` followed by `fence`. The host downloads the output array
//! and verifies that every cell holds the expected Minion index.
//!
//! # What this tests
//!
//! On the ET-SoC-1, host DMA bypasses all Minion L1 caches. A CPU store
//! followed by `fence rw, rw` alone leaves the written value in L1: the host
//! would read stale DDR data. `cache_writeback` (CSR `0x8BF`, `flush_va`)
//! pushes the dirty line from L1 to DDR before the fence, making the write
//! visible to the host. A failure in this test indicates that the writeback
//! did not complete before the kernel returned.

#![no_std]
#![no_main]

use core::mem::size_of;

use et_abi::{CacheTestArgs, DeviceArgs, CACHE_LINE, MINIONS_PER_SHIRE};
use et_kernel::{
    cache::cache_writeback,
    fence, hart_id, kernel_entry, shire_id,
};

kernel_entry!();

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // SAFETY: firmware staged a valid CacheTestArgs at args_ptr before launch.
    let args: &CacheTestArgs = unsafe { CacheTestArgs::from_ptr(args_ptr as *const u8) };

    // Only the primary hart of each Minion (even mhartid within a shire)
    // performs work; the companion hart (odd mhartid) returns immediately.
    let h = hart_id();
    if h & 1 != 0 {
        return 0;
    }

    let shire           = shire_id();
    let hart_in_shire   = h & 63;         // 6 low bits: 0..63
    let minion_in_shire = hart_in_shire >> 1; // 0..31

    let my_minion = shire * MINIONS_PER_SHIRE + minion_in_shire;
    let total_minions = args.n_shires as u32 * MINIONS_PER_SHIRE;

    if my_minion >= total_minions {
        return 0;
    }

    // Compute the address of this Minion's output cell.
    // The host allocated one u32 per cache line (stride = CACHE_LINE = 64),
    // so cells do not share a cache line and there is no false-sharing hazard.
    let cell_addr = args.output as usize + my_minion as usize * CACHE_LINE;

    // Write the Minion index to the cell. volatile prevents the compiler
    // from eliding the store (it has no other visible caller).
    // SAFETY: cell_addr points to an exclusive u32 cell allocated by the host
    // (one per Minion, distinct cache lines); no other hart aliases this address.
    unsafe {
        core::ptr::write_volatile(cell_addr as *mut u32, my_minion);
    }

    // Flush the dirty L1 line to DDR before issuing the fence. Without this,
    // the write may still be in L1 when the host's DMA engine reads the buffer:
    // the tensor DMA and the host DMA both bypass the Minion L1 caches.
    // SAFETY: cell_addr is valid; size_of::<u32>() bytes lie within the cell.
    unsafe {
        cache_writeback(cell_addr, size_of::<u32>());
    }

    // Order the writeback completion relative to the ecall return.
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
