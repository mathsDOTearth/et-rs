//! Cache-coherence test kernel for the ET-SoC-1.
//!
//! Each primary Minion hart (even mhartid) writes its global Minion index as a
//! `u32` to its own cache-line-padded output cell, then applies the cache
//! operation selected by `args.op` (`cache_writeback_to`, `cache_invalidate_to`
//! or `cache_flush`) followed by `fence`. The host downloads the output array
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
//!
//! The invalidate and flush operations issue only `evict_va` (CSR `0x89F`).
//! They pass only if eviction writes a dirty line back before invalidating it
//! (PRM Section 8.4); an eviction that discarded the line would leave the
//! host's sentinel in DDR.

#![no_std]
#![no_main]

use core::mem::size_of;

use et_abi::{
    CACHE_LINE, CACHE_TEST_OP_FLUSH, CACHE_TEST_OP_INVALIDATE, CacheTestArgs, DeviceArgs,
    MINIONS_PER_SHIRE,
};
use et_kernel::{
    cache::{CacheDest, cache_flush, cache_invalidate_to, cache_writeback_to},
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

    let shire = shire_id();
    let hart_in_shire = h & 63; // 6 low bits: 0..63
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

    // Commit the store to L1 before the writeback (PRM Section 8.1.3): the cache
    // op must observe the dirty line, so the producing store has to be ordered
    // ahead of it. Previously this fence was missing (it sat only after the
    // writeback), leaving the flush ordered only by the asm memory clobber.
    fence();

    // Push the dirty L1 line to the requested level. Each operation issues its
    // CSR write then TensorWait(6) (PRM Table 9-2, event 6), stalling the hart
    // until the cache operation completes. `dest` selects L2/L3/Mem so the fault
    // can be narrowed to the DDR path; only Mem is visible to host DMA.
    let dest = match args.dest {
        1 => CacheDest::L2,
        2 => CacheDest::L3,
        _ => CacheDest::Mem,
    };
    // SAFETY: cell_addr is valid; size_of::<u32>() bytes lie within the cell,
    // and no other hart writes to it, so writing the line back is safe.
    unsafe {
        match args.op {
            CACHE_TEST_OP_INVALIDATE => cache_invalidate_to(dest, cell_addr, size_of::<u32>()),
            CACHE_TEST_OP_FLUSH => cache_flush(cell_addr, size_of::<u32>()),
            _ => cache_writeback_to(dest, cell_addr, size_of::<u32>()),
        }
    }

    // Order the writeback completion relative to the ecall return so the
    // firmware interrupt handler does not observe an older memory state.
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
