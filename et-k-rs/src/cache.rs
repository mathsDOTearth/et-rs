//! Cache management operations for the ET-SoC-1 Minion processor.
//!
//! The ET-SoC-1 implements a software-coherent memory model: the RISC-V
//! `fence` instruction orders CPU-visible stores but does not flush dirty L1
//! data cache lines to L2 or DDR. Cross-hart, cross-shire, and host-visible
//! coherence therefore require explicit cache management via dedicated CSRs.
//!
//! # Cache hierarchy
//!
//! Each Minion core has a private L1 data cache with 64-byte lines. The L2
//! is shared among all Minions in a shire (512 KB on aifoundry3). The L3 is
//! shared across all compute shires (32 MB on aifoundry3). Host DMA reads
//! bypass all Minion caches and observe only DDR.
//!
//! # Producer/consumer protocol
//!
//! Per PRM Section 8.1.3, software must `fence` before a cache op (to commit
//! all prior CPU stores to L1) and issue `TensorWait(CacheOp)` after (to
//! guarantee the op completed before any subsequent memory access to the
//! affected lines). The high-level functions below handle the TensorWait
//! internally; only the preceding `fence` is the caller's responsibility.
//!
//! ```text
//! // Hart A (producer):
//! // ... write data ...
//! fence();                                          // commit stores to L1
//! unsafe { cache_writeback(ptr as usize, len); }  // flush L1 to DDR + TensorWait
//!
//! // <synchronisation, e.g. via a shared flag + fence on both sides>
//!
//! // Hart B (consumer):
//! fence();                                          // receive synchronisation
//! unsafe { cache_invalidate(ptr as usize, len); }  // discard stale L1 + TensorWait
//! // ... read data ...
//! ```
//!
//! Use [`cache_flush`] when a region may contain both dirty (locally modified)
//! and stale lines, performing writeback then invalidation atomically at the
//! function level.
//!
//! # Cache levels
//!
//! The high-level functions [`cache_writeback`], [`cache_invalidate`], and
//! [`cache_flush`] propagate to main memory ([`CacheDest::Mem`]), which is the
//! safest choice for cross-shire and host-DMA coherence. The lower-level
//! `_to` variants accept an explicit [`CacheDest`] for intra-shire operations
//! that need only reach L2.

use core::arch::asm;

// ---------------------------------------------------------------------------
// CSR addresses (cacheops.h, Ainekko SDK)
// ---------------------------------------------------------------------------

/// `evict_va` CSR: evicts cache lines by virtual address up to a target level.
pub const CSR_EVICT_VA: u16 = 0x89F;
/// `flush_va` CSR: writes back dirty cache lines by virtual address to a target level.
pub const CSR_FLUSH_VA: u16 = 0x8BF;

// ---------------------------------------------------------------------------
// Cache destination enum
// ---------------------------------------------------------------------------

/// Target cache hierarchy level for cache management operations.
///
/// Specifies how far up the cache hierarchy a writeback or eviction
/// propagates. Use [`Mem`](CacheDest::Mem) for host-DMA visibility;
/// [`L2`](CacheDest::L2) to make data visible to other Minions in the same
/// shire without a full writeback to DDR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum CacheDest {
    /// Propagate to L1 only (reserved; provided for completeness).
    L1  = 0,
    /// Propagate to the shire-local L2 shared cache.
    L2  = 1,
    /// Propagate to the globally shared L3 cache.
    L3  = 2,
    /// Propagate to main memory (DDR); required for host DMA visibility.
    Mem = 3,
}

// ---------------------------------------------------------------------------
// Hardware primitives
// ---------------------------------------------------------------------------

/// Issues a single `evict_va` CSR write (CSR `0x89F`).
///
/// Evicts `hw_count + 1` cache lines starting at `line_addr` (64-byte
/// aligned), using a stride of 64 bytes per hardware iteration. The
/// hardware reads x31 (t6) implicitly at the moment of the CSR write; this
/// function loads t6 = 64 (stride=64, id=0) immediately before the
/// instruction to satisfy that dependency.
///
/// CSR field layout (cacheops.h `evict_va`):
/// - \[63\]: `use_tmask` = 0
/// - \[59:58\]: `dst` (`CacheDest` discriminant)
/// - \[57:6\]: VA bits \[57:6\] (`line_addr` is 64B-aligned, so bits \[5:0\] = 0)
/// - \[3:0\]: `hw_count` (0..=15, encodes 1..=16 lines)
///
/// x31 layout: `(stride & !63) | id`. For stride=64, id=0: x31 = 64.
///
/// # Safety
/// `line_addr` must be 64-byte aligned. `hw_count` must be in `0..=15`.
#[inline(always)]
unsafe fn evict_va_hw(dst: CacheDest, line_addr: usize, hw_count: u64) {
    let csr_enc: u64 = ((dst as u64) << 58)
        | (line_addr as u64 & 0x0000_FFFF_FFFF_FFC0)
        | hw_count;
    // Omit `nomem`: the asm is treated as a memory barrier; the compiler will
    // not move loads/stores across it.
    unsafe {
        asm!(
            "mv t6, {x31val}",
            "csrw 0x89f, {csr_enc}",
            x31val  = in(reg) 64_u64,
            csr_enc = in(reg) csr_enc,
            out("t6") _,
            options(nostack, preserves_flags),
        );
    }
}

/// Issues a single `flush_va` CSR write (CSR `0x8BF`).
///
/// Writes back `hw_count + 1` dirty cache lines to `dst`; the lines remain
/// cached as clean. All parameters and the CSR field layout are identical to
/// [`evict_va_hw`], differing only in the CSR address.
///
/// # Safety
/// `line_addr` must be 64-byte aligned. `hw_count` must be in `0..=15`.
#[inline(always)]
unsafe fn flush_va_hw(dst: CacheDest, line_addr: usize, hw_count: u64) {
    let csr_enc: u64 = ((dst as u64) << 58)
        | (line_addr as u64 & 0x0000_FFFF_FFFF_FFC0)
        | hw_count;
    unsafe {
        asm!(
            "mv t6, {x31val}",
            "csrw 0x8bf, {csr_enc}",
            x31val  = in(reg) 64_u64,
            csr_enc = in(reg) csr_enc,
            out("t6") _,
            options(nostack, preserves_flags),
        );
    }
}

/// Waits for all outstanding cache operations to complete.
///
/// Issues `TensorWait(ID=6)` (CSR `0x830`, xs bits \[3:0\] = 6), which stalls
/// the hart until every previously issued `evict_va`, `flush_va`,
/// `prefetch_va`, and `TensorLoadL2Scp` has completed. Required after any
/// cache management instruction and before any subsequent memory access to the
/// affected cache lines (PRM Table 9-2, event code 6; PRM Section 8.1.3).
#[inline(always)]
fn wait_cacheops() {
    // Only compiled for the device target; host-side unit tests see a no-op.
    #[cfg(target_arch = "riscv64")]
    // SAFETY: csrrw to the U-mode-accessible TensorWait CSR (0x830) with
    // EVENT=6 stalls the hart until cache ops complete; no memory effects
    // other than the ordering it enforces.
    unsafe {
        asm!(
            "csrrw x0, 0x830, {xs}",
            xs = in(reg) 6_u64,
            options(nostack, preserves_flags),
        );
    }
}

// ---------------------------------------------------------------------------
// Range helpers
// ---------------------------------------------------------------------------

/// Number of 64-byte cache lines covering the byte range `[addr, addr + len)`.
///
/// The result accounts for a partially covered first line: if `addr` is not
/// 64-byte aligned, the first line begins at `addr & !63`.
#[inline(always)]
fn line_count(addr: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let line_start = addr & !63;
    let line_end   = (addr + len + 63) & !63;
    (line_end - line_start) >> 6
}

/// Evicts all cache lines in `[addr, addr + len)` to `dst`, in batches of 16.
///
/// The hardware field `hw_count` is 0-indexed (0 = 1 line, 15 = 16 lines).
/// Each batch issues one `evict_va` CSR write covering `batch` lines.
#[inline]
fn do_evict(dst: CacheDest, addr: usize, len: usize) {
    let n = line_count(addr, len);
    if n == 0 {
        return;
    }
    let mut line = addr & !63;
    let mut rem  = n;
    while rem > 0 {
        let batch = rem.min(16);
        // SAFETY: `line` is 64-byte aligned; `batch - 1` is in 0..=15.
        unsafe { evict_va_hw(dst, line, (batch - 1) as u64); }
        line += batch * 64;
        rem  -= batch;
    }
}

/// Writes back all dirty cache lines in `[addr, addr + len)` to `dst`,
/// in batches of 16.
#[inline]
fn do_flush(dst: CacheDest, addr: usize, len: usize) {
    let n = line_count(addr, len);
    if n == 0 {
        return;
    }
    let mut line = addr & !63;
    let mut rem  = n;
    while rem > 0 {
        let batch = rem.min(16);
        // SAFETY: `line` is 64-byte aligned; `batch - 1` is in 0..=15.
        unsafe { flush_va_hw(dst, line, (batch - 1) as u64); }
        line += batch * 64;
        rem  -= batch;
    }
}

// ---------------------------------------------------------------------------
// Public API - high-level (always targets main memory)
// ---------------------------------------------------------------------------

/// Writes back dirty L1 cache lines in `[addr, addr + len)` to main memory.
///
/// Issues `flush_va` for every covered line, then stalls via `TensorWait(6)`
/// until all writeback traffic has reached DDR. After this call, the flushed
/// data is visible to host DMA and to other shires reading from DDR.
/// The lines remain cached as clean.
///
/// Callers must issue [`crate::fence`] before this function to commit all
/// prior CPU stores to L1 (PRM Section 8.1.3).
///
/// Equivalent to [`cache_writeback_to`]`(CacheDest::Mem, addr, len)`.
///
/// # Safety
/// `addr` must be a valid virtual address; `[addr, addr + len)` must lie
/// within device memory accessible to this hart.
#[inline]
pub unsafe fn cache_writeback(addr: usize, len: usize) {
    do_flush(CacheDest::Mem, addr, len);
    wait_cacheops();
}

/// Invalidates (evicts) L1 cache lines in `[addr, addr + len)`.
///
/// Issues `evict_va` for every covered line, then stalls via `TensorWait(6)`
/// until all eviction traffic is complete. Subsequent loads to the range will
/// fetch fresh data from DDR. Issue on the consumer side of a cross-hart or
/// host-DMA coherence protocol after receiving the producer's synchronisation
/// signal and before reading the produced data.
///
/// Callers must issue [`crate::fence`] before this function (PRM Section
/// 8.1.3).
///
/// Equivalent to [`cache_invalidate_to`]`(CacheDest::Mem, addr, len)`.
///
/// # Safety
/// `addr` must be a valid virtual address; `[addr, addr + len)` must lie
/// within device memory accessible to this hart. Invalidating dirty lines
/// without a prior writeback discards uncommitted data; use [`cache_flush`]
/// when lines may be dirty.
#[inline]
pub unsafe fn cache_invalidate(addr: usize, len: usize) {
    do_evict(CacheDest::Mem, addr, len);
    wait_cacheops();
}

/// Writes back then invalidates L1 cache lines in `[addr, addr + len)`.
///
/// Issues `flush_va` for every covered line followed by `evict_va` for the
/// same lines, then stalls via `TensorWait(6)`. Use when the calling hart has
/// both dirty data to publish and potentially stale lines to discard.
///
/// Callers must issue [`crate::fence`] before this function (PRM Section
/// 8.1.3).
///
/// # Safety
/// `addr` must be a valid virtual address; `[addr, addr + len)` must lie
/// within device memory accessible to this hart.
#[inline]
pub unsafe fn cache_flush(addr: usize, len: usize) {
    do_flush(CacheDest::Mem, addr, len);
    do_evict(CacheDest::Mem, addr, len);
    wait_cacheops();
}

// ---------------------------------------------------------------------------
// Public API - lower-level (explicit destination)
// ---------------------------------------------------------------------------

/// Writes back dirty cache lines in `[addr, addr + len)` to `dst`.
///
/// Lower-level variant of [`cache_writeback`] with an explicit destination.
/// Issues `flush_va` then `TensorWait(6)`. Pass [`CacheDest::L2`] to make
/// data visible to other Minions in the same shire without propagating to DDR.
///
/// # Safety
/// Same constraints as [`cache_writeback`].
#[inline]
pub unsafe fn cache_writeback_to(dst: CacheDest, addr: usize, len: usize) {
    do_flush(dst, addr, len);
    wait_cacheops();
}

/// Invalidates cache lines in `[addr, addr + len)`, evicting to `dst`.
///
/// Lower-level variant of [`cache_invalidate`] with an explicit destination.
/// Issues `evict_va` then `TensorWait(6)`.
///
/// # Safety
/// Same constraints as [`cache_invalidate`].
#[inline]
pub unsafe fn cache_invalidate_to(dst: CacheDest, addr: usize, len: usize) {
    do_evict(dst, addr, len);
    wait_cacheops();
}

// ---------------------------------------------------------------------------
// Tests (host-only; do not touch the hardware CSRs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_count_zero_len() {
        assert_eq!(line_count(0x1000, 0), 0);
    }

    #[test]
    fn line_count_aligned_exact() {
        // Exactly 1, 2, 3 cache lines starting at a 64-byte boundary.
        assert_eq!(line_count(0x100, 64),  1);
        assert_eq!(line_count(0x100, 128), 2);
        assert_eq!(line_count(0x100, 192), 3);
    }

    #[test]
    fn line_count_unaligned_addr_single_line() {
        // addr=0x110 (offset 16 within a line), len=48: range is 0x110..0x140,
        // wholly within the single line 0x100..0x140.
        assert_eq!(line_count(0x110, 48), 1);
    }

    #[test]
    fn line_count_unaligned_addr_two_lines() {
        // addr=0x110, len=64: range is 0x110..0x150, crosses the 0x140 boundary.
        assert_eq!(line_count(0x110, 64), 2);
    }

    #[test]
    fn line_count_one_byte_past_boundary() {
        // A single byte at the start of a new cache line adds exactly one line.
        assert_eq!(line_count(0x100, 65), 2);
    }

    #[test]
    fn cache_dest_discriminants() {
        assert_eq!(CacheDest::L1  as u64, 0);
        assert_eq!(CacheDest::L2  as u64, 1);
        assert_eq!(CacheDest::L3  as u64, 2);
        assert_eq!(CacheDest::Mem as u64, 3);
    }

    #[test]
    fn evict_csr_encoding() {
        // Verify the CSR encoding for a 64-byte-aligned address with Mem dest.
        let addr:     usize = 0x0000_8000_0001_0000; // 64B-aligned
        let hw_count: u64   = 15;                    // 16 lines
        let dst             = CacheDest::Mem;
        let csr_enc: u64    = ((dst as u64) << 58)
            | (addr as u64 & 0x0000_FFFF_FFFF_FFC0)
            | hw_count;
        // dst=3 at bits 59:58
        assert_eq!((csr_enc >> 58) & 0x3, 3);
        // hw_count at bits 3:0
        assert_eq!(csr_enc & 0xF, 15);
        // addr embedded at bits 57:6 (addr is 64B-aligned, bits 5:0 = 0)
        assert_eq!(csr_enc & (addr as u64), addr as u64);
    }

    #[test]
    fn flush_csr_encoding_matches_evict_layout() {
        // flush_va (0x8BF) uses the same field layout as evict_va (0x89F);
        // verify the encoding formula produces the same bit pattern.
        let addr     = 0x0000_8000_0002_0000_usize;
        let hw_count = 7_u64;
        let dst      = CacheDest::L2;
        let enc = ((dst as u64) << 58)
            | (addr as u64 & 0x0000_FFFF_FFFF_FFC0)
            | hw_count;
        assert_eq!((enc >> 58) & 0x3, CacheDest::L2 as u64);
        assert_eq!(enc & 0xF, 7);
    }
}
