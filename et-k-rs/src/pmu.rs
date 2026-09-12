//! Performance Monitoring Unit (PMU) counter API for the ET-SoC-1 Minion core.
//!
//! The ET-SoC-1 implements a subset of the RISC-V Zihpm extension (PRM
//! section 1.3.2). The PMU is shared across 8 Minions in a neighbourhood and
//! provides **six** counters per hart, not the full 3-31 range that the
//! RISC-V spec permits:
//!
//! - `hpmcounter3`-`hpmcounter6` (`mhpmevent3`-`mhpmevent6`): Minion-level
//!   events from [`PmuEvent`] -- one event per counter, configured by firmware.
//! - `hpmcounter7`-`hpmcounter8` (`mhpmevent7`-`mhpmevent8`): neighbourhood-
//!   level events from [`NeighborhoodEvent`] -- shared across the 8-Minion
//!   neighbourhood; when different harts program different events the lower
//!   `mhartid` wins.
//! - `hpmcounter9`-`hpmcounter31`: tied to 0 on this implementation.
//!
//! # Note on `mcycle` and `minstret`
//!
//! The standard `mcycle` (CSR `0xC00`) and `minstret` (CSR `0xC02`) counters
//! are **permanently zero** on the ET-SoC-1 (PRM section 1.3.2). Use
//! `hpmcounter3` (or any of 3-6) configured with [`PmuEvent::Cycles`] to
//! count clock cycles, and [`PmuEvent::RetiredInst0`] / [`PmuEvent::RetiredInst1`]
//! to count retired instructions. The firmware on aifoundry3 assigns
//! `PmuEvent::Cycles` to `hpmcounter3` by default, which is why
//! [`crate::timestamp`] reads CSR `0xC03`.
//!
//! # Enabling the PMU
//!
//! The PMU must be enabled by firmware (an M-mode ESR write) before any
//! counter increments. U-mode code can read counts but typically cannot
//! reconfigure `mhpmeventN` without M-mode delegation.
//!
//! # Usage pattern
//!
//! ```no_run
//! use et_kernel::pmu::{PmuEvent, pmu_read};
//!
//! // Read counter 4 before and after a tensor operation.
//! // (Assumes firmware has assigned PmuEvent::TfmaWaitTenb to mhpmevent4.)
//! let before = pmu_read(4);
//! // ... tensor operations ...
//! let after  = pmu_read(4);
//! let delta  = after.wrapping_sub(before);
//! ```

use core::arch::asm;

// ---------------------------------------------------------------------------
// PMU event codes
// ---------------------------------------------------------------------------

/// Minion-level PMU event codes (PRM section 1.3.2, Table 1-3).
///
/// Written to `mhpmeventN` (CSR `0x320 + N`, for N in `3..=6`) to select what
/// `hpmcounterN` accumulates. Firmware or a privileged shim configures the
/// mapping; U-mode code reads counts via [`pmu_read`] and typically cannot
/// write `mhpmeventN` without M-mode delegation.
///
/// These events apply only to `hpmcounter3`-`hpmcounter6`. For
/// neighbourhood-level events on `hpmcounter7`-`hpmcounter8`, use
/// [`NeighborhoodEvent`].
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmuEvent {
    /// No event; counter does not increment.
    NoEvent = 0,
    /// Clock cycles executed by the core.
    ///
    /// Use this event in `mhpmevent3`-`mhpmevent6` to count cycles;
    /// `mcycle` (CSR `0xC00`) is permanently zero on this implementation.
    Cycles = 1,
    /// An instruction retired by thread 0 of the core.
    ///
    /// `minstret` (CSR `0xC02`) is permanently zero; use this event instead.
    RetiredInst0 = 2,
    /// An instruction retired by thread 1 of the core.
    ///
    /// `minstret` (CSR `0xC02`) is permanently zero; use this event instead.
    RetiredInst1 = 3,
    /// A branch taken by thread 0 of the core.
    Branches0 = 4,
    /// A branch taken by thread 1 of the core.
    Branches1 = 5,
    /// A load/store by thread 0 accessed the data cache (hit or miss).
    ///
    /// Excludes tensor and cache-management operations.
    DcacheAccess0 = 6,
    /// A load/store by thread 1 accessed the data cache (hit or miss).
    ///
    /// Excludes tensor and cache-management operations.
    DcacheAccess1 = 7,
    /// A load/store by thread 0 missed in the data cache.
    ///
    /// Excludes tensor and cache-management operations.
    DcacheMisses0 = 8,
    /// A load/store by thread 1 missed in the data cache.
    ///
    /// Excludes tensor and cache-management operations.
    DcacheMisses1 = 9,
    /// The data cache sent a miss request to the L2 cache.
    L2MissReq = 10,
    /// The L2 cache rejected a miss request from the data cache.
    L2MissReqRej = 11,
    /// The data cache sent an evict request to the L2 cache.
    L2EvictReq = 12,
    /// The L2 cache rejected an evict request from the data cache.
    L2EvictReqRej = 13,
    /// Started execution of a TensorLoad instruction.
    TlInst = 14,
    /// A TensorLoad sent a request to the L2 cache.
    TlOps = 15,
    /// Started execution of a TensorStore instruction.
    TsInst = 16,
    /// A TensorStore sent a request to the L2 cache.
    TsOps = 17,
    /// Cycles a TensorFMA paired with TensorLoadB was blocked waiting for data
    /// from L2. Measures the B-load serialisation cost; high values indicate
    /// that the crossbar or DRAM is the bottleneck for B tiles.
    TfmaWaitTenb = 18,
    /// Started execution of a micro-op generated by a TensorIMA8A32 instruction.
    TimaOps = 19,
    /// Retired a micro-op generated by a TensorFMA16A32 instruction.
    TxFma3216Ops = 20,
    /// Retired an FP instruction (packed or scalar), integer multiplication,
    /// or a micro-op from TensorFMA32.
    TxFma32Ops = 21,
    /// Retired a packed integer instruction, int-to-FP conversion, or micro-op
    /// from integer TensorQuant.
    TxFmaIntOps = 22,
    /// Retired a micro-op generated by a transcendental instruction.
    TransOps = 23,
    /// Retired a packed integer instruction or a micro-op from TensorFMA32.
    ShortOps = 24,
    /// Retired a mask instruction.
    MaskOps = 25,
    /// Started execution of a TensorFMA instruction.
    TfmaInst = 26,
    /// Started execution of a tensor reduction instruction.
    TreduceInst = 27,
    /// Started execution of a TensorQuant instruction.
    TquantInst = 28,
}

/// Neighbourhood-level PMU event codes (PRM section 1.3.2, Table 1-4).
///
/// Written to `mhpmevent7` or `mhpmevent8` to select what `hpmcounter7` or
/// `hpmcounter8` accumulates. The counter is shared across the 8-Minion
/// neighbourhood; when harts program different events the lower `mhartid`
/// hart's choice takes precedence.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborhoodEvent {
    /// No event; counter does not increment.
    NoEvent = 0,
    /// Any Minion sent an ET Link request.
    EtLinkSend = 1,
    /// Any Minion received an ET Link response.
    EtLinkRecv = 2,
    /// A cooperative load request was sent.
    CoopLoadSend = 3,
    /// An inter-neighbourhood cooperative load request was sent.
    CoopLoadInterNeighSend = 4,
    /// A cooperative load response was received.
    CoopLoadRecv = 5,
    /// A cooperative store request was sent.
    CoopStoreSend = 6,
    /// A cooperative store response was received.
    CoopStoreRecv = 7,
    /// Any Minion sent a request to the I-cache.
    IcacheReqSend = 8,
    /// Any Minion received a response from the I-cache.
    IcacheRespRecv = 9,
    /// Any Minion sent a request to the page table walker.
    PtwReqSend = 10,
    /// Any Minion received a response from the page table walker.
    PtwRespRecv = 11,
    /// A message was sent between Minions through the FLN.
    FlnMsg = 12,
    /// The I-cache sent an ET Link request.
    IcacheEtLinkSend = 13,
    /// The I-cache received an ET Link response.
    IcacheEtLinkRecv = 14,
    /// The I-cache sent a request to the L1 data SRAM.
    IcacheL1Req = 15,
    /// The I-cache received a response from the L1 data SRAM.
    IcacheL1Resp = 16,
    /// Any PTW sent an ET Link request.
    PtwEtLinkSend = 17,
    /// Any PTW received an ET Link response.
    PtwEtLinkRecv = 18,
    // Codes 19-20 are reserved.
    /// An ET Link request was pushed into the intermediate FIFO.
    EtLinkFifoIn = 21,
    /// An ET Link request was pushed into any BANK/UC FIFO.
    EtLinkBankFifoIn = 22,
    /// An ET Link response was received from the SC/UC input.
    EtLinkScUcIn = 23,
}

// ---------------------------------------------------------------------------
// CSR read helper macro
// ---------------------------------------------------------------------------

// Reads an hpmcounterN CSR where N is a compile-time literal, with the
// RTLMIN-6496 workaround: four back-to-back reads of the same CSR in a
// 16-byte-aligned block. The first three reads are discarded; the fourth
// is the architecturally correct value. `.align 4` aligns the block to
// 2^4 = 16 bytes. `nomem` is omitted so the compiler treats the block as
// a potential memory barrier, preventing it from reordering other
// loads/stores across the four reads.
macro_rules! csr_read {
    ($csr:literal) => {{
        let v: u64;
        // SAFETY: csrrs with rs1 = x0 reads without side effect.
        // The four reads must be consecutive in the instruction stream;
        // placing them in one asm block prevents the compiler inserting
        // any intervening instructions.
        unsafe {
            asm!(
                ".align 4",
                concat!("csrrs {v0}, ", stringify!($csr), ", x0"),
                concat!("csrrs {v1}, ", stringify!($csr), ", x0"),
                concat!("csrrs {v2}, ", stringify!($csr), ", x0"),
                concat!("csrrs {v},  ", stringify!($csr), ", x0"),
                v0 = out(reg) _,
                v1 = out(reg) _,
                v2 = out(reg) _,
                v  = out(reg) v,
                options(nostack, preserves_flags),
            );
        }
        v
    }};
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read the `mcycle` counter (CSR `0xC00`).
///
/// **Always returns 0 on the ET-SoC-1.** The standard `mcycle` CSR is
/// permanently tied to zero on this implementation (PRM section 1.3.2).
/// To count clock cycles, read `hpmcounter3` via [`pmu_read`]`(3)` after
/// firmware has assigned [`PmuEvent::Cycles`] to `mhpmevent3`. On aifoundry3
/// the firmware assigns this event by default; [`crate::timestamp`] relies on
/// it.
#[inline(always)]
pub fn pmu_read_cycle() -> u64 {
    csr_read!(0xC00)
}

/// Read the `minstret` counter (CSR `0xC02`).
///
/// **Always returns 0 on the ET-SoC-1.** The standard `minstret` CSR is
/// permanently tied to zero on this implementation (PRM section 1.3.2).
/// To count retired instructions, read an `hpmcounter3`-`hpmcounter6` via
/// [`pmu_read`] after firmware has assigned [`PmuEvent::RetiredInst0`] or
/// [`PmuEvent::RetiredInst1`] to the corresponding `mhpmeventN`.
#[inline(always)]
pub fn pmu_read_instret() -> u64 {
    csr_read!(0xC02)
}

/// Read hardware performance counter `N` (`hpmcounterN`, CSR `0xC03 + (N-3)`).
///
/// `counter` must be in `3..=8` on the ET-SoC-1; counters 9-31 are tied to 0
/// by the hardware. Values outside `3..=31` also return 0.
///
/// The semantics of the count depend on the event assigned to counter N by
/// firmware via `mhpmeventN`:
/// - Counters 3-6: Minion-level events from [`PmuEvent`].
/// - Counters 7-8: neighbourhood-level events from [`NeighborhoodEvent`].
///
/// Counter 3 (`hpmcounter3`, CSR `0xC03`) is also used by [`crate::timestamp`].
#[inline(always)]
pub fn pmu_read(counter: u8) -> u64 {
    match counter {
        3  => csr_read!(0xC03),
        4  => csr_read!(0xC04),
        5  => csr_read!(0xC05),
        6  => csr_read!(0xC06),
        7  => csr_read!(0xC07),
        8  => csr_read!(0xC08),
        9  => csr_read!(0xC09),
        10 => csr_read!(0xC0A),
        11 => csr_read!(0xC0B),
        12 => csr_read!(0xC0C),
        13 => csr_read!(0xC0D),
        14 => csr_read!(0xC0E),
        15 => csr_read!(0xC0F),
        16 => csr_read!(0xC10),
        17 => csr_read!(0xC11),
        18 => csr_read!(0xC12),
        19 => csr_read!(0xC13),
        20 => csr_read!(0xC14),
        21 => csr_read!(0xC15),
        22 => csr_read!(0xC16),
        23 => csr_read!(0xC17),
        24 => csr_read!(0xC18),
        25 => csr_read!(0xC19),
        26 => csr_read!(0xC1A),
        27 => csr_read!(0xC1B),
        28 => csr_read!(0xC1C),
        29 => csr_read!(0xC1D),
        30 => csr_read!(0xC1E),
        31 => csr_read!(0xC1F),
        _  => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify PmuEvent discriminants match PRM section 1.3.2, Table 1-3.
    #[test]
    fn pmu_event_discriminants() {
        assert_eq!(PmuEvent::NoEvent       as u64,  0);
        assert_eq!(PmuEvent::Cycles        as u64,  1);
        assert_eq!(PmuEvent::RetiredInst0  as u64,  2);
        assert_eq!(PmuEvent::RetiredInst1  as u64,  3);
        assert_eq!(PmuEvent::Branches0     as u64,  4);
        assert_eq!(PmuEvent::Branches1     as u64,  5);
        assert_eq!(PmuEvent::DcacheAccess0 as u64,  6);
        assert_eq!(PmuEvent::DcacheAccess1 as u64,  7);
        assert_eq!(PmuEvent::DcacheMisses0 as u64,  8);
        assert_eq!(PmuEvent::DcacheMisses1 as u64,  9);
        assert_eq!(PmuEvent::L2MissReq     as u64, 10);
        assert_eq!(PmuEvent::L2MissReqRej  as u64, 11);
        assert_eq!(PmuEvent::L2EvictReq    as u64, 12);
        assert_eq!(PmuEvent::L2EvictReqRej as u64, 13);
        assert_eq!(PmuEvent::TlInst        as u64, 14);
        assert_eq!(PmuEvent::TlOps         as u64, 15);
        assert_eq!(PmuEvent::TsInst        as u64, 16);
        assert_eq!(PmuEvent::TsOps         as u64, 17);
        assert_eq!(PmuEvent::TfmaWaitTenb  as u64, 18);
        assert_eq!(PmuEvent::TimaOps       as u64, 19);
        assert_eq!(PmuEvent::TxFma3216Ops  as u64, 20);
        assert_eq!(PmuEvent::TxFma32Ops    as u64, 21);
        assert_eq!(PmuEvent::TxFmaIntOps   as u64, 22);
        assert_eq!(PmuEvent::TransOps      as u64, 23);
        assert_eq!(PmuEvent::ShortOps      as u64, 24);
        assert_eq!(PmuEvent::MaskOps       as u64, 25);
        assert_eq!(PmuEvent::TfmaInst      as u64, 26);
        assert_eq!(PmuEvent::TreduceInst   as u64, 27);
        assert_eq!(PmuEvent::TquantInst    as u64, 28);
    }

    /// Verify NeighborhoodEvent discriminants match PRM section 1.3.2, Table 1-4.
    #[test]
    fn neighborhood_event_discriminants() {
        assert_eq!(NeighborhoodEvent::NoEvent               as u64,  0);
        assert_eq!(NeighborhoodEvent::EtLinkSend            as u64,  1);
        assert_eq!(NeighborhoodEvent::EtLinkRecv            as u64,  2);
        assert_eq!(NeighborhoodEvent::CoopLoadSend          as u64,  3);
        assert_eq!(NeighborhoodEvent::CoopLoadInterNeighSend as u64, 4);
        assert_eq!(NeighborhoodEvent::CoopLoadRecv          as u64,  5);
        assert_eq!(NeighborhoodEvent::CoopStoreSend         as u64,  6);
        assert_eq!(NeighborhoodEvent::CoopStoreRecv         as u64,  7);
        assert_eq!(NeighborhoodEvent::IcacheReqSend         as u64,  8);
        assert_eq!(NeighborhoodEvent::IcacheRespRecv        as u64,  9);
        assert_eq!(NeighborhoodEvent::PtwReqSend            as u64, 10);
        assert_eq!(NeighborhoodEvent::PtwRespRecv           as u64, 11);
        assert_eq!(NeighborhoodEvent::FlnMsg                as u64, 12);
        assert_eq!(NeighborhoodEvent::IcacheEtLinkSend      as u64, 13);
        assert_eq!(NeighborhoodEvent::IcacheEtLinkRecv      as u64, 14);
        assert_eq!(NeighborhoodEvent::IcacheL1Req           as u64, 15);
        assert_eq!(NeighborhoodEvent::IcacheL1Resp          as u64, 16);
        assert_eq!(NeighborhoodEvent::PtwEtLinkSend         as u64, 17);
        assert_eq!(NeighborhoodEvent::PtwEtLinkRecv         as u64, 18);
        assert_eq!(NeighborhoodEvent::EtLinkFifoIn          as u64, 21);
        assert_eq!(NeighborhoodEvent::EtLinkBankFifoIn      as u64, 22);
        assert_eq!(NeighborhoodEvent::EtLinkScUcIn          as u64, 23);
    }
}
