//! Performance Monitoring Unit (PMU) counter API for the ET-SoC-1 Minion core.
//!
//! The ET-SoC-1 implements the RISC-V Zihpm extension: hardware performance
//! counters accessible from U-mode via the `hpmcounterN` CSRs (PRM Chapter 8).
//! Each counter is a 64-bit read-only accumulator that increments on each
//! occurrence of the event assigned to it by firmware (or `pmu_configure` if
//! U-mode write access to `mhpmeventN` is confirmed).
//!
//! # Available counters
//!
//! - `hpmcounter3` (CSR `0xC03`): also read by [`crate::timestamp`] as a
//!   cycle counter. Whether the assigned event is `cycle` or a custom PMU event
//!   depends on the firmware's `mhpmeventN` configuration.
//! - `hpmcounter4` .. `hpmcounter31` (CSR `0xC04` .. `0xC1F`): available for
//!   application use subject to firmware assignment.
//!
//! # Usage pattern
//!
//! ```no_run
//! use et_kernel::pmu::{PmuEvent, pmu_read};
//!
//! // Read counter 4 before and after a tensor operation; the delta is the
//! // number of TFMA_WAIT_TENB events that occurred (assuming firmware assigned
//! // PmuEvent::TfmaWaitTenb to counter 4 via mhpmevent4).
//! let before = pmu_read(4);
//! // ... tensor operations ...
//! let after  = pmu_read(4);
//! let delta  = after.wrapping_sub(before);
//! ```

use core::arch::asm;

// ---------------------------------------------------------------------------
// PMU event codes (PRM Chapter 8)
// ---------------------------------------------------------------------------

/// PMU event codes for the ET-SoC-1. The value is written to `mhpmeventN`
/// (CSR `0x320 + N`) to select what counter N accumulates.
///
/// Firmware or a privileged shim configures the mapping; U-mode can read
/// the resulting counts via [`pmu_read`] but typically cannot write
/// `mhpmeventN` without M-mode delegation.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmuEvent {
    /// Cycles spent waiting for TenB load to complete before TensorFMA32.
    /// Measures the B-load serialisation cost; high values indicate that
    /// the crossbar or DRAM is the bottleneck for B tiles.
    ///
    /// (PRM Chapter 8, event code 18.)
    TfmaWaitTenb = 18,
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

/// Read the `cycle` counter (CSR `0xC00`).
///
/// Returns the number of cycles elapsed since firmware initialised the
/// counter. Rolls over at 2^64 cycles.
#[inline(always)]
pub fn pmu_read_cycle() -> u64 {
    csr_read!(0xC00)
}

/// Read the `instret` counter (CSR `0xC02`).
///
/// Returns the number of instructions retired since firmware initialised
/// the counter.
#[inline(always)]
pub fn pmu_read_instret() -> u64 {
    csr_read!(0xC02)
}

/// Read hardware performance counter `N` (`hpmcounterN`, CSR `0xC03 + (N-3)`).
///
/// `counter` must be in the range `3..=31`; values outside this range return 0.
/// The semantics of the returned value depend on the event assigned to
/// counter N by firmware via `mhpmeventN`.
///
/// Counter 3 (CSR `0xC03`) is also used by [`crate::timestamp`].
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

    /// Verify that PmuEvent discriminants match PRM Chapter 8 event codes.
    #[test]
    fn pmu_event_discriminants() {
        assert_eq!(PmuEvent::TfmaWaitTenb as u64, 18);
    }
}
