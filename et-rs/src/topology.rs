//! Compute topology of the ET-SoC-1.
//!
//! Some of the topology is architectural and fixed for the part (how many harts
//! a shire and a neighbourhood contain); the rest is per-device (which compute
//! shires are present) and is queried through [`crate::transport::DeviceConfig`].
//! [`crate::Device::topology`] combines the two into a [`Topology`], so callers
//! can size work to the device instead of hard-coding constants.

/// Harts per compute shire on the ET-SoC-1 (architectural constant).
pub const HARTS_PER_SHIRE: u32 = 64;

/// Harts per neighbourhood on the ET-SoC-1 (architectural constant).
pub const HARTS_PER_NEIGHBOURHOOD: u32 = 16;

/// The device's compute topology: which shires are present, and the fixed
/// per-shire geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    /// Bitmask of compute shires present/enabled on this device.
    pub shire_mask: u64,
    /// Harts per compute shire ([`HARTS_PER_SHIRE`]).
    pub harts_per_shire: u32,
    /// Harts per neighbourhood ([`HARTS_PER_NEIGHBOURHOOD`]).
    pub harts_per_neighbourhood: u32,
    /// Cache line size in bytes.
    pub cache_line: u32,
}

impl Topology {
    /// Number of compute shires present.
    pub fn num_shires(&self) -> u32 {
        self.shire_mask.count_ones()
    }

    /// Total harts available across all present shires.
    pub fn num_harts(&self) -> u32 {
        self.num_shires() * self.harts_per_shire
    }

    /// The lowest-numbered present shire, as a single-shire mask, or `0` if the
    /// device reports no shires. Useful for launching on one shire.
    pub fn first_shire(&self) -> u64 {
        // Isolate the lowest set bit.
        self.shire_mask & self.shire_mask.wrapping_neg()
    }
}
