//! Shared host/device ABI for the ET-SoC-1: the kernel-launch argument structs,
//! defined **once** and used by both the host launcher and the device kernel.
//!
//! Kernel arguments are passed by pointer: the host stages an argument struct in
//! device memory and the firmware delivers its address to the kernel (in `a0`).
//! Because both the host (x86-64) and the device (RV64) are little-endian, the
//! in-memory `#[repr(C)]` layout *is* the wire layout, so no explicit
//! serialisation is needed -- the host takes the struct's bytes and the kernel
//! reinterprets the pointer. Defining each struct here keeps the two sides from
//! drifting (mismatched field order, sizes, or padding).

#![no_std]

/// ET-SoC-1 cache-line size, in bytes.
///
/// Per-hart outputs are laid out at this stride on both sides: the host strides
/// its padded arrays by it and the device writes each hart's cell at
/// `base + hart * CACHE_LINE`. Defining it once here keeps the two from drifting,
/// which on this software-coherent part would cause silent false-sharing
/// corruption.
pub const CACHE_LINE: usize = 64;

/// Harts per compute shire on the ET-SoC-1 (architectural constant).
pub const HARTS_PER_SHIRE: u32 = 64;

/// Harts per neighbourhood on the ET-SoC-1 (architectural constant).
pub const HARTS_PER_NEIGHBOURHOOD: u32 = 16;

/// A plain-old-data kernel-argument struct exchanged between host and device.
///
/// # Safety
/// Implementors must be `#[repr(C)]`, contain only integer fields with no
/// padding, and be valid for any bit pattern. Then [`DeviceArgs::as_bytes`] and
/// [`DeviceArgs::from_ptr`] are a faithful round-trip on little-endian hosts and
/// devices.
pub unsafe trait DeviceArgs: Sized + Copy {
    /// Borrow the struct as its on-wire bytes (host side: stage these in device
    /// memory as the launch arguments).
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `Self` is repr(C) POD (trait contract), so its bytes are a
        // valid representation of length `size_of::<Self>()`.
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const u8,
                core::mem::size_of::<Self>(),
            )
        }
    }

    /// Reinterpret a device-memory pointer as these arguments (device side).
    ///
    /// # Safety
    /// `ptr` must point to at least `size_of::<Self>()` bytes of a valid,
    /// suitably aligned instance -- e.g. the launch-args region the firmware
    /// passed in `a0`.
    unsafe fn from_ptr<'a>(ptr: *const u8) -> &'a Self {
        // SAFETY: forwarded to the caller's contract on `ptr`.
        unsafe { &*(ptr as *const Self) }
    }
}

/// Arguments for the data-parallel reduction kernel (`reduce-rs`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReduceArgs {
    /// Device address of the input array (`n` × `u32`).
    pub input: u64,
    /// Device address of the output array (`n_harts` × one `u64` per cache line).
    pub out: u64,
    /// Number of input elements.
    pub n: u32,
    /// Number of participating harts.
    pub n_harts: u32,
}

// SAFETY: repr(C), only u64/u32 fields ordered by decreasing size -> no padding.
unsafe impl DeviceArgs for ReduceArgs {}
const _: () = assert!(core::mem::size_of::<ReduceArgs>() == 24);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_args_roundtrip() {
        let a = ReduceArgs {
            input: 0x0080_0580_1000,
            out: 0x0080_0590_0000,
            n: 262_144,
            n_harts: 64,
        };
        let bytes = a.as_bytes();
        assert_eq!(bytes.len(), 24);
        // The device would do exactly this from the args pointer.
        let b = unsafe { ReduceArgs::from_ptr(bytes.as_ptr()) };
        assert_eq!(*b, a);
    }
}
