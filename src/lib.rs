//! Host-side Rust interface to the Esperanto ET-SoC-1 RISC-V accelerator.
//!
//! This crate speaks the ET-SoC-1 PCIe kernel driver's character-device
//! protocol directly: it opens `/dev/etN_ops`, discovers the device DRAM region,
//! loads kernels, launches them on selected shires, moves results back over DMA,
//! and extracts on-device trace buffers, all without the vendor C++ runtime.
//!
//! # Layers
//!
//! * [`Device`] is the high-level handle. It owns a [`transport::Transport`],
//!   a DRAM bump allocator, and the command tag counter, and exposes
//!   [`Device::load_kernel`], [`Device::alloc`], [`Device::launch`],
//!   [`Device::memcpy_d2h`] and [`Device::extract_cm_trace`].
//! * [`transport`] abstracts the command channel. [`transport::IoctlTransport`]
//!   targets real hardware; under the `emu` feature, `transport::FfiTransport`
//!   drives the SDK software emulator through the vendor C++ device-layer, so a
//!   host without a card can develop and test.
//! * [`trace`] is a standalone, pure-Rust decoder for the et-trace buffer
//!   layout, usable on any captured buffer independently of a live device.
//!
//! # Hardware versus emulator
//!
//! The default ioctl backend requires a real card with the `et` kernel driver
//! loaded. The `emu` feature adds an FFI backend over the vendor device-layer's
//! emulator (`Device::open_emulator`), which needs neither hardware nor the
//! driver. Both paths build the identical device-ops command bytes, so the
//! [`proto`], [`device`] and [`trace`] code is exercised the same way against
//! each. The [`trace`] decoder and the [`proto`] builders are additionally
//! testable with no device at all.

mod elf;
mod error;
mod ffi;
mod ioctl;

pub mod device;
pub mod proto;
pub mod trace;
pub mod transport;

pub use device::{
    Device, DeviceRegion, LaunchOptions, LaunchResult, LaunchTiming, LoadedKernel, TraceConfig,
};
pub use error::{Error, Result};
pub use transport::{DramInfo, IoctlTransport, PoppedResponse, Transport};

#[cfg(feature = "emu")]
pub use transport::FfiTransport;
