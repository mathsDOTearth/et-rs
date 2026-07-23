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
//! # Quick start
//!
//! Add `et-rs = "0.1"` to `Cargo.toml` (the library is named `et_soc1` in Rust
//! source, matching the C/C++ SDK convention), then:
//!
//! ```no_run
//! use et_soc1::{Device, Error, LaunchOptions, TraceConfig};
//! use et_soc1::trace::{DecodedEntry, TraceBuffer};
//!
//! fn main() -> et_soc1::Result<()> {
//!     let elf = std::fs::read("hello.elf").map_err(|e| Error::Io {
//!         op: "read kernel ELF",
//!         source: e,
//!     })?;
//!
//!     // Open device 0 (/dev/et0_ops) and query its DRAM geometry.
//!     let device = Device::open(0)?;
//!
//!     // Load the kernel before any alloc() calls so it lands at the DRAM
//!     // base, matching its link address.
//!     let kernel = device.load_kernel(&elf)?;
//!
//!     // Allocate an 8 MiB trace buffer in device DRAM.
//!     let trace_buf = device.alloc(8 * 1024 * 1024)?;
//!
//!     // Launch on shire 0 with a barrier and full user tracing.
//!     let opts = LaunchOptions::new(0x1)
//!         .with_trace(TraceConfig::full(trace_buf, 0x1))
//!         .with_args(vec![0u8; 64]);
//!     device.launch(&kernel, &opts)?;
//!
//!     // Copy the trace buffer to host memory and decode it.
//!     let mut host = vec![0u8; trace_buf.size as usize];
//!     device.memcpy_d2h(trace_buf.addr, &mut host)?;
//!
//!     for entry in TraceBuffer::parse(&host)?.entries() {
//!         if let DecodedEntry::String(s) = entry.decoded() {
//!             println!("[hart {}] {}", entry.hart_id, s.trim_end());
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! Substitute `Device::open_emulator("/opt/et", "/tmp/et-run")?` for
//! `Device::open(0)?` when building with `--features emu` to target the SDK
//! software emulator rather than real hardware.
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
