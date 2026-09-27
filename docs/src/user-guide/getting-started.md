# Getting started

This walks through the smallest useful program: open a device, load a kernel,
launch it on one shire with tracing enabled, and print the decoded trace. It
mirrors [`et-rs/examples/hello.rs`](https://github.com/mathsDOTearth/et-rs/blob/main/et-rs/examples/hello.rs),
which you can run directly.

## Opening a device

To keep a program buildable both with and without hardware, select the backend
by `cfg`. This is the pattern every example uses:

```rust,ignore
#[cfg(feature = "emu")]
fn open_device() -> et_soc1::Result<et_soc1::Device<et_soc1::FfiTransport>> {
    let sdk_prefix = std::env::var("ET_SDK_PREFIX").unwrap_or_else(|_| "/opt/et".into());
    et_soc1::Device::open_emulator(&sdk_prefix, "sysemu-run")
}

#[cfg(not(feature = "emu"))]
fn open_device() -> et_soc1::Result<et_soc1::Device> {
    et_soc1::Device::open(0) // /dev/et0_ops
}
```

See [`Device::open`] and [`Device::open_emulator`] for the details.

## Load, launch, trace

```rust,ignore
use et_soc1::trace::{DecodedEntry, TraceBuffer};
use et_soc1::{LaunchOptions, TraceConfig};

let device = open_device()?;

// Load a compute-kernel ELF into device DRAM.
let elf = std::fs::read("hello-rs")?;
let kernel = device.load_kernel(&elf)?;

// Reserve a device-side trace buffer and launch on shire 0 with full tracing.
let shire_mask = 0x1;
let trace_buf = device.alloc(8 * 1024 * 1024)?;
let opts = LaunchOptions::new(shire_mask).with_trace(TraceConfig::full(trace_buf, shire_mask));
device.launch(&kernel, &opts)?;

// DMA the trace buffer back and decode it with the pure-Rust decoder.
let mut host = vec![0u8; trace_buf.size as usize];
device.memcpy_d2h(trace_buf.addr, &mut host)?;
for entry in TraceBuffer::parse(&host)?.entries() {
    if let DecodedEntry::String(s) = entry.decoded() {
        println!("[hart {}] {}", entry.hart_id, s.trim_end());
    }
}
```

Running the `hello-rs` kernel this way prints one greeting per participating hart
and decodes the on-device trace entirely in Rust.

## Running the examples

From the repository root (bare `cargo` commands operate on `et-rs`, the default
workspace member):

```bash
K=et-k-rs/target/riscv64imac-unknown-none-elf/release
cargo run --features emu --example hello_sysemu -- $K/hello-rs   # emulator
cargo run                --example hello         -- $K/hello-rs   # real hardware
```

## Where to go next

- [Device memory](device-memory.md): typed buffers for uploading inputs and
  downloading results without byte casts.
- [Launching kernels](launching-kernels.md): launch options, kernel arguments,
  and reading launch failures.

[`Device::open`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.open
[`Device::open_emulator`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.open_emulator
