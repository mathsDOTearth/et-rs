# et-abi

Shared host/device ABI types for the Esperanto ET-SoC-1, used by the
[`et-rs`](https://github.com/mathsDOTearth/et-rs) host driver and the
[`et-k-rs`](https://github.com/mathsDOTearth/et-rs) device-side kernel library.

A kernel launch passes its arguments by pointer: the host stages an argument
struct in device memory and the firmware delivers its address to the kernel (in
register `a0`). This crate defines those argument structs **once**, so the host
launcher and the device kernel cannot drift on field order, sizes, or padding.

Because both the host (x86-64) and the device (RV64) are little-endian, the
in-memory `#[repr(C)]` layout is the wire layout: the host takes the struct's
bytes with [`DeviceArgs::as_bytes`] and the kernel reinterprets the pointer with
[`DeviceArgs::from_ptr`]. No serialisation step is involved.

The crate is `no_std` with no dependencies, so it builds for the host and for the
`riscv64imac-unknown-none-elf` device target alike.

```rust
use et_abi::{DeviceArgs, ReduceArgs};

// Host: build the args the kernel will read.
let args = ReduceArgs { input: 0x8005_0000, out: 0x8006_0000, n: 1024, n_harts: 64 };
let bytes: &[u8] = args.as_bytes();

// Device: recover them from the pointer the firmware passed in a0.
// let args = unsafe { ReduceArgs::from_ptr(a0 as *const u8) };
```

It also holds the architectural constants both sides must agree on:
`CACHE_LINE` (the per-hart output stride, so host padding and device writes
match and avoid false sharing), `HARTS_PER_SHIRE`, and
`HARTS_PER_NEIGHBOURHOOD`.

## Licence

Apache-2.0.
