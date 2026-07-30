# Installation

The host crate is `et-rs` (library name `et_soc1`). Add it to a project with:

```toml
[dependencies]
et-rs = "0.2"
```

The FFI bindings to the SDK are vendored in the crate, so a plain `cargo build`
needs neither the SDK nor `bindgen` and works anywhere, including docs.rs. The
SDK is required only at run time (to talk to a device or the emulator) and, for
two optional build paths, at build time.

## Choosing a backend

`et_soc1::Device` is generic over a `Transport`. Two backends are provided:

| Backend | Selected by | Needs |
|---------|-------------|-------|
| `IoctlTransport` (default) | `Device::open(0)` | A physical card with the `et` driver, exposing `/dev/etN_ops`. |
| `FfiTransport` | `Device::open_emulator(..)`, `emu` feature | The SDK C++ device-layer and a CMake toolchain; no hardware. |

The emulator backend lets you build and test the same code that will drive the
real device, so a developer without a card is not blocked. See
[Getting started](getting-started.md) for the `open_device` pattern that selects
the backend by `cfg`.

## Run-time requirements, real hardware (default)

`Device::open` talks to the PCIe kernel driver, so you need:

- a physical ET-SoC-1 card with the `et` kernel driver loaded, exposing
  `/dev/etN_ops`, and
- permission to open and `ioctl` that device node.

There is no software fallback in the default build; on a machine without the card
`Device::open` fails.

## Run-time requirements, `emu` feature (no card)

`Device::open_emulator` drives the SDK's software emulator through a C++ shim, so
you additionally need, at build time of the shim:

- a CMake toolchain and a C++ compiler (the shim is compiled during
  `cargo build --features emu`), and
- the SDK's C++ device-layer libraries and firmware ELFs, under `/opt/et` (or the
  path in `ET_SDK_PREFIX`).

## Optional build-time features

- `--features emu` compiles the C++ emulator shim (as above).
- `--features regenerate-bindings` regenerates the vendored `src/bindings_*.rs`
  from the SDK headers and needs `libclang`. Maintainers only; see
  [Regenerating bindings](../developer-guide/regenerating-bindings.md).

## Building the device kernels

Kernels live in the `et-k-rs` crate and cross-compile to the compute harts:

```bash
rustup target add riscv64imac-unknown-none-elf   # once
( cd et-k-rs && cargo build --release )
# -> et-k-rs/target/riscv64imac-unknown-none-elf/release/{hello-rs,spsc-rs,reduce-rs}
```

The target, code model, and linker script are set in `et-k-rs/.cargo/config.toml`,
so a plain `cargo build --release` inside `et-k-rs` is all that is needed.
