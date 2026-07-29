# Introduction

`et-rs` is pure-Rust tooling for the Esperanto **ET-SoC-1** RISC-V accelerator: a
host driver, a device-side kernel library, and the ABI they share. It loads
compute kernels, launches them across the chip's compute shires, moves results
back over DMA, and decodes on-device trace buffers, all without wrapping the
vendor C++ runtime.

The repository is a Cargo workspace of three crates:

| Crate | Library | Side | Role |
|-------|---------|------|------|
| `et-abi` | `et_abi` | shared (`no_std`) | Launch-argument structs defined once, so host and device cannot drift on layout. |
| `et-rs` | `et_soc1` | host (`std`) | Kernel load, launch, DMA, and the trace decoder. |
| `et-k-rs` | `et_kernel` | device (`no_std`) | Library for writing compute kernels in Rust, plus demo kernels. |

`et-rs` and `et-abi` are host crates; `et-k-rs` cross-compiles to
`riscv64imac-unknown-none-elf`. Both `et-rs` and `et-k-rs` depend on `et-abi`;
neither depends on the other.

## What this book covers, and what it does not

This book is the **narrative** guide: how to install the crates, write a first
program, manage device memory, launch kernels, and write kernels of your own,
followed by a developer's tour of the architecture, the wire protocol, and the
chip's coherence model.

It is **not** the API reference. Every public item is documented with `///`
comments and published to [docs.rs](https://docs.rs); this book links into those
pages rather than restating them. When you see a type or method mentioned here,
follow the link for its full signature and contract:

- [`et_soc1` on docs.rs](https://docs.rs/et-rs) (the host driver)
- [`et_kernel` on docs.rs](https://docs.rs/et-k-rs) (the device library)
- [`et_abi` on docs.rs](https://docs.rs/et-abi) (the shared ABI)

## Hardware or emulator

The default build drives a real card through the PCIe kernel driver. With the
`emu` feature the same code drives the SDK software emulator instead, so you can
develop and test without hardware. See [Installation](user-guide/installation.md).
