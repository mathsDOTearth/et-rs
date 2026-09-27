# Architecture

## Three crates

```
              et-abi            (no_std, no deps: shared wire/ABI types)
             /      \
        et-rs        et-k-rs     (both depend on et-abi; neither on the other)
     (host, std)   (device, no_std)
     lib et_soc1   lib et_kernel
```

- **`et-abi`** holds types that both sides must agree on, chiefly the kernel
  launch-argument structs. It is `no_std` with no dependencies, so it builds for
  the host and for the RISC-V device target alike.
- **`et-rs`** (`et_soc1`) is the host driver: allocation, kernel loading, launch,
  DMA, and the trace decoder, all pure Rust.
- **`et-k-rs`** (`et_kernel`) is the device-side library plus demo kernels; it
  cross-compiles to `riscv64imac-unknown-none-elf`.

`et-rs` and `et-abi` form the host Cargo workspace; `et-k-rs` is excluded from it
because it targets bare-metal RISC-V with its own `.cargo/config.toml`.

## The `Transport` trait

The host driver never calls the kernel driver or the emulator directly. All I/O
goes through the [`Transport`] trait: DRAM info, firmware update, submission-queue
push/pop, DMA host-buffer staging, and trace extraction. [`Device`] is generic
over it, defaulting to [`IoctlTransport`]:

```rust,ignore
pub struct Device<T: Transport = IoctlTransport> { /* ... */ }
```

Two implementations exist:

| Transport | Path | Notes |
|-----------|------|-------|
| [`IoctlTransport`] | real hardware | `ioctl` on `/dev/etN_ops`, magic `0xE7`. |
| [`FfiTransport`] | `emu` feature | C ABI over the SDK C++ device layer (`DeviceSysEmu`), built by an in-tree CMake shim. |

A third, in-memory transport double lives in the test suite
(`et-rs/tests/device_mock.rs`) and drives everything up to the driver boundary
without hardware.

## DMA host buffers

Host memory used for DMA differs by backend: the driver requires a CMA `mmap` of
the ops node, while the emulator requires a specifically allocated buffer whose
physical-address field it dereferences directly. The [`Transport::dma_host_buffer`]
method abstracts this, returning a [`DmaHostBuffer`] with virtual and physical
addresses; `memcpy_h2d`/`memcpy_d2h` stage through it and copy out, so the same
transfer code works on both backends.

## Command model

The device command model is single-threaded: commands are pushed onto a
submission queue and their responses popped from a completion queue, correlated
by a tag. `Device` therefore takes `&self` on its command methods and keeps the
mutating state (the DRAM bump pointer and the tag counter) in `Cell`s. Waits are
sliced against a deadline so a backend whose `wait` returns immediately (the
emulator) is not mistaken for a timeout.

The wire format itself is covered in [Wire protocol](wire-protocol.md).

[`Transport`]: https://docs.rs/et-rs/latest/et_soc1/transport/trait.Transport.html
[`Transport::dma_host_buffer`]: https://docs.rs/et-rs/latest/et_soc1/transport/trait.Transport.html
[`DmaHostBuffer`]: https://docs.rs/et-rs/latest/et_soc1/transport/trait.DmaHostBuffer.html
[`Device`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html
[`IoctlTransport`]: https://docs.rs/et-rs/latest/et_soc1/transport/struct.IoctlTransport.html
[`FfiTransport`]: https://docs.rs/et-rs/latest/et_soc1/transport/struct.FfiTransport.html
