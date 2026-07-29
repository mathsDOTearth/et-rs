# Device memory

Device DRAM is managed by a bump allocator on the [`Device`] handle. There are
two layers:

- the low-level, byte-oriented primitives ([`Device::alloc`],
  [`Device::memcpy_h2d`], [`Device::memcpy_d2h`]), and
- the typed buffers ([`DeviceBuffer<T>`], [`PaddedArray<T>`]) layered on top,
  which carry element type and count so call sites need no byte arithmetic or
  `unsafe` slice casts.

Prefer the typed layer for data; reach for the raw primitives for opaque byte
regions such as trace buffers.

## Typed buffers

Upload a host slice and get back a typed handle; download into a typed `Vec`:

```rust,ignore
let host_in: Vec<u32> = (0..n).collect();

let input = device.upload(&host_in)?;          // DeviceBuffer<u32>
let output = device.alloc_array::<u32>(n)?;     // uninitialised DeviceBuffer<u32>

// ... launch a kernel that reads `input.addr()` and writes `output.addr()` ...

let result: Vec<u32> = device.download(&output)?;
```

[`DeviceBuffer::addr`] gives the device address to hand to a kernel as an
argument (see [Launching kernels](launching-kernels.md)). The element type is
bounded by [`DevicePod`], an unsafe marker implemented for the integer and float
scalars; implement it for your own `#[repr(C)]` plain-old-data structs to store
them.

## Cache-line-padded arrays

When distinct harts each write their own result, put each result on its own cache
line. The ET-SoC-1 is software-coherent, so two harts writing different values
into the *same* cache line corrupt each other silently (false sharing). See the
[Coherence model](../developer-guide/coherence-model.md).

[`PaddedArray<T>`] encodes this: it lays out one element per cache line, and
pairs with the device-side `Grid::output_cell`, which writes hart *h*'s result at
`base + h * CACHE_LINE`.

```rust,ignore
let partials = device.alloc_padded::<u64>(n_harts)?;   // one u64 per cache line

// ... launch; each hart writes its own cell ...

let parts: Vec<u64> = device.download_padded(&partials)?;  // one u64 per line
let total: u64 = parts.iter().sum();
```

## Raw regions

For buffers the device fills in an opaque format (for example a trace buffer),
use the untyped primitives directly:

```rust,ignore
let trace_buf = device.alloc(8 * 1024 * 1024)?;   // DeviceRegion { addr, size }
// ... launch with tracing into trace_buf ...
let mut host = vec![0u8; trace_buf.size as usize];
device.memcpy_d2h(trace_buf.addr, &mut host)?;
```

## Notes on the allocator

- Allocation is a forward-only bump within the device's user DRAM region;
  there is no device-side free, so buffers live until the handle is dropped.
- Sizes are rounded up to the device's DMA alignment (see [`DramInfo`]).
- Load kernels before allocating other regions, so the kernel lands at the DRAM
  base that matches its link address (see [Writing kernels](../developer-guide/writing-kernels.md)).

[`Device`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html
[`Device::alloc`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.alloc
[`Device::memcpy_h2d`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.memcpy_h2d
[`Device::memcpy_d2h`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.memcpy_d2h
[`DeviceBuffer<T>`]: https://docs.rs/et-rs/latest/et_soc1/buffer/struct.DeviceBuffer.html
[`DeviceBuffer::addr`]: https://docs.rs/et-rs/latest/et_soc1/buffer/struct.DeviceBuffer.html#method.addr
[`PaddedArray<T>`]: https://docs.rs/et-rs/latest/et_soc1/buffer/struct.PaddedArray.html
[`DevicePod`]: https://docs.rs/et-rs/latest/et_soc1/buffer/trait.DevicePod.html
[`DramInfo`]: https://docs.rs/et-rs/latest/et_soc1/transport/struct.DramInfo.html
