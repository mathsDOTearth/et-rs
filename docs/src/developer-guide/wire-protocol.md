# Wire protocol

The host and firmware exchange fixed-layout messages over submission and
completion queues (the device-ops RPC). The command builders and response parsers
live in `et_soc1::proto`, hand-written as `#[repr(C)]` structs with compile-time
size assertions rather than generated, because `bindgen` miscompiles the SDK's
`packed`+`aligned` message structs (see [Regenerating bindings](regenerating-bindings.md)).

## Common header

Every command and response begins with an 8-byte common header (`cmn_header_t`).
The single most important detail, and a frequent source of firmware aborts:

> `cmn_header.size` is the **total** command size, header included, not the
> payload size.

Getting this wrong causes the emulator to abort on the first `push_sq`. The
builders in `proto` set it correctly for you.

## DMA read and write lists

Bulk transfers use DMA descriptor lists (`dma_readlist_cmd_t` /
`dma_writelist_cmd_t`), each node describing a host/device address pair and a
size. The host driver splits a transfer to honour the device's advertised limits
(`dma_max_elem_size`, `dma_max_elem_count` in [`DramInfo`]).

Two gotchas:

- A node's **physical address must be valid**: the emulator dereferences it
  directly, so host memory must come from a proper DMA host buffer (see
  [Architecture](architecture.md)), not an arbitrary pointer.
- `DramInfo.align_in_bits` from the SDK is really a **byte quantum** (64 = one
  cache line), not a shift; treating it as `1 << 64` exhausts DRAM instantly. The
  crate normalises it to a power-of-two byte alignment.

## Kernel launch

`kernel_launch_cmd_t` carries the entry address, `pointer_to_args`, an exception
buffer address, the shire mask, and an optional payload (trace and stack config).
Kernels are **not** loaded by `FW_UPDATE` (that is for firmware and rejects kernel
ELFs); `Device::load_kernel` DMA-writes each `PT_LOAD` segment to its link address
instead. See [Writing kernels](writing-kernels.md).

Arguments are delivered by pointer and arrive in register `a0` at kernel entry,
despite the SDK docs naming `ra` (verified on device: `ra` is 0 at entry, and an
embedded payload populates neither register).

## Responses

Launch and DMA responses share a prefix: an 8-byte response header, three 8-byte
timing counters, then a 32-bit status at byte offset 32
(`proto::RSP_STATUS_OFFSET`). A failed kernel launch may append a
`kernel_rsp_error_ptr_t` (at offset 40) carrying the U-mode exception-buffer
pointer, the trace-buffer pointer, and the faulting shire mask;
`proto::parse_kernel_error_ptr` decodes it and `Error::KernelLaunch` surfaces it
(see [Launching kernels](../user-guide/launching-kernels.md)).

## ioctl request codes

Request codes are computed by a `const fn` mirroring the kernel's `_IOC` macro,
with a golden test (`ioctl::tests::request_codes_match_c_header`) asserting they
match the values in the SDK header. This keeps the hand-written codes honest
without depending on `bindgen` for them.

[`DramInfo`]: https://docs.rs/et-rs/latest/et_soc1/transport/struct.DramInfo.html
