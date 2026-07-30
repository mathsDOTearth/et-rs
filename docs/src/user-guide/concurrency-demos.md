# Concurrency demos

Two `et-k-rs` kernels explore Rust concurrency on the ET-SoC-1. Both are launched
from host examples in `et-rs`.

## Data-parallel reduction (`reduce-rs`)

The reduction sums a large array across a full shire of harts (sized from the
device via [`Device::topology`], not hard-coded). Each hart reduces its
**disjoint** slice of the input and writes its partial into its **own
cache-line-padded** cell; the host then combines the partials. There is no
cross-hart sharing during the compute, so it is coherence-clean: it passes the
emulator's consistency checkers and runs correctly on hardware.

The data-safety story is the point of the demo. The kernel body is safe Rust over
the device-side `Grid` abstraction, which hands each hart only its own slice
(`my_slice`) and its own output cell (`output_cell`). A kernel author cannot name
another hart's data, so a cross-hart data race is unrepresentable; the only
`unsafe` is the thin, commented boundary that turns the launch arguments and
device addresses into typed slices.

Run it:

```bash
R=et-k-rs/target/riscv64imac-unknown-none-elf/release/reduce-rs
cargo run --features emu --example reduce -- "$R"   # emulator
cargo run                --example reduce -- "$R"   # real hardware -> RESULT PASS
```

The complete host launcher, embedded from the real example file so it always
matches what actually compiles and runs (`cargo build --examples`):

```rust,ignore
{{#include ../../../et-rs/examples/reduce.rs}}
```

Note how the host side uses only typed buffers ([`Device::upload`],
[`Device::alloc_padded`], [`Device::download_padded`]) and the shared
`ReduceArgs` struct: no raw addresses or byte casts appear at the call site.

## Lock-free SPSC (`spsc-rs`), a coherence probe

`spsc-rs` implements a single-producer, single-consumer, lock-free, **non-atomic**
queue across two harts using plain volatile loads and stores plus a `fence`, no
atomics and no locks. It was written as a probe, and it established an important
fact: the ET-SoC-1 is **software-coherent**. Fence-only cross-hart sharing does
not propagate, even between two harts of one minion, so a design like this needs
explicit cache management or genuinely shared memory. See the
[Coherence model](../developer-guide/coherence-model.md) for what this means and
why the reduction avoids the problem entirely.

Because it is a probe, not a correctness test, the launcher reports the outcome
and exits successfully either way: on hardware it runs to completion and reports
that the queue did not propagate, and on the software emulator the consistency
checkers abort the illegal sharing outright. Both are the same finding.

[`Device::topology`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.topology
[`Device::upload`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.upload
[`Device::alloc_padded`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.alloc_padded
[`Device::download_padded`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.download_padded
