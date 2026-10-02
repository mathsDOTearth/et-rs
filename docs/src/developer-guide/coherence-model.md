# Coherence model

The ET-SoC-1 is **software-coherent**, not hardware cache-coherent. This is the
single most important fact for anyone writing multi-hart kernels, and it was
established empirically with the `spsc-rs` probe.

## What the probe showed

`spsc-rs` is a single-producer, single-consumer, lock-free, non-atomic ring
buffer across two harts, using plain volatile loads and stores plus a
`fence rw, rw`. On a hardware cache-coherent machine this is a valid (if
delicate) construction. On the ET-SoC-1 it fails: the producer fills the ring and
the consumer sees nothing, both on the emulator (whose consistency checkers abort
the run) and on real hardware. Fence-only cross-hart sharing does not propagate,
even between the two harts of a single minion, and the per-hart scratchpad view at
`0x80000000` is hart-local rather than shared.

The conclusion: cross-hart communication needs either explicit cache management
(the SDK's cache-ops, fast comm channels, fast local barriers) or genuinely
shared, uncached memory. A `fence` alone is not enough.

## Why the reduction is clean

The data-parallel reduction (`reduce-rs`) is designed to avoid the problem
entirely. During the compute there is **no** cross-hart sharing:

- each hart reads a disjoint slice of the input, and
- each hart writes only its own output cell.

The host does the only cross-hart step, combining the partials after the kernel
returns, by which point the device has flushed. This is why the reduction passes
on the emulator with the consistency checkers **on**, whereas the SPSC probe does
not.

## False sharing is silent corruption

Even disjoint writes are unsafe if two harts write into the same cache line.
Under software coherence, each hart's cached copy of the line is written back
independently, so two harts sharing a line clobber each other's results with no
error. This is why per-hart outputs must be padded to a full cache line.

The framework encodes this rule in types: the host-side
[`PaddedArray<T>`](../user-guide/device-memory.md) lays out one element per cache
line, and the device-side `Grid::output_cell` writes hart *h* at
`base + h * CACHE_LINE`. Follow both and false sharing cannot occur; ignore
either and results corrupt silently.

## Practical guidance

- Prefer disjoint partitioning (the `Grid` pattern) over shared mutable state.
- Pad any per-hart output to a cache line.
- Do genuine cross-hart reduction/exchange on the host after the kernel returns,
  or, when it must happen on-device, with explicit cache operations and barriers
  rather than fences alone.

## On-device message passing

The `channel-bench-rs` kernel and `channel_bench` example demonstrate, and
time, the explicit-cache-operation protocol between two Minions. The producer
writes the payload, writes it back with `cache_writeback_to(level, ...)`, then
writes and writes back a flag; the consumer invalidates the flag line with
`cache_invalidate_to(level, ...)` before every poll, and invalidates the
payload before reading it. The levels must be shared by both Minions: L3 or
DDR in general, and L2 only within one shire. On aifoundry3 the L3 and DDR
levels are correct for every placement, whereas L2 within a shire stalls
after a few messages; the mixed L2/L3 and L3/L2 runs of the example identify
which of the two L2 operations is ineffective. Until that is resolved, L3 is
the recommended level. Each flag and payload occupies cache lines that only one
Minion writes: an invalidation writes back any dirty line it removes, so a
consumer holding dirty data in the producer's lines would overwrite the
message.
