# Launching kernels

A launch takes a [`LoadedKernel`] (from [`Device::load_kernel`]) and a
[`LaunchOptions`], and runs the kernel on the selected shires in SPMD fashion:
every hart of every selected shire runs the same kernel.

```rust,ignore
use et_soc1::{LaunchOptions, TraceConfig};
use std::time::Duration;

let opts = LaunchOptions::new(shire_mask)          // which shires run the kernel
    .with_trace(TraceConfig::full(trace_buf, shire_mask))
    .with_args(args_bytes)
    .with_timeout(Duration::from_secs(120));        // override the 10 s default

let result = device.launch(&kernel, &opts)?;
println!("{} cycles", result.timing.execute_dur);
```

[`LaunchOptions::new`] enables a barrier by default. All builder methods are
documented on [`LaunchOptions`]; `#[non_exhaustive]` means struct literals cannot
be used outside the crate -- always call `LaunchOptions::new` and the builders.

## Sizing to the device

Rather than hard-coding a shire mask and hart count, query the device with
[`Device::topology`], which reports the present compute-shire mask alongside the
architectural per-shire geometry:

```rust,ignore
let topo = device.topology()?;
let shire_mask = topo.first_shire();   // launch on the lowest present shire
let n_harts = topo.harts_per_shire;    // 64 on the ET-SoC-1
```

[`Topology`] also provides `num_shires()`, `num_harts()`, and the `cache_line`
size. The reduction demo uses this to size itself to the device instead of
assuming a 64-hart shire 0.

## Passing arguments

Kernel arguments are delivered by **pointer**, not embedded in the launch
command: the host stages an argument struct in device DRAM and the firmware
delivers its address in register `a0` at kernel entry. (This is verified on the
device; the SDK docs claim `ra`, but `ra` is 0 at entry.)

Define the argument struct **once**, in the `et-abi` crate, so the host launcher
and the device kernel cannot disagree on its layout:

```rust,ignore
use et_abi::{DeviceArgs, ReduceArgs};

let args = ReduceArgs { input: input.addr(), out: partials.addr(), n, n_harts };
let opts = LaunchOptions::new(shire_mask).with_args(args.as_bytes().to_vec());
```

The kernel recovers the same struct from the pointer it receives in `a0`:

```rust,ignore
// device side (et-k-rs)
let args = unsafe { ReduceArgs::from_ptr(args_ptr as *const u8) };
```

Because both host and device are little-endian, the `#[repr(C)]` layout is the
wire layout; no serialisation step is involved. See [`et_abi::DeviceArgs`] and
[Writing kernels](../developer-guide/writing-kernels.md).

### One typed call: `launch_spmd`

[`Device::launch_spmd`] bundles the shire mask and the argument staging into a
single call taking the typed struct directly:

```rust,ignore
let r = device.launch_spmd(&kernel, shire_mask, &args)?;
let r = device.launch_spmd_traced(&kernel, shire_mask, &args, trace)?;   // + trace
```

When you need to set a custom timeout or other option, use the `_opts` variants:
they accept a caller-supplied `LaunchOptions` and inject args and trace into it:

```rust,ignore
use std::time::Duration;

let opts = LaunchOptions::new(shire_mask).with_timeout(Duration::from_secs(300));
let r = device.launch_spmd_opts(&kernel, &args, opts)?;
let r = device.launch_spmd_traced_opts(&kernel, &args, trace, opts)?;
```

## Kernel completion timeouts

The host blocks in [`Device::wait_launch`] (or the synchronous [`Device::launch`])
until the firmware sends a completion response. The default wait limit is
transport-dependent:

| Transport | Default |
|---|---|
| `IoctlTransport` (real hardware) | 10 s |
| `FfiTransport` (emulator) | 1 h |

The emulator default is intentionally long: emulated execution is orders of
magnitude slower than hardware, and 512^3 sGEMM passes 256^3 but times out at
10 s.

Override per-launch:

```rust,ignore
LaunchOptions::new(shire_mask).with_timeout(Duration::from_secs(300))
```

Or set a device-wide default for all launches that do not specify one:

```rust,ignore
device.set_default_launch_timeout(Duration::from_secs(300));
```

A timeout surfaces as [`Error::Timeout`], carrying the `operation` name and the
`limit` that elapsed:

```text
timed out after 10.0s waiting for kernel completion; for kernel launches,
set a longer limit with LaunchOptions::with_timeout or Device::set_default_launch_timeout
```

## Async launch and double-buffering

[`Device::launch_async`] submits a kernel and returns immediately with a
[`PendingLaunch`] handle; [`Device::wait_launch`] blocks for the response. The
gap between the two calls can be used for concurrent DMA:

```rust,ignore
use et_soc1::DmaOptions;

// Upload buffer B to SQ 1 while kernel processes buffer A on SQ 0.
let pending = device.launch_async(&kernel, &opts)?;
device.memcpy_h2d_opts(&buf_b_host, buf_b_dev.addr, &DmaOptions::new().on_sq(1))?;
let result = device.wait_launch(pending)?;
```

The `double_buffer` example demonstrates this pattern with hardware-verified
overlap. Responses for other in-flight commands that arrive during `wait_launch`
are stashed and returned by their own collection call; there is no loss of
completions from concurrent launches.

## DMA options

[`DmaOptions`] routes DMA commands to a specific submission queue and
optionally sets a completion timeout. Like `LaunchOptions`, it is
`#[non_exhaustive]` -- use `DmaOptions::new()` and the builders:

```rust,ignore
use et_soc1::DmaOptions;
use std::time::Duration;

let opts = DmaOptions::new()
    .on_sq(1)                                    // route to SQ 1 for concurrency
    .with_timeout(Duration::from_secs(60));       // impose an explicit bound
```

Pass to [`Device::memcpy_h2d_opts`] or [`Device::memcpy_d2h_opts`]. By default
DMA waits are unlimited: the call returns as soon as the firmware acknowledges
the transfer. Use `with_timeout` only to impose a defensive upper bound; it does
not affect kernel completion timeouts. [`Device::set_default_launch_timeout`]
governs kernel completion only and does not apply to DMA.

## Reading a launch failure

A failed launch returns [`Error::KernelLaunch`], which decodes the raw device
status into a symbolic name and, when the firmware appended diagnostics, the
faulting shire mask and the device addresses of the U-mode exception and trace
buffers:

```text
kernel-launch failed: EXCEPTION (status 2); faulting shires 0x1; \
    exception buffer @ 0x8006000000; trace buffer @ 0x8006100000
```

On an exception the firmware still fills the trace buffer, so it is worth
decoding the trace even on failure, as the examples do. To have the firmware
populate an exception buffer with per-hart records, set
[`LaunchOptions::exception_buffer`] to a device region you allocated.

[`Device::topology`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.topology
[`Device::launch`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.launch
[`Device::launch_async`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.launch_async
[`Device::wait_launch`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.wait_launch
[`Device::launch_spmd`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.launch_spmd
[`Device::memcpy_h2d_opts`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.memcpy_h2d_opts
[`Device::memcpy_d2h_opts`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.memcpy_d2h_opts
[`Topology`]: https://docs.rs/et-rs/latest/et_soc1/topology/struct.Topology.html
[`LoadedKernel`]: https://docs.rs/et-rs/latest/et_soc1/struct.LoadedKernel.html
[`PendingLaunch`]: https://docs.rs/et-rs/latest/et_soc1/struct.PendingLaunch.html
[`Device::load_kernel`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.load_kernel
[`LaunchOptions`]: https://docs.rs/et-rs/latest/et_soc1/struct.LaunchOptions.html
[`LaunchOptions::new`]: https://docs.rs/et-rs/latest/et_soc1/struct.LaunchOptions.html#method.new
[`LaunchOptions::exception_buffer`]: https://docs.rs/et-rs/latest/et_soc1/struct.LaunchOptions.html
[`DmaOptions`]: https://docs.rs/et-rs/latest/et_soc1/struct.DmaOptions.html
[`Error::KernelLaunch`]: https://docs.rs/et-rs/latest/et_soc1/enum.Error.html
[`Error::Timeout`]: https://docs.rs/et-rs/latest/et_soc1/enum.Error.html#variant.Timeout
[`et_abi::DeviceArgs`]: https://docs.rs/et-abi/latest/et_abi/trait.DeviceArgs.html
