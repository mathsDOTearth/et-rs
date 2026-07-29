# Launching kernels

A launch takes a [`LoadedKernel`] (from [`Device::load_kernel`]) and a
[`LaunchOptions`], and runs the kernel on the selected shires in SPMD fashion:
every hart of every selected shire runs the same kernel.

```rust,ignore
use et_soc1::{LaunchOptions, TraceConfig};

let opts = LaunchOptions::new(shire_mask)          // which shires run the kernel
    .with_trace(TraceConfig::full(trace_buf, shire_mask))
    .with_args(args_bytes);                          // optional, see below

let result = device.launch(&kernel, &opts)?;
println!("{} cycles", result.timing.execute_dur);
```

[`LaunchOptions::new`] enables a barrier by default; other fields (an L3 flush, a
U-mode stack configuration, an exception buffer, the submission-queue index) are
documented on [`LaunchOptions`].

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

[`LoadedKernel`]: https://docs.rs/et-rs/latest/et_soc1/struct.LoadedKernel.html
[`Device::load_kernel`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.load_kernel
[`LaunchOptions`]: https://docs.rs/et-rs/latest/et_soc1/struct.LaunchOptions.html
[`LaunchOptions::new`]: https://docs.rs/et-rs/latest/et_soc1/struct.LaunchOptions.html#method.new
[`LaunchOptions::exception_buffer`]: https://docs.rs/et-rs/latest/et_soc1/struct.LaunchOptions.html
[`Error::KernelLaunch`]: https://docs.rs/et-rs/latest/et_soc1/enum.Error.html
[`et_abi::DeviceArgs`]: https://docs.rs/et-abi/latest/et_abi/trait.DeviceArgs.html
