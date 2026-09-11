//! Hardware test for [`Device::reset_shires`] and [`Device::reset_device`].
//!
//! Runs three steps in order:
//!
//! 1. **CM reset**: calls `reset_shires` for all present shires and verifies the
//!    firmware returns a success response. The device node stays open.
//! 2. **Post-CM-reset launch**: launches the supplied kernel to confirm the device
//!    is still usable after a compute-minion reset.
//! 3. **Full ETSOC reset**: calls `reset_device`, which closes the device node and
//!    polls until the firmware brings it back up.  Then launches the kernel once
//!    more to confirm the freshly re-opened device is healthy.
//!
//! Usage:
//! ```text
//! cargo run --release --example reset_test -- \
//!     et-k-rs/target/riscv64imac-unknown-none-elf/release/reduce-rs
//! ```
//!
//! The `reduce-rs` kernel is a convenient, self-verifying payload: it sums a
//! known array and the host confirms the total.  Any kernel that prints a
//! RESULT PASS line will do.

use std::process::ExitCode;

use et_abi::{DeviceArgs, ReduceArgs};
use et_soc1::{Device, LaunchOptions};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> et_soc1::Result<()> {
    let kernel_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: reset_test <path-to-reduce-rs>");
        std::process::exit(1);
    });
    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::io("read kernel ELF", e))?;

    // --- Step 1: open and query topology ---------------------------------
    let device = Device::open(0)?;
    let topo = device.topology()?;
    // Use all shires so that n_harts equals the number of output slots the
    // kernel writes. Launching on first_shire() with n_harts = num_harts()
    // (total across all shires) would leave most slots uninitialized, giving
    // a corrupted reduction sum.
    let shire_mask = topo.shire_mask;
    let n_harts = topo.num_harts() as u32;
    println!(
        "device: {} shire(s) present (mask {:#x}), {} harts/shire",
        topo.num_shires(),
        topo.shire_mask,
        topo.harts_per_shire
    );

    // --- Step 2: CM reset (does not close the device node) ---------------
    println!("\nStep 1: reset_shires({:#x}) ...", topo.shire_mask);
    device.reset_shires(topo.shire_mask)?;
    println!("  CM reset acknowledged by firmware -- OK");

    // --- Step 3: launch to confirm the device is still usable ------------
    println!("\nStep 2: kernel launch after CM reset ...");
    let kernel = device.load_kernel(&elf)?;
    launch_reduce(&device, &kernel, shire_mask, n_harts)?;
    println!("  post-CM-reset launch -- PASS");

    // --- Step 4: full ETSOC reset (consumes device, closes fd) -----------
    println!("\nStep 3: reset_device() (full ETSOC reset) ...");
    let device = device.reset_device()?;
    println!("  device re-opened after ETSOC reset -- OK");

    // --- Step 5: launch on the fresh device to confirm health ------------
    println!("\nStep 4: kernel launch on re-opened device ...");
    let topo2 = device.topology()?;
    let shire_mask2 = topo2.shire_mask;
    let n_harts2 = topo2.num_harts() as u32;
    let kernel2 = device.load_kernel(&elf)?;
    launch_reduce(&device, &kernel2, shire_mask2, n_harts2)?;
    println!("  post-ETSOC-reset launch -- PASS");

    println!("\nreset_test PASSED");
    Ok(())
}

/// Launch one reduction pass on `shire_mask` / `n_harts` and verify the result.
fn launch_reduce(
    device: &Device<et_soc1::transport::IoctlTransport>,
    kernel: &et_soc1::LoadedKernel,
    shire_mask: u64,
    n_harts: u32,
) -> et_soc1::Result<()> {
    const N: u32 = 4096;
    // The reduce-rs kernel reads the input as &[u32], so upload u32 values.
    // The expected total is sum(0..N) regardless of whether i64 or u32 is used,
    // but the partial values written by the kernel (out[h] = sum of the h-th
    // slice of u32 elements) must be consistent with what the kernel computes.
    let input: Vec<u32> = (0..N).collect();
    let expected: i64 = (0..N as i64).sum();

    let buf = device.upload(&input)?;
    let partials = device.alloc_padded::<i64>(n_harts as usize)?;

    let args = ReduceArgs {
        input: buf.addr(),
        out: partials.addr(),
        n: N,
        n_harts,
    };
    let opts = LaunchOptions::new(shire_mask).with_args(args.as_bytes().to_vec());
    device.launch(kernel, &opts)?;

    let result: Vec<i64> = device.download_padded(&partials)?;
    let total: i64 = result.iter().copied().take(n_harts as usize).sum();
    if total != expected {
        return Err(et_soc1::Error::Protocol(format!(
            "reduction mismatch: got {total}, expected {expected}"
        )));
    }
    println!("  reduction over {N} elements: total = {total} (correct)");
    Ok(())
}
