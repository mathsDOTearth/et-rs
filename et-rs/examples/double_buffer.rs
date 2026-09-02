//! Double-buffered launch demonstration.
//!
//! Exercises three features added in v0.5.0:
//!
//! 1. **`Device::launch_async` + `Device::wait_launch`**: fire a kernel onto
//!    the device without blocking, then collect the result later.
//! 2. **Response stash**: if the kernel's completion arrives on the CQ while
//!    the host is waiting for a different command (a concurrent DMA, for
//!    example), it is parked automatically and returned by `wait_launch`.
//! 3. **`DmaOptions::on_sq`**: route DMA commands to a separate submission
//!    queue so the firmware can process them concurrently with a running kernel.
//!
//! # Pattern
//!
//! ```text
//! Phase A: launch_async + wait_launch
//!
//!   launch_async(all shires) ──────────────────────────────────┐
//!   wait_launch()                                              │ kernel running
//!   verify output_a                                            │
//!                                                              ┘
//!
//! Phase B: response stash + concurrent DMA
//!
//!   launch_async(all shires) ──────────────────────────────────┐
//!     while kernel (possibly) runs on SQ 0:                    │ concurrent
//!     memcpy_h2d_opts(staging, SQ 1)  ─────────────────────────┤ DMA on SQ 1
//!   wait_launch()  ─────────── response may come from stash    │
//!   verify output_b                                            ┘
//! ```
//!
//! In a real inference loop the "staging DMA" would upload the next batch's
//! input matrix while the current batch's kernel runs. Here it uploads a
//! 64 KB block to demonstrate the SQ routing and stash paths on hardware.
//!
//! # Usage
//!
//! ```text
//! cargo run --example double_buffer -- <path-to-cache-test-rs.elf>
//! ```

use std::process::ExitCode;

use et_abi::{CacheTestArgs, DeviceArgs, MINIONS_PER_SHIRE};
use et_soc1::{Device, DmaOptions, LaunchOptions};

// Size of the staging DMA used in Phase B to create a meaningful overlap
// window (large enough that the kernel's response may arrive first).
const STAGING_BYTES: usize = 64 * 1024;

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
        eprintln!("usage: double_buffer <cache-test-rs.elf>");
        std::process::exit(2);
    });

    let elf = std::fs::read(&kernel_path)
        .map_err(|e| et_soc1::Error::io("read kernel ELF", e))?;

    let device    = Device::open(0)?;
    let topo      = device.topology()?;
    let n_shires  = topo.num_shires() as usize;
    let n_minions = n_shires * MINIONS_PER_SHIRE as usize;

    println!(
        "Device: {} shires (mask {:#x}), {} Minions",
        n_shires, topo.shire_mask, n_minions,
    );

    let kernel = device.load_kernel(&elf)?;

    // Two separate output arrays: each Minion writes its global index to its
    // own 64-byte-padded cell.
    let out_a = device.alloc_padded::<u32>(n_minions)?;
    let out_b = device.alloc_padded::<u32>(n_minions)?;

    // Device region used as the staging target for the Phase B DMA.
    // Content does not matter; we just need a valid writable region.
    let staging = device.alloc(STAGING_BYTES as u64)?;
    let staging_data = vec![0u8; STAGING_BYTES];

    // -----------------------------------------------------------------------
    // Phase A: basic launch_async + wait_launch
    // -----------------------------------------------------------------------
    println!("\nPhase A: launch_async + wait_launch");

    let args_a = CacheTestArgs { output: out_a.addr(), n_shires: n_shires as u64 };
    let opts_a = LaunchOptions::new(topo.shire_mask)
        .without_barrier()           // first launch: no prior command to wait for
        .with_args(args_a.as_bytes().to_vec());

    let pending_a = device.launch_async(&kernel, &opts_a)?;
    println!("  kernel A fired (async, SQ 0)");

    let _r = device.wait_launch(pending_a)?;
    println!("  kernel A complete");

    let results_a = device.download_padded(&out_a)?;
    check("A", &results_a, n_minions)?;

    // -----------------------------------------------------------------------
    // Phase B: response stash + concurrent DMA on SQ 1
    // -----------------------------------------------------------------------
    println!("\nPhase B: kernel (SQ 0) overlapped with DMA (SQ 1)");

    let args_b = CacheTestArgs { output: out_b.addr(), n_shires: n_shires as u64 };
    let opts_b = LaunchOptions::new(topo.shire_mask)
        // barrier=true (default): wait for prior SQ 0 commands (kernel A is
        // already done, so this barrier is immediate).
        .with_args(args_b.as_bytes().to_vec());

    // Fire kernel B asynchronously.
    let pending_b = device.launch_async(&kernel, &opts_b)?;
    println!("  kernel B fired (async, SQ 0)");

    // While kernel B may be running on SQ 0, DMA 64 KB on SQ 1.
    // The firmware can schedule both streams concurrently. If kernel B's CQ
    // response arrives during this DMA wait, `collect_response` stashes it;
    // `wait_launch` below retrieves it from the stash without re-polling.
    device.memcpy_h2d_opts(&staging_data, staging.addr, &DmaOptions::new().on_sq(1))?;
    println!("  {} KB staging DMA complete (SQ 1)", STAGING_BYTES / 1024);

    // Collect kernel B's result, possibly from the stash.
    let _r = device.wait_launch(pending_b)?;
    println!("  kernel B complete");

    let results_b = device.download_padded(&out_b)?;
    check("B", &results_b, n_minions)?;

    println!("\ndouble_buffer PASSED");
    Ok(())
}

/// Assert that `results[i] == i` for all `i` in `0..n_minions`.
fn check(label: &str, results: &[u32], n_minions: usize) -> et_soc1::Result<()> {
    let mut failures = 0_usize;
    for (i, &val) in results.iter().enumerate() {
        if val != i as u32 {
            eprintln!("  FAIL out_{label}[{i}] = {val}  (expected {i})");
            failures += 1;
        }
    }
    if failures == 0 {
        println!("  output_{label}: all {n_minions} cells correct");
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "output_{label}: {failures}/{n_minions} cells incorrect"
        )))
    }
}
