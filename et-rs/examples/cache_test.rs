//! Cache-coherence test: verifies that `cache_writeback` makes Minion-written
//! data visible to host DMA.
//!
//! Each primary Minion hart writes its global Minion index into its own
//! cache-line-padded output cell and calls `cache_writeback` before `fence`.
//! The host downloads the output array and asserts every cell holds the
//! expected Minion index.
//!
//! # What a failure means
//!
//! If the writeback is missing or broken, the host DMA reads stale DDR data
//! (the write is still in the Minion's private L1 cache). The stale value is
//! either zero (if DDR was freshly allocated) or whatever was in memory
//! before launch -- in either case it differs from the Minion index and the
//! test reports which cells are wrong.
//!
//! # Usage
//! ```text
//! cargo run --example cache_test -- <path-to-cache-test-rs.elf>
//! ```

use std::process::ExitCode;

use et_abi::{CacheTestArgs, MINIONS_PER_SHIRE};
use et_soc1::Device;

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
        eprintln!("usage: cache_test <cache-test-rs.elf>");
        std::process::exit(2);
    });

    let elf = std::fs::read(&kernel_path)
        .map_err(|e| et_soc1::Error::io("read kernel ELF", e))?;

    let device = Device::open(0)?;
    let topo   = device.topology()?;

    let n_shires  = topo.num_shires() as usize;
    let n_minions = n_shires * MINIONS_PER_SHIRE as usize;

    println!(
        "Device: {} shires (mask {:#x}), {} Minions total",
        n_shires, topo.shire_mask, n_minions
    );

    // Allocate output: one u32 per Minion, each on its own cache line.
    // PaddedArray gives each element a 64-byte cell; the kernel writes at
    // `output_addr + minion_idx * CACHE_LINE`.
    let output = device.alloc_padded::<u32>(n_minions)?;

    let args = CacheTestArgs {
        output:   output.addr(),
        n_shires: n_shires as u64,
    };

    let kernel = device.load_kernel(&elf)?;

    println!("Launching cache_writeback test on {} shires...", n_shires);
    device.launch_spmd(&kernel, topo.shire_mask, &args)?;
    println!("Kernel returned.");

    // Download the per-Minion outputs (strips the 64-byte padding, returns
    // one u32 per Minion).
    let results = device.download_padded(&output)?;

    let mut failures = 0_usize;
    for (i, &val) in results.iter().enumerate() {
        if val != i as u32 {
            eprintln!("  FAIL output[{i}] = {val:#010x}  (expected {i:#010x})");
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "cache_writeback PASSED: all {} cells correct",
            n_minions
        );
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "cache_writeback FAILED: {failures}/{n_minions} cells incorrect"
        )))
    }
}
