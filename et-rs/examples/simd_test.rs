//! PS SIMD verification: confirms that `FBCX.PS` and `FMUL.PS`, issued through
//! the `et_kernel::simd` library functions, execute correctly on ET-SoC-1
//! silicon.
//!
//! Each Minion computes `(minion_idx + 1) as f32 * 3.0` in all 16 lanes of one
//! C-tile row (FP registers f0 and f1) and stores the row to a cache-line
//! output cell. The host prefills the output with a NaN sentinel, then checks
//! all 16 values of every cell, so a broadcast or multiply that reached only
//! some lanes, or a cell that was never written, fails.
//!
//! # Usage
//! ```text
//! cargo run --example simd_test -- <simd-test-rs> [shires]
//! ```
//! `shires` limits the run to the lowest `shires` present shires; it defaults
//! to all present. Build the kernel ELF with:
//! ```text
//! cargo build --release --bin simd-test-rs
//! ```

use std::process::ExitCode;

use et_abi::{CACHE_LINE, DeviceArgs, MINIONS_PER_SHIRE, SimdTestArgs};
use et_soc1::{Device, LaunchOptions};

/// f32 lanes per output cell: one C-tile row (two 8-lane PS registers).
const LANES: usize = CACHE_LINE / size_of::<f32>();
/// Byte written over the output buffer before launch (all-ones is a NaN
/// pattern for f32, never an expected result).
const SENTINEL: u8 = 0xFF;

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
    let argv: Vec<String> = std::env::args().collect();
    let kernel_path = argv.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: simd_test <simd-test-rs> [shires]");
        std::process::exit(2);
    });
    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::io("read kernel ELF", e))?;

    let device = Device::open(0)?;
    let topo = device.topology()?;
    let max_shires = topo.num_shires() as usize;

    let n_shires = argv
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .map(|k| k.clamp(1, max_shires))
        .unwrap_or(max_shires);

    // The lowest `n_shires` present shires. Taking them from the topology mask
    // (rather than assuming shires 0..n_shires) never launches an absent shire.
    let mut shire_mask = 0u64;
    for shire in (0..u64::BITS)
        .filter(|&s| topo.shire_mask & (1 << s) != 0)
        .take(n_shires)
    {
        shire_mask |= 1 << shire;
    }
    // Cells are indexed by physical shire: the array spans every shire up to
    // the highest launched one.
    let shire_extent = (u64::BITS - shire_mask.leading_zeros()) as usize;
    let n_cells = shire_extent * MINIONS_PER_SHIRE as usize;
    let n_minions = n_shires * MINIONS_PER_SHIRE as usize;

    println!(
        "Device: {} shires present (mask {:#x}); running {} shire(s) (mask {:#x}), {} Minions",
        max_shires, topo.shire_mask, n_shires, shire_mask, n_minions,
    );

    let kernel = device.load_kernel(&elf)?;
    // One full cache line (LANES f32) per cell.
    let output = device.alloc_array::<f32>(n_cells * LANES)?;
    device.fill(output.region(), SENTINEL)?;
    let args = SimdTestArgs {
        output: output.addr(),
        n_shires: shire_extent as u64,
    };

    let opts = LaunchOptions::new(shire_mask).with_args(args.as_bytes().to_vec());
    println!("Launching PS SIMD test ...");
    device.launch(&kernel, &opts)?;
    println!("Kernel returned.");

    let results = device.download(&output)?;

    let mut failures = 0_usize;
    for (i, cell) in results.as_chunks::<LANES>().0.iter().enumerate() {
        let launched = shire_mask & (1 << (i / MINIONS_PER_SHIRE as usize)) != 0;
        if !launched {
            if cell
                .iter()
                .any(|v| v.to_bits() != u32::from_ne_bytes([SENTINEL; 4]))
            {
                eprintln!("  FAIL output[{i}]: cell of a shire not launched was written");
                failures += 1;
            }
            continue;
        }
        // All values (i+1)*3.0 for i < 1024 are exactly representable as f32
        // (max 3072, far below 2^24), so a bit-exact comparison is valid.
        let expected = (i as f32 + 1.0) * 3.0;
        let wrong: Vec<usize> = (0..LANES)
            .filter(|&lane| cell[lane].to_bits() != expected.to_bits())
            .collect();
        if !wrong.is_empty() {
            eprintln!(
                "  FAIL output[{i}]: lanes {wrong:?} wrong (lane {} = {}, expected {expected})",
                wrong[0], cell[wrong[0]]
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!("PS SIMD PASSED: all {n_minions} cells correct in all {LANES} lanes");
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "PS SIMD FAILED: {failures}/{n_minions} cells incorrect"
        )))
    }
}
