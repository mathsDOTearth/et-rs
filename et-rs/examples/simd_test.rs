//! PS SIMD verification: confirms that `FBCX.PS` and `FMUL.PS` execute
//! correctly on ET-SoC-1 silicon.
//!
//! Each Minion computes `(minion_idx + 1) as f32 * 3.0` using the PS SIMD
//! instructions and writes the f32 result to a cache-line-padded output cell.
//! The host downloads the results and verifies every cell against the expected
//! scalar value computed on the host.
//!
//! On a clean pass across all 1024 Minions this confirms that the instruction
//! encodings in `et_kernel::simd` (`broadcast_ps`, `fmul_ps_row`,
//! `scale_c_row`) are correct on hardware, and `#[doc(hidden)]` may be removed
//! from the `simd` module in `et-k-rs/src/lib.rs`.
//!
//! # Usage
//! ```text
//! cargo run --example simd_test -- <simd-test-rs> [shires]
//! ```
//! `shires` defaults to all present. Build the kernel ELF with:
//! ```text
//! cargo build --release --bin simd-test-rs
//! ```
//! (`.cargo/config.toml` enables `target-feature=+f` required for the F-
//! extension instructions used in the kernel.)

use std::process::ExitCode;

use et_abi::{DeviceArgs, MINIONS_PER_SHIRE, SimdTestArgs};
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
    let n_minions = n_shires * MINIONS_PER_SHIRE as usize;

    let shire_mask = if n_shires >= max_shires {
        topo.shire_mask
    } else {
        (1u64 << n_shires) - 1
    };

    println!(
        "Device: {} shires present (mask {:#x}); running {} shire(s), {} Minions",
        max_shires, topo.shire_mask, n_shires, n_minions,
    );

    let kernel = device.load_kernel(&elf)?;
    let output = device.alloc_padded::<f32>(n_minions)?;
    let args = SimdTestArgs {
        output: output.addr(),
        n_shires: n_shires as u64,
    };

    let opts = LaunchOptions::new(shire_mask).with_args(args.as_bytes().to_vec());
    println!("Launching PS SIMD test ...");
    device.launch(&kernel, &opts)?;
    println!("Kernel returned.");

    let results = device.download_padded(&output)?;

    let mut failures = 0_usize;
    for (i, &val) in results.iter().enumerate() {
        let expected = (i as f32 + 1.0) * 3.0;
        // All values (i+1)*3.0 for i in 0..1023 are exactly representable as
        // f32 (max = 3072 << 2^24), so a bit-exact comparison is valid.
        if val.to_bits() != expected.to_bits() {
            eprintln!("  FAIL output[{i}] = {val}  (expected {expected}, bits {:#010x} vs {:#010x})",
                val.to_bits(), expected.to_bits());
            failures += 1;
        }
    }

    if failures == 0 {
        println!("PS SIMD PASSED: all {} cells correct", n_minions);
        println!("  -> `#[doc(hidden)]` may be removed from `et_kernel::simd`");
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "PS SIMD FAILED: {failures}/{n_minions} cells incorrect"
        )))
    }
}
