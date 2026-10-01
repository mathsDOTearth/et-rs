//! Host-side performance baseline: launch overhead, DMA staging cost and DMA
//! throughput.
//!
//! Measures the host paths that the 0.7.0 optimisations target, so that each
//! change can be compared against the same figures on the same card:
//!
//! - **Staging buffer**: allocation and release of a transport DMA host buffer
//!   (`mmap`/`munmap` of the driver's CMA region on the ioctl transport), which
//!   every `memcpy_h2d`/`memcpy_d2h` currently pays once per call, and the host
//!   copies into and out of a held buffer, which each transfer also performs.
//! - **Launch**: wall time of `launch` with the empty `null-rs` kernel on one
//!   shire and on all shires, without arguments and with a 32-byte argument
//!   blob. The difference between the last two is the cost of staging the
//!   arguments by DMA before the launch command.
//! - **DMA**: `memcpy_h2d` and `memcpy_d2h` from 64 B to 64 MiB. Each size is
//!   checked once by a host-to-device-to-host round trip, so an optimisation
//!   that corrupts data cannot report a speed-up.
//!
//! Each measurement is preceded by warm-up iterations and reported as the
//! minimum, median, 90th percentile and maximum wall time; throughput is the
//! transfer size divided by the median.
//!
//! # Usage
//! ```text
//! cargo run --release --example bench -- <null-rs.elf> [iterations] [staging MiB]
//! ```
//! `iterations` (default 50) applies to launches and transfers up to 1 MiB;
//! larger transfers use proportionally fewer, with a minimum of 5. `staging MiB`
//! sets the combined capacity of the device's persistent staging buffers
//! (default `DEFAULT_STAGING_CAPACITY`); transfers larger than half of it are
//! pipelined through two half-capacity buffers.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use et_soc1::{Device, LaunchOptions, Transport};

/// Warm-up iterations discarded before each measurement.
const WARMUP: usize = 3;

/// Transfer sizes exercised by the staging and DMA sections.
const SIZES: &[usize] = &[64, 4 << 10, 64 << 10, 1 << 20, 16 << 20, 64 << 20];

/// Size of the argument blob in the "with args" launch measurements; matches
/// the 32-byte `CacheTestArgs`.
const ARGS_BYTES: usize = 32;

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
        eprintln!("usage: bench <null-rs.elf> [iterations] [staging MiB]");
        std::process::exit(2);
    });
    let iterations = argv
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50)
        .max(1);
    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::io("read kernel ELF", e))?;

    let device = Device::open(0)?;
    if let Some(mebibytes) = argv.get(3).and_then(|s| s.parse::<usize>().ok()) {
        device.set_staging_capacity(mebibytes << 20);
    }
    let topo = device.topology()?;
    let all_shires = topo.shire_mask;
    let lowest_shire = all_shires.isolate_lowest_one();
    let dram = device.dram_info();
    println!(
        "Device: {} shires (mask {all_shires:#x}); {iterations} iterations, {WARMUP} warm-up",
        topo.num_shires()
    );
    println!(
        "DMA limits: element size {} B, {} elements per command, alignment {} B; \
         staging capacity {}",
        dram.dma_max_elem_size,
        dram.dma_max_elem_count,
        dram.dma_alignment,
        format_size(device.staging_capacity())
    );

    // Kernel first, so that it occupies its link address at DRAM base.
    let kernel = device.load_kernel(&elf)?;
    let largest = *SIZES.iter().max().unwrap();
    let region = device.alloc(largest as u64)?;

    print_header();

    // Staging buffer: allocation and release only.
    for &size in SIZES {
        let n = iterations_for(size, iterations);
        let samples = measure(n, || {
            let buffer = device.transport().dma_host_buffer(size)?;
            drop(buffer);
            Ok(())
        });
        report("staging alloc+free", size, &samples, false);
    }

    // Staging buffer: host copies into and out of an already mapped buffer.
    for &size in SIZES {
        let n = iterations_for(size, iterations);
        let source = vec![0x5Au8; size];
        let mut sink = vec![0u8; size];
        let mut buffer = match device.transport().dma_host_buffer(size) {
            Ok(buffer) => buffer,
            Err(e) => {
                report("staging write", size, &Err(e), true);
                continue;
            }
        };
        let samples = measure(n, || {
            buffer.as_mut_slice().copy_from_slice(&source);
            Ok(())
        });
        report("staging write", size, &samples, true);
        let samples = measure(n, || {
            sink.copy_from_slice(buffer.as_slice());
            Ok(())
        });
        report("staging read", size, &samples, true);
    }

    // Launch overhead.
    let no_args_one = LaunchOptions::new(lowest_shire);
    let no_args_all = LaunchOptions::new(all_shires);
    let with_args_one = LaunchOptions::new(lowest_shire).with_args(vec![0u8; ARGS_BYTES]);
    let launches: [(&str, &LaunchOptions); 3] = [
        ("launch 1 shire", &no_args_one),
        ("launch all shires", &no_args_all),
        ("launch 1 shire +args", &with_args_one),
    ];
    for (label, opts) in launches {
        let samples = measure(iterations, || device.launch(&kernel, opts).map(drop));
        report(label, 0, &samples, false);
    }

    // DMA throughput, with a round-trip check per size.
    for &size in SIZES {
        let n = iterations_for(size, iterations);
        let pattern: Vec<u8> = (0..size)
            .map(|i| (i.wrapping_mul(131) >> 3) as u8)
            .collect();
        let mut readback = vec![0u8; size];

        let upload = measure(n, || device.memcpy_h2d(&pattern, region.addr));
        report("memcpy_h2d", size, &upload, true);
        let download = measure(n, || device.memcpy_d2h(region.addr, &mut readback));
        report("memcpy_d2h", size, &download, true);

        if upload.is_ok() && download.is_ok() && readback != pattern {
            return Err(et_soc1::Error::Protocol(format!(
                "round trip of {} corrupted the data",
                format_size(size)
            )));
        }
    }

    println!("\nAll round trips verified.");
    Ok(())
}

/// Iteration count for a transfer of `size` bytes: `base` up to 1 MiB, then
/// scaled down in proportion to the size, with a floor of 5.
fn iterations_for(size: usize, base: usize) -> usize {
    let scale = (size >> 20).max(1);
    (base / scale).max(5)
}

/// Runs `operation` `WARMUP` times untimed, then `iterations` times timed.
/// Returns the first error encountered instead of the samples.
fn measure<F>(iterations: usize, mut operation: F) -> et_soc1::Result<Vec<Duration>>
where
    F: FnMut() -> et_soc1::Result<()>,
{
    for _ in 0..WARMUP {
        operation()?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        operation()?;
        samples.push(start.elapsed());
    }
    Ok(samples)
}

fn print_header() {
    println!(
        "\n{:<22} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "measurement", "size", "iters", "min us", "median us", "p90 us", "max us", "GB/s"
    );
    println!("{}", "-".repeat(92));
}

/// Prints one table row. A failed measurement is reported in place of its
/// statistics, so that one unsupported size does not hide the others.
fn report(label: &str, size: usize, samples: &et_soc1::Result<Vec<Duration>>, throughput: bool) {
    let size_text = if size == 0 {
        "-".to_string()
    } else {
        format_size(size)
    };
    let samples = match samples {
        Ok(s) => s,
        Err(e) => {
            println!("{label:<22} {size_text:>8}  failed: {e}");
            return;
        }
    };
    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let microseconds = |d: Duration| d.as_secs_f64() * 1e6;
    let percentile = |p: usize| sorted[(sorted.len() - 1) * p / 100];
    let median = percentile(50);
    let rate = if throughput {
        format!("{:.3}", size as f64 / median.as_secs_f64() / 1e9)
    } else {
        "-".to_string()
    };
    println!(
        "{label:<22} {size_text:>8} {:>6} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {rate:>9}",
        sorted.len(),
        microseconds(sorted[0]),
        microseconds(median),
        microseconds(percentile(90)),
        microseconds(sorted[sorted.len() - 1]),
    );
}

fn format_size(size: usize) -> String {
    match size {
        s if s >= 1 << 20 => format!("{} MiB", s >> 20),
        s if s >= 1 << 10 => format!("{} KiB", s >> 10),
        s => format!("{s} B"),
    }
}
