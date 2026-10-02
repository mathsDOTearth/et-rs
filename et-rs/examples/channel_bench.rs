//! Inter-Minion channel latency: measures the round-trip time of a message
//! exchanged between two Minions through device memory, as the basis for
//! deciding whether memory-mediated channels are viable between pipeline
//! stages.
//!
//! The `channel-bench-rs` kernel runs a ping-pong between the primary harts
//! of two Minions. Each message is written, written back to a chosen cache
//! level (`flush_va`), and announced by a flag that is itself written back;
//! the receiver polls the flag with an invalidation (`evict_va`) before every
//! read, then invalidates and reads the payload. The pong Minion echoes the
//! complement of each payload word, and both sides verify every word.
//!
//! The sweep covers four placements:
//!
//! | Placement | Ping | Pong | Levels |
//! |---|---|---|---|
//! | same neighbourhood | lowest shire, Minion 0 | same shire, Minion 1 | L2, L3, DDR |
//! | same shire | lowest shire, Minion 0 | same shire, Minion 31 | L2, L3, DDR |
//! | adjacent shire | lowest shire, Minion 0 | next shire, Minion 0 | L3, DDR |
//! | far shire | lowest shire, Minion 0 | highest shire, Minion 0 | L3, DDR |
//!
//! and payloads of 0 (flag only), 64 B, 1 KiB, 4 KiB, 16 KiB and 64 KiB in
//! each direction. "Adjacent" and "far" refer to shire numbering, which need
//! not reflect distance on the mesh.
//!
//! For each run the first [`WARMUP_ITERATIONS`] round trips are discarded and
//! the minimum, median, 90th percentile and maximum of the rest are reported,
//! with the median send time (payload writes and both writebacks, as seen by
//! the sender) and the effective bandwidth of one direction, payload bytes
//! over half the median round trip. A round trip includes the receiver's
//! reading and the reply's writing of every payload word, so it bounds the
//! cost of a send-receive pair from above.
//!
//! The benchmark fails if any run times out, any payload word is wrong, or
//! either Minion reports an error.
//!
//! # Usage
//! ```text
//! cargo run --release --example channel_bench -- <channel-bench-rs> [iterations]
//! ```
//! `iterations` is the number of measured round trips per run (default 200).
//! Build the kernel ELF (no file extension) with:
//! ```text
//! cargo build --release --bin channel-bench-rs
//! ```

use std::process::ExitCode;

use et_abi::{
    CHANNEL_HEADER_COMPLETED, CHANNEL_HEADER_ERRORS, CHANNEL_HEADER_FAILED_ITERATION,
    CHANNEL_HEADER_STATUS, CHANNEL_PING_HEADER_OFFSET, CHANNEL_PONG_HEADER_OFFSET,
    CHANNEL_SAMPLE_WORDS, CHANNEL_SAMPLES_OFFSET, CHANNEL_STATUS_BAD_ARGUMENTS, CHANNEL_STATUS_OK,
    CHANNEL_STATUS_TIMEOUT, ChannelBenchArgs, DeviceArgs, channel_buffer_bytes,
    channel_results_bytes,
};
use et_soc1::{Device, LaunchOptions, LoadedKernel};

/// Payload bytes per message in each direction.
const MESSAGE_SIZES: [u64; 6] = [0, 64, 1 << 10, 4 << 10, 16 << 10, 64 << 10];
/// Round trips discarded at the start of each run (cold caches and TLB).
const WARMUP_ITERATIONS: u32 = 10;
/// Measured round trips per run unless overridden on the command line.
const DEFAULT_MEASURED_ITERATIONS: u32 = 200;
/// Clock assumed when the transport does not report one (aifoundry3 runs at
/// 600 MHz).
const FALLBACK_CLOCK_MHZ: u32 = 600;
/// Time after which a Minion abandons a wait for a flag.
const WAIT_TIMEOUT_SECONDS: u64 = 1;

/// Cache level at which messages are exchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    L2,
    L3,
    Memory,
}

impl Level {
    /// [`ChannelBenchArgs::cache_level`] code: the `CacheDest` discriminant.
    fn code(self) -> u32 {
        match self {
            Level::L2 => 1,
            Level::L3 => 2,
            Level::Memory => 3,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Level::L2 => "L2",
            Level::L3 => "L3",
            Level::Memory => "DDR",
        }
    }
}

/// A Minion, identified by shire and index within the shire.
#[derive(Clone, Copy, Debug)]
struct Minion {
    shire: u32,
    index: u32,
}

/// A pair of Minions and the levels at which they are measured.
struct Placement {
    name: &'static str,
    ping: Minion,
    pong: Minion,
    levels: &'static [Level],
}

/// Statistics of one run, in cycles.
struct Summary {
    minimum: u64,
    median: u64,
    percentile_90: u64,
    maximum: u64,
    median_send: u64,
}

/// Header written by one Minion.
struct Header {
    status: u64,
    errors: u64,
    completed: u64,
    failed_iteration: u64,
}

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
        eprintln!("usage: channel_bench <channel-bench-rs> [iterations]");
        std::process::exit(2);
    });
    let measured_iterations = argv
        .get(2)
        .and_then(|s| s.parse::<u32>().ok())
        .map(|n| n.clamp(1, 100_000))
        .unwrap_or(DEFAULT_MEASURED_ITERATIONS);
    let iterations = WARMUP_ITERATIONS + measured_iterations;
    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::io("read kernel ELF", e))?;

    let device = Device::open(0)?;
    let topo = device.topology()?;
    let (clock_mhz, clock_note) = match device.properties()?.minion_boot_freq {
        Some(mhz) if mhz > 0 => (mhz, "reported"),
        _ => (FALLBACK_CLOCK_MHZ, "assumed"),
    };
    let present: Vec<u32> = (0..u64::BITS)
        .filter(|&s| topo.shire_mask & (1 << s) != 0)
        .collect();
    let Some(&lowest) = present.first() else {
        return Err(et_soc1::Error::Protocol("no shires present".into()));
    };
    println!(
        "Device: {} shires present; clock {clock_mhz} MHz ({clock_note}); \
         {measured_iterations} measured round trips per run after {WARMUP_ITERATIONS} warm-up",
        present.len()
    );

    let placements = placements(&present, lowest);
    let kernel = device.load_kernel(&elf)?;
    let timeout_cycles = u64::from(clock_mhz) * 1_000_000 * WAIT_TIMEOUT_SECONDS;

    let mut failures = Vec::new();
    let mut epoch = 0u32;
    for placement in &placements {
        println!(
            "\n{}: shire {} Minion {} <-> shire {} Minion {}",
            placement.name,
            placement.ping.shire,
            placement.ping.index,
            placement.pong.shire,
            placement.pong.index
        );
        println!(
            "  {:<5} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}",
            "level", "payload", "min us", "median us", "p90 us", "max us", "send us", "MB/s"
        );
        for &level in placement.levels {
            for message_bytes in MESSAGE_SIZES {
                epoch += 1;
                let args = ChannelBenchArgs {
                    buffer: 0,
                    results: 0,
                    message_bytes,
                    timeout_cycles,
                    ping_shire: placement.ping.shire,
                    ping_minion: placement.ping.index,
                    pong_shire: placement.pong.shire,
                    pong_minion: placement.pong.index,
                    cache_level: level.code(),
                    iterations,
                    epoch,
                    reserved: 0,
                };
                let label = format!(
                    "{} {} {}",
                    placement.name,
                    level.name(),
                    size_label(message_bytes)
                );
                match execute(&device, &kernel, args) {
                    Ok(samples) => {
                        let measured = &samples[WARMUP_ITERATIONS as usize..];
                        let summary = summarise(measured);
                        print_row(level, message_bytes, &summary, clock_mhz);
                    }
                    Err(reason) => {
                        println!(
                            "  {:<5} {:>8}  FAILED: {reason}",
                            level.name(),
                            size_label(message_bytes)
                        );
                        failures.push(label);
                    }
                }
            }
        }
    }

    if failures.is_empty() {
        println!("\nCHANNEL BENCH PASSED: every message delivered and verified");
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "CHANNEL BENCH FAILED: {}",
            failures.join(", ")
        )))
    }
}

/// The placements available on a device with the given present shires.
fn placements(present: &[u32], lowest: u32) -> Vec<Placement> {
    const SAME_SHIRE_LEVELS: &[Level] = &[Level::L2, Level::L3, Level::Memory];
    const CROSS_SHIRE_LEVELS: &[Level] = &[Level::L3, Level::Memory];
    let origin = Minion {
        shire: lowest,
        index: 0,
    };
    let mut placements = vec![
        Placement {
            name: "same neighbourhood",
            ping: origin,
            pong: Minion {
                shire: lowest,
                index: 1,
            },
            levels: SAME_SHIRE_LEVELS,
        },
        Placement {
            name: "same shire, other neighbourhood",
            ping: origin,
            pong: Minion {
                shire: lowest,
                index: 31,
            },
            levels: SAME_SHIRE_LEVELS,
        },
    ];
    if let Some(&adjacent) = present.get(1) {
        placements.push(Placement {
            name: "adjacent shire",
            ping: origin,
            pong: Minion {
                shire: adjacent,
                index: 0,
            },
            levels: CROSS_SHIRE_LEVELS,
        });
    }
    if present.len() > 2 {
        let far = present[present.len() - 1];
        placements.push(Placement {
            name: "far shire",
            ping: origin,
            pong: Minion {
                shire: far,
                index: 0,
            },
            levels: CROSS_SHIRE_LEVELS,
        });
    }
    placements
}

/// Allocates fresh buffers, runs one ping-pong and returns the round-trip and
/// send cycles of every iteration, or the reason the run failed.
///
/// Fresh buffers are used for every run so that no line of an earlier run
/// can be cached at the same address.
fn execute(
    device: &Device,
    kernel: &LoadedKernel,
    mut args: ChannelBenchArgs,
) -> Result<Vec<(u64, u64)>, String> {
    let buffer_words = (channel_buffer_bytes(args.message_bytes) / 8) as usize;
    let results_words = (channel_results_bytes(args.iterations) / 8) as usize;
    let buffer = device
        .upload(&vec![0u64; buffer_words])
        .map_err(|e| format!("buffer upload: {e}"))?;
    // All ones marks a header or sample the kernel did not write.
    let results = device
        .upload(&vec![u64::MAX; results_words])
        .map_err(|e| format!("results upload: {e}"))?;
    args.buffer = buffer.addr();
    args.results = results.addr();

    let shire_mask = (1u64 << args.ping_shire) | (1u64 << args.pong_shire);
    let opts = LaunchOptions::new(shire_mask).with_args(args.as_bytes().to_vec());
    device
        .launch(kernel, &opts)
        .map_err(|e| format!("launch: {e}"))?;
    let words = device
        .download(&results)
        .map_err(|e| format!("results download: {e}"))?;

    let ping = header(&words, CHANNEL_PING_HEADER_OFFSET);
    let pong = header(&words, CHANNEL_PONG_HEADER_OFFSET);
    let iterations = u64::from(args.iterations);
    for (side, h) in [("ping", &ping), ("pong", &pong)] {
        match h.status {
            CHANNEL_STATUS_OK => {}
            CHANNEL_STATUS_TIMEOUT => {
                return Err(format!(
                    "{side} timed out waiting for message {} ({} completed)",
                    h.failed_iteration, h.completed
                ));
            }
            CHANNEL_STATUS_BAD_ARGUMENTS => return Err(format!("{side} rejected the arguments")),
            u64::MAX => return Err(format!("{side} did not run")),
            other => return Err(format!("{side} reported unknown status {other}")),
        }
        if h.errors != 0 {
            return Err(format!("{side} received {} wrong payload words", h.errors));
        }
        if h.completed != iterations {
            return Err(format!("{side} completed {} of {iterations}", h.completed));
        }
    }

    let first = (CHANNEL_SAMPLES_OFFSET / 8) as usize;
    let stride = CHANNEL_SAMPLE_WORDS as usize;
    Ok((0..args.iterations as usize)
        .map(|i| (words[first + i * stride], words[first + i * stride + 1]))
        .collect())
}

/// Decodes the header line at byte `offset` of the results area.
fn header(words: &[u64], offset: u64) -> Header {
    let base = (offset / 8) as usize;
    Header {
        status: words[base + CHANNEL_HEADER_STATUS],
        errors: words[base + CHANNEL_HEADER_ERRORS],
        completed: words[base + CHANNEL_HEADER_COMPLETED],
        failed_iteration: words[base + CHANNEL_HEADER_FAILED_ITERATION],
    }
}

/// Order statistics of the round-trip times and the median send time.
fn summarise(samples: &[(u64, u64)]) -> Summary {
    let mut round_trips: Vec<u64> = samples.iter().map(|&(rtt, _)| rtt).collect();
    let mut sends: Vec<u64> = samples.iter().map(|&(_, send)| send).collect();
    round_trips.sort_unstable();
    sends.sort_unstable();
    let n = round_trips.len();
    Summary {
        minimum: round_trips[0],
        median: round_trips[n / 2],
        percentile_90: round_trips[(n * 9 / 10).min(n - 1)],
        maximum: round_trips[n - 1],
        median_send: sends[n / 2],
    }
}

fn print_row(level: Level, message_bytes: u64, summary: &Summary, clock_mhz: u32) {
    let micros = |cycles: u64| cycles as f64 / f64::from(clock_mhz);
    // Bytes per microsecond is MB/s; one direction takes half a round trip.
    let bandwidth = if message_bytes == 0 {
        "-".to_string()
    } else {
        format!(
            "{:.1}",
            message_bytes as f64 / (micros(summary.median) / 2.0)
        )
    };
    println!(
        "  {:<5} {:>8} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>10}",
        level.name(),
        size_label(message_bytes),
        micros(summary.minimum),
        micros(summary.median),
        micros(summary.percentile_90),
        micros(summary.maximum),
        micros(summary.median_send),
        bandwidth
    );
}

fn size_label(bytes: u64) -> String {
    match bytes {
        0 => "flag".to_string(),
        b if b >= 1024 && b % 1024 == 0 => format!("{} KiB", b / 1024),
        b => format!("{b} B"),
    }
}
