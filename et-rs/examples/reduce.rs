//! Host driver for the data-parallel reduction kernel (`et-k-rs` -> `reduce-rs`).
//!
//! Uploads a large array, launches the reduction across a shire of harts (each
//! reduces its disjoint slice into its own cache-line-padded partial), DMAs the
//! partials back, and combines them -- verifying the total against the known sum.
//!
//! Emulator (no hardware):
//! ```text
//! cargo run --features emu --example reduce -- \
//!     et-k-rs/target/riscv64imac-unknown-none-elf/release/reduce-rs
//! ```
//! Real hardware:
//! ```text
//! cargo run --example reduce -- \
//!     et-k-rs/target/riscv64imac-unknown-none-elf/release/reduce-rs
//! ```

use std::process::ExitCode;

use et_abi::{DeviceArgs, ReduceArgs};
use et_soc1::trace::{DecodedEntry, TraceBuffer};
use et_soc1::{Device, LaunchOptions, TraceConfig};

const SHIRE_MASK: u64 = 0x1;
/// Harts participating (a full shire).
const N_HARTS: u32 = 64;
/// Input length (elements). A multiple of `N_HARTS * 16` keeps each hart's slice
/// cache-line aligned. 2^18 elements = 1 MiB.
const N: u32 = 1 << 18;
/// Per-hart output stride: one cache line, to avoid false sharing.
const CACHE_LINE: usize = 64;
const TRACE_BUFFER_SIZE: u64 = 4096 * 2048;

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
        eprintln!("usage: reduce <reduce-rs.elf>");
        std::process::exit(2);
    });
    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::Io {
        op: "read kernel ELF",
        source: e,
    })?;

    let device = open_device()?;
    let di = device.dram_info();
    println!(
        "Device ready: DRAM base {:#x}, size {} bytes",
        di.base, di.size
    );

    let kernel = device.load_kernel(&elf)?;

    // Device regions: input array, per-hart partials (cache-line padded), trace.
    let input_region = device.alloc(N as u64 * 4)?;
    let out_region = device.alloc(N_HARTS as u64 * CACHE_LINE as u64)?;
    let trace_buf = device.alloc(TRACE_BUFFER_SIZE)?;

    // Host input: element i = i + 1, so the exact sum is known.
    let host_in: Vec<u32> = (0..N).map(|i| i + 1).collect();
    // SAFETY: reinterpret the u32 vector as bytes for the byte-oriented DMA API.
    let in_bytes =
        unsafe { std::slice::from_raw_parts(host_in.as_ptr() as *const u8, host_in.len() * 4) };
    device.memcpy_h2d(in_bytes, input_region.addr)?;

    // Kernel args: the same struct the kernel reads (et-abi), so host and device
    // cannot drift on layout.
    let args = ReduceArgs {
        input: input_region.addr,
        out: out_region.addr,
        n: N,
        n_harts: N_HARTS,
    };

    let opts = LaunchOptions::new(SHIRE_MASK)
        .with_trace(TraceConfig::full(trace_buf, SHIRE_MASK))
        .with_args(args.as_bytes().to_vec());
    let launch = device.launch(&kernel, &opts);

    // Print any trace lines regardless of launch outcome (on an exception the
    // firmware still fills the trace buffer -- useful for diagnostics).
    if let Ok(tb) = TraceBuffer::parse(&out_trace(&device, trace_buf)?) {
        for entry in tb.entries() {
            if let DecodedEntry::String(s) = entry.decoded() {
                println!("[hart {}] {}", entry.hart_id, s.trim_end());
            }
        }
    }

    match &launch {
        Ok(r) => println!(
            "Kernel completed in {} cycles (waited {}).",
            r.timing.execute_dur, r.timing.wait_dur
        ),
        Err(e) => println!("Launch error: {e}"),
    }
    launch?;

    // Combine the per-hart partials (one u64 at each cache line).
    let mut out = vec![0u8; N_HARTS as usize * CACHE_LINE];
    device.memcpy_d2h(out_region.addr, &mut out)?;
    let mut total: u64 = 0;
    let mut nonzero = 0;
    for h in 0..N_HARTS as usize {
        let off = h * CACHE_LINE;
        let partial = u64::from_le_bytes(out[off..off + 8].try_into().unwrap());
        total = total.wrapping_add(partial);
        if partial != 0 {
            nonzero += 1;
        }
    }

    let expected = (N as u64) * (N as u64 + 1) / 2;
    println!(
        "\nReduction over {N} elements on {N_HARTS} harts ({nonzero} contributed):\n  \
         device total = {total}\n  expected     = {expected}"
    );
    if total == expected {
        println!("RESULT PASS");
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "reduction mismatch: got {total}, expected {expected}"
        )))
    }
}

/// DMA the trace buffer back for decoding.
fn out_trace<T: et_soc1::Transport>(
    device: &Device<T>,
    trace_buf: et_soc1::DeviceRegion,
) -> et_soc1::Result<Vec<u8>> {
    let mut host = vec![0u8; trace_buf.size as usize];
    device.memcpy_d2h(trace_buf.addr, &mut host)?;
    Ok(host)
}

#[cfg(feature = "emu")]
fn open_device() -> et_soc1::Result<Device<et_soc1::FfiTransport>> {
    let sdk_prefix = std::env::var("ET_SDK_PREFIX").unwrap_or_else(|_| "/opt/et".to_string());
    let run_dir = std::env::current_dir()
        .map_err(|e| et_soc1::Error::Io {
            op: "current_dir",
            source: e,
        })?
        .join("sysemu-run");
    std::fs::create_dir_all(&run_dir).map_err(|e| et_soc1::Error::Io {
        op: "create run dir",
        source: e,
    })?;
    eprintln!("Booting software emulator (this can take a while)...");
    Device::open_emulator(&sdk_prefix, &run_dir)
}

#[cfg(not(feature = "emu"))]
fn open_device() -> et_soc1::Result<Device> {
    Device::open(0)
}
