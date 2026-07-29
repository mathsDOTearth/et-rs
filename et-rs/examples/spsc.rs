//! Host launcher for the lock-free non-atomic SPSC kernel (`et-k-rs` -> `spsc-rs`).
//!
//! Loads and launches the kernel on one shire; the producer/consumer harts run a
//! lock-free non-atomic queue in shire-local scratchpad, and the consumer reports
//! the verified result through the trace buffer, which this program decodes.
//!
//! Software emulator (no hardware):
//! ```text
//! cargo run --features emu --example spsc -- \
//!     et-k-rs/target/riscv64imac-unknown-none-elf/release/spsc-rs
//! ```
//! Real hardware:
//! ```text
//! cargo run --example spsc -- \
//!     et-k-rs/target/riscv64imac-unknown-none-elf/release/spsc-rs
//! ```

use std::process::ExitCode;

use et_soc1::trace::{DecodedEntry, TraceBuffer};
use et_soc1::{Device, LaunchOptions, TraceConfig};

/// Single shire (the producer/consumer harts live in shire 0, neighbourhood 0).
const SHIRE_MASK: u64 = 0x1;
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
        eprintln!("usage: spsc <spsc-rs.elf>");
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
    let trace_buf = device.alloc(TRACE_BUFFER_SIZE)?;

    let opts = LaunchOptions::new(SHIRE_MASK).with_trace(TraceConfig::full(trace_buf, SHIRE_MASK));
    let result = device.launch(&kernel, &opts)?;
    println!(
        "Kernel completed in {} cycles (waited {}).",
        result.timing.execute_dur, result.timing.wait_dur
    );

    let mut host_trace = vec![0u8; TRACE_BUFFER_SIZE as usize];
    device.memcpy_d2h(trace_buf.addr, &mut host_trace)?;

    let mut pass = false;
    match TraceBuffer::parse(&host_trace) {
        Ok(tb) => {
            for entry in tb.entries() {
                if let DecodedEntry::String(s) = entry.decoded() {
                    let line = s.trim_end();
                    println!("[hart {}] {}", entry.hart_id, line);
                    if line.contains("RESULT PASS") {
                        pass = true;
                    }
                }
            }
        }
        Err(e) => eprintln!("trace buffer not decodable: {e}"),
    }

    println!(
        "\nSPSC lock-free non-atomic queue: {}",
        if pass { "PASS" } else { "FAIL / no result" }
    );
    if pass {
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol("SPSC did not report PASS".into()))
    }
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
