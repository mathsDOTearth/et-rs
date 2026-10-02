//! PS transcendental accuracy: measures `FEXP.PS` (2^x), `FLOG.PS` (log2 x)
//! and `FRCP.PS` (1/x) on ET-SoC-1 silicon against an f64 reference, and
//! checks the special cases specified by the PRM.
//!
//! Four launches of `ps-math-test-rs` are made:
//!
//! 1. **copy**: FLQ2 load and FSQ2 store with no arithmetic. Every lane must
//!    be bit-identical to its input, including NaN payloads and subnormals.
//!    The arithmetic results are only meaningful if this passes.
//! 2. **exp2**, 3. **log2**, 4. **reciprocal**: each input set combines the
//!    special values of that function, a strided sample of the whole f32 bit
//!    space (every 1021st pattern, about 4.2 million values covering every
//!    exponent, both signs, subnormals and NaNs) and one million pseudo-random
//!    values in the function's numerically interesting domain.
//!
//! # Expected results
//!
//! The PRM specifies a maximum error of 1 ULP with round-towards-zero,
//! subnormal inputs treated as zero and subnormal results flushed to zero.
//! Each input is classified as:
//!
//! - **exact**: a special case with a single correct bit pattern (for
//!   example 2^-inf = +0, log2(+-0) = -inf, 1/-0 = -inf, and the saturating
//!   ranges 2^x = +0 for x < -126, 2^x = +inf for x >= 128, 1/x = +-0 for
//!   |x| > 2^126). The output must match bit-exactly, sign included;
//! - **NaN**: any quiet NaN is accepted;
//! - **approximate**: the error is `|result - reference| / ulp(reference)`,
//!   with the reference computed in f64 and `ulp` the spacing of f32 values
//!   at the reference's binade (no smaller than 2^-149). The function passes
//!   if no approximate lane exceeds its regression limit.
//!
//! # Measured accuracy (aifoundry3, 2026-10-02)
//!
//! The PRM's 1 ULP bound holds for `FRCP.PS` only. Maximum errors over this
//! input set were 1.313 ULP for `FEXP.PS` (0.18% of lanes above 1 ULP),
//! 2.379 ULP for `FLOG.PS` (1.5% above 1 ULP, worst where the result
//! approaches zero) and 0.998 ULP for `FRCP.PS`. No function truncates
//! consistently towards zero: 34%, 1.5% and 47% of results respectively
//! exceed the exact value in magnitude. Every special case matched the PRM
//! bit-exactly. The regression limits ([`Operation::ulp_limit`]) are set just
//! above these measurements, so that a pass on correct silicon is reproducible
//! and a change in the hardware or the wrappers is detected; lanes beyond the
//! PRM's 1 ULP are counted separately for information.
//!
//! The output is prefilled with a signalling-NaN sentinel, which no
//! arithmetic instruction produces, so an unwritten lane cannot satisfy a NaN
//! expectation.
//!
//! # Usage
//! ```text
//! cargo run --release --example ps_math_test -- <ps-math-test-rs> [shires]
//! ```
//! `shires` limits the run to the lowest `shires` present shires; it defaults
//! to all present. Build the kernel ELF (no file extension) with:
//! ```text
//! cargo build --release --bin ps-math-test-rs
//! ```

use std::process::ExitCode;

use et_abi::{
    DeviceArgs, PS_MATH_OP_COPY, PS_MATH_OP_EXP2, PS_MATH_OP_LOG2, PS_MATH_OP_RECIPROCAL,
    PsMathTestArgs,
};
use et_soc1::{Device, LaunchOptions, LoadedKernel};

/// `f32` elements per kernel unit (one 64-byte cache line).
const UNIT_ELEMENTS: usize = 16;
/// Output prefill: a signalling NaN, never produced by arithmetic.
const SENTINEL_BITS: u32 = 0x7FBA_DBAD;
/// Stride through the f32 bit space for the whole-range sample; odd, so the
/// multiples are distinct modulo 2^32.
const BIT_SPACE_STRIDE: u64 = 1021;
/// Pseudo-random values drawn from each function's interesting domain.
const DOMAIN_SAMPLES: usize = 1 << 20;
/// Error bound stated by the PRM for all three functions, in ULP; reported,
/// not enforced.
const PRM_ULP_BOUND: f64 = 1.0;
/// Failing lanes printed per function, largest error first.
const REPORT_LIMIT: usize = 8;

const SIGN_BIT: u32 = 0x8000_0000;
const QUIET_NAN: u32 = 0x7FC0_0000;
const SIGNALLING_NAN: u32 = 0x7FA0_0000;
const NEGATIVE_QUIET_NAN: u32 = 0xFFC0_0000;
const MIN_SUBNORMAL: u32 = 0x0000_0001;
const MAX_SUBNORMAL: u32 = 0x007F_FFFF;
const MIN_NORMAL: u32 = 0x0080_0000;

/// The function under test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Copy,
    Exp2,
    Log2,
    Reciprocal,
}

impl Operation {
    fn code(self) -> u32 {
        match self {
            Operation::Copy => PS_MATH_OP_COPY,
            Operation::Exp2 => PS_MATH_OP_EXP2,
            Operation::Log2 => PS_MATH_OP_LOG2,
            Operation::Reciprocal => PS_MATH_OP_RECIPROCAL,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Operation::Copy => "copy (FLQ2/FSQ2)",
            Operation::Exp2 => "FEXP.PS 2^x",
            Operation::Log2 => "FLOG.PS log2 x",
            Operation::Reciprocal => "FRCP.PS 1/x",
        }
    }

    /// Regression limit on the error of an approximate lane, in ULP, set
    /// just above the maximum measured on hardware (see the module
    /// documentation).
    fn ulp_limit(self) -> f64 {
        match self {
            Operation::Copy => 0.0,
            Operation::Exp2 => 1.5,
            Operation::Log2 => 2.5,
            Operation::Reciprocal => 1.0,
        }
    }
}

/// The correct result for one input lane.
#[derive(Clone, Copy)]
enum Expected {
    /// Exactly this bit pattern.
    Exact(u32),
    /// Any quiet NaN.
    NotANumber,
    /// Within [`Operation::ulp_limit`] of this reference value.
    Approximate(f64),
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
        eprintln!("usage: ps_math_test <ps-math-test-rs> [shires]");
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
    let mut shire_mask = 0u64;
    for shire in (0..u64::BITS)
        .filter(|&s| topo.shire_mask & (1 << s) != 0)
        .take(n_shires)
    {
        shire_mask |= 1 << shire;
    }
    println!("Device: {max_shires} shires present; running {n_shires} (mask {shire_mask:#x})");

    let kernel = device.load_kernel(&elf)?;

    let mut failed = Vec::new();
    for operation in [
        Operation::Copy,
        Operation::Exp2,
        Operation::Log2,
        Operation::Reciprocal,
    ] {
        let inputs = inputs_for(operation);
        let outputs = execute(&device, &kernel, shire_mask, operation, &inputs)?;
        let passed = if operation == Operation::Copy {
            check_copy(&inputs, &outputs)
        } else {
            check_function(operation, &inputs, &outputs)
        };
        if !passed {
            failed.push(operation.name());
            if operation == Operation::Copy {
                return Err(et_soc1::Error::Protocol(
                    "FLQ2/FSQ2 copy failed; arithmetic results would be meaningless".into(),
                ));
            }
        }
    }

    if failed.is_empty() {
        println!("\nPS MATH PASSED: all functions within their limits, all special cases exact");
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "PS MATH FAILED: {}",
            failed.join(", ")
        )))
    }
}

/// Uploads `inputs` (padded to whole units), launches the kernel for
/// `operation` and returns the outputs corresponding to `inputs`.
fn execute(
    device: &Device,
    kernel: &LoadedKernel,
    shire_mask: u64,
    operation: Operation,
    inputs: &[f32],
) -> et_soc1::Result<Vec<f32>> {
    let padded_len = inputs.len().next_multiple_of(UNIT_ELEMENTS);
    let mut padded = inputs.to_vec();
    padded.resize(padded_len, 1.0);

    let input = device.upload(&padded)?;
    let output = device.upload(&vec![f32::from_bits(SENTINEL_BITS); padded_len])?;
    let args = PsMathTestArgs {
        input: input.addr(),
        output: output.addr(),
        count: padded_len as u64,
        shire_mask: shire_mask as u32,
        operation: operation.code(),
    };
    let opts = LaunchOptions::new(shire_mask).with_args(args.as_bytes().to_vec());
    device.launch(kernel, &opts)?;

    let mut results = device.download(&output)?;
    results.truncate(inputs.len());
    Ok(results)
}

/// Input set for `operation`: special values, the strided bit-space sample
/// and, for the arithmetic functions, the domain sample.
fn inputs_for(operation: Operation) -> Vec<f32> {
    let mut bits: Vec<u32> = vec![
        0,
        SIGN_BIT,
        MIN_SUBNORMAL,
        MIN_SUBNORMAL | SIGN_BIT,
        MAX_SUBNORMAL,
        MAX_SUBNORMAL | SIGN_BIT,
        MIN_NORMAL,
        MIN_NORMAL | SIGN_BIT,
        f32::MAX.to_bits(),
        f32::MIN.to_bits(),
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        QUIET_NAN,
        SIGNALLING_NAN,
        NEGATIVE_QUIET_NAN,
    ];
    let boundaries: &[f32] = match operation {
        Operation::Copy => &[],
        Operation::Exp2 => &[
            1.0, -1.0, 0.5, -0.5, 1e-7, -1e-7, 127.0, -126.0, -125.5, 128.0, -127.0,
        ],
        Operation::Log2 => &[1.0, 2.0, 0.5, 3.0, 1e-30, 1e30, -1.0],
        Operation::Reciprocal => &[1.0, -1.0, 2.0, 3.0, -3.0, 7.0, 1e-30, 1e30],
    };
    bits.extend(boundaries.iter().map(|v| v.to_bits()));
    // Neighbours of the saturation thresholds and of 1.0.
    let neighbours: &[f32] = match operation {
        Operation::Copy => &[],
        Operation::Exp2 => &[-126.0, 128.0],
        Operation::Log2 => &[1.0],
        Operation::Reciprocal => &[2f32.powi(126), -2f32.powi(126)],
    };
    for &value in neighbours {
        bits.push(value.to_bits() - 1);
        bits.push(value.to_bits() + 1);
    }

    // Strided whole-space sample.
    let mut pattern = 0u64;
    while pattern < 1 << 32 {
        bits.push(pattern as u32);
        pattern += BIT_SPACE_STRIDE;
    }

    let mut values: Vec<f32> = bits.into_iter().map(f32::from_bits).collect();

    // Domain sample.
    let mut generator = XorShift(0x9E37_79B9_7F4A_7C15 ^ u64::from(operation.code()));
    let domain_samples = if operation == Operation::Copy {
        0
    } else {
        DOMAIN_SAMPLES
    };
    for _ in 0..domain_samples {
        let value = match operation {
            Operation::Copy => unreachable!(),
            // Uniform over the transition region, including both saturation
            // thresholds.
            Operation::Exp2 => (generator.unit() * 260.0 - 130.0) as f32,
            // Half log-uniform over all positive normals, half uniform over
            // [0.5, 2), where log2 approaches zero and relative error is most
            // demanding.
            Operation::Log2 => {
                if generator.next_u64() & 1 == 0 {
                    f32::from_bits(
                        MIN_NORMAL
                            + (generator.next_u64() % u64::from(0x7F80_0000 - MIN_NORMAL)) as u32,
                    )
                } else {
                    (0.5 + generator.unit() * 1.5) as f32
                }
            }
            // Log-uniform over all normals of either sign.
            Operation::Reciprocal => {
                let magnitude = MIN_NORMAL
                    + (generator.next_u64() % u64::from(0x7F80_0000 - MIN_NORMAL)) as u32;
                f32::from_bits(magnitude | (generator.next_u64() as u32 & SIGN_BIT))
            }
        };
        values.push(value);
    }
    values
}

/// Verifies that every lane of the copy is bit-identical to its input.
fn check_copy(inputs: &[f32], outputs: &[f32]) -> bool {
    let mismatches: Vec<usize> = (0..inputs.len())
        .filter(|&i| inputs[i].to_bits() != outputs[i].to_bits())
        .collect();
    println!("\n{}: {} lanes", Operation::Copy.name(), inputs.len());
    for &i in mismatches.iter().take(REPORT_LIMIT) {
        println!(
            "  FAIL lane {i}: input {:#010x}, output {:#010x}",
            inputs[i].to_bits(),
            outputs[i].to_bits()
        );
    }
    if mismatches.is_empty() {
        println!("  all lanes bit-exact");
        true
    } else {
        println!("  {} lanes differ", mismatches.len());
        false
    }
}

/// Classifies every lane, prints the error statistics and returns whether
/// the function met the specification.
fn check_function(operation: Operation, inputs: &[f32], outputs: &[f32]) -> bool {
    // Histogram buckets of approximate-lane error, upper bounds in ULP.
    const BUCKETS: [(f64, &str); 5] = [
        (0.0, "exact"),
        (0.5, "<= 0.5"),
        (1.0, "<= 1"),
        (2.0, "<= 2"),
        (f64::INFINITY, "> 2"),
    ];
    let mut histogram = [0usize; BUCKETS.len()];
    let mut approximate = 0usize;
    let mut rounded_away = 0usize;
    let mut beyond_prm_bound = 0usize;
    let mut max_absolute_error = 0f64;
    let mut worst: Option<(f64, usize, f64)> = None;
    let mut exact_cases = 0usize;
    let mut nan_cases = 0usize;
    // (error in ULP, description); special-case failures carry infinity.
    let mut failures: Vec<(f64, String)> = Vec::new();
    let mut nan_patterns: Vec<u32> = Vec::new();

    for (lane, (&input, &output)) in inputs.iter().zip(outputs).enumerate() {
        let output_bits = output.to_bits();
        match expected(operation, input) {
            Expected::Exact(bits) => {
                exact_cases += 1;
                if output_bits != bits {
                    failures.push((f64::INFINITY, format!(
                        "lane {lane}: input {input:e} ({:#010x}) gave {output:e} ({output_bits:#010x}), \
                         expected {:e} ({bits:#010x})",
                        input.to_bits(),
                        f32::from_bits(bits),
                    )));
                }
            }
            Expected::NotANumber => {
                nan_cases += 1;
                if !output.is_nan() || output_bits == SENTINEL_BITS {
                    failures.push((f64::INFINITY, format!(
                        "lane {lane}: input {input:e} ({:#010x}) gave {output:e} ({output_bits:#010x}), \
                         expected NaN",
                        input.to_bits(),
                    )));
                } else if !nan_patterns.contains(&output_bits) {
                    nan_patterns.push(output_bits);
                }
            }
            Expected::Approximate(reference) => {
                approximate += 1;
                let error = ulp_error(output, reference);
                let bucket = BUCKETS
                    .iter()
                    .position(|&(bound, _)| error <= bound)
                    .unwrap_or(BUCKETS.len() - 1);
                histogram[bucket] += 1;
                if f64::from(output).abs() > reference.abs() {
                    rounded_away += 1;
                }
                if error > PRM_ULP_BOUND {
                    beyond_prm_bound += 1;
                }
                max_absolute_error = max_absolute_error.max((f64::from(output) - reference).abs());
                if worst.is_none_or(|(largest, _, _)| error > largest) {
                    worst = Some((error, lane, reference));
                }
                if error > operation.ulp_limit() {
                    failures.push((error, format!(
                        "lane {lane}: input {input:e} ({:#010x}) gave {output:e} ({output_bits:#010x}), \
                         reference {reference:e}, error {error:.3} ULP",
                        input.to_bits(),
                    )));
                }
            }
        }
    }

    println!(
        "\n{}: {} lanes ({approximate} approximate, {exact_cases} exact special, {nan_cases} NaN)",
        operation.name(),
        inputs.len()
    );
    if let Some((error, lane, reference)) = worst {
        println!(
            "  max error {error:.4} ULP at input {:e} ({:#010x}): output {:e}, reference {reference:e}",
            inputs[lane],
            inputs[lane].to_bits(),
            outputs[lane],
        );
    }
    let histogram_text: Vec<String> = BUCKETS
        .iter()
        .zip(histogram)
        .map(|(&(_, label), count)| format!("{label}: {count}"))
        .collect();
    println!("  error histogram: {}", histogram_text.join(", "));
    println!("  max absolute error: {max_absolute_error:e}");
    println!(
        "  above the PRM bound of {PRM_ULP_BOUND} ULP: {beyond_prm_bound} of {approximate}; \
         regression limit {} ULP",
        operation.ulp_limit()
    );
    println!(
        "  |result| > |reference| (not truncated towards zero): {rounded_away} of {approximate}"
    );
    let nan_text: Vec<String> = nan_patterns.iter().map(|b| format!("{b:#010x}")).collect();
    println!("  NaN results observed: [{}]", nan_text.join(", "));

    failures.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, failure) in failures.iter().take(REPORT_LIMIT) {
        println!("  FAIL {failure}");
    }
    if failures.is_empty() {
        println!("  PASS");
        true
    } else {
        println!("  {} lanes failed", failures.len());
        false
    }
}

/// The PRM-specified result of `operation` on `input`.
fn expected(operation: Operation, input: f32) -> Expected {
    if input.is_nan() {
        return Expected::NotANumber;
    }
    // Subnormal inputs are treated as zero of the same sign.
    let x = if input.is_subnormal() {
        f32::from_bits(input.to_bits() & SIGN_BIT)
    } else {
        input
    };
    let negative = x.is_sign_negative();
    let signed_zero = if negative { SIGN_BIT } else { 0 };
    let signed_infinity = f32::INFINITY.to_bits() | signed_zero;
    match operation {
        Operation::Copy => Expected::Exact(input.to_bits()),
        Operation::Exp2 => {
            if x < -126.0 {
                Expected::Exact(0)
            } else if x >= 128.0 {
                Expected::Exact(f32::INFINITY.to_bits())
            } else if x == 0.0 {
                Expected::Exact(1f32.to_bits())
            } else {
                Expected::Approximate(f64::from(x).exp2())
            }
        }
        Operation::Log2 => {
            if x == 0.0 {
                Expected::Exact(f32::NEG_INFINITY.to_bits())
            } else if negative {
                Expected::NotANumber
            } else if x == f32::INFINITY {
                Expected::Exact(f32::INFINITY.to_bits())
            } else if x == 1.0 {
                Expected::Exact(0)
            } else {
                Expected::Approximate(f64::from(x).log2())
            }
        }
        Operation::Reciprocal => {
            if x == 0.0 {
                Expected::Exact(signed_infinity)
            } else if x.abs() > 2f32.powi(126) {
                // Includes +-inf.
                Expected::Exact(signed_zero)
            } else {
                Expected::Approximate(1.0 / f64::from(x))
            }
        }
    }
}

/// Error of `output` relative to `reference`, in units of the spacing of f32
/// values in the reference's binade. Non-finite outputs give infinity.
fn ulp_error(output: f32, reference: f64) -> f64 {
    if !output.is_finite() {
        return f64::INFINITY;
    }
    let biased_exponent = ((reference.to_bits() >> 52) & 0x7FF) as i32;
    let binade = (biased_exponent - 1023).max(-126);
    let ulp = 2f64.powi(binade - 23);
    (f64::from(output) - reference).abs() / ulp
}

/// Deterministic xorshift64 generator; reproducible across runs without a
/// dependency.
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.0 = state;
        state
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
