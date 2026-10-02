//! Inter-Minion channel latency kernel for the ET-SoC-1.
//!
//! Measures the round-trip time of a message exchanged between two Minions
//! through device memory, using only the software coherence operations
//! available to a kernel: `flush_va` (write back) on the producer and
//! `evict_va` (invalidate) on the consumer, both to a chosen cache level.
//!
//! The primary hart of the ping Minion runs, for each iteration `i`:
//!
//! 1. writes the payload to the forward data area and writes it back;
//! 2. writes the forward flag `(epoch << 32) | i` and writes it back;
//! 3. polls the reply flag, invalidating its line before every read, until it
//!    holds the same value;
//! 4. invalidates and reads the reply payload, verifying every word.
//!
//! The round-trip time (step 1 to step 4 inclusive) and the send time (steps
//! 1 and 2) are recorded in cycles. The primary hart of the pong Minion
//! mirrors the protocol: it waits for the forward flag, invalidates and reads
//! the payload, writes the complement of each word to the reply data area,
//! writes it back and raises the reply flag. Every other hart returns at once.
//!
//! Each flag and each payload occupies whole cache lines written by one
//! Minion only, so that invalidating a line never discards another Minion's
//! dirty data. Before returning, each Minion writes its lines back to DDR, so
//! that no dirty line of the buffer remains in L2 or L3 after the launch.

#![no_std]
#![no_main]

use core::ptr::{read_volatile, write_volatile};

use et_abi::{
    CACHE_LINE, CHANNEL_FORWARD_DATA_OFFSET, CHANNEL_FORWARD_FLAG_OFFSET, CHANNEL_HEADER_COMPLETED,
    CHANNEL_HEADER_ERRORS, CHANNEL_HEADER_FAILED_ITERATION, CHANNEL_HEADER_STATUS,
    CHANNEL_PING_HEADER_OFFSET, CHANNEL_PONG_HEADER_OFFSET, CHANNEL_REPLY_FLAG_OFFSET,
    CHANNEL_SAMPLE_WORDS, CHANNEL_SAMPLES_OFFSET, CHANNEL_STATUS_BAD_ARGUMENTS, CHANNEL_STATUS_OK,
    CHANNEL_STATUS_TIMEOUT, ChannelBenchArgs, DeviceArgs, MINIONS_PER_SHIRE,
    channel_reply_data_offset, channel_results_bytes,
};
use et_kernel::cache::{CacheDest, cache_invalidate_to, cache_writeback, cache_writeback_to};
use et_kernel::{hart_id, kernel_entry, timestamp};

kernel_entry!();

/// Bytes per payload word.
const WORD_BYTES: usize = size_of::<u64>();
/// Upper bound on [`ChannelBenchArgs::iterations`], so that an iteration
/// number fits its field of the payload word.
const MAX_ITERATIONS: u32 = 1 << 20;
/// Constant mixed into every payload word, so that a zeroed buffer never
/// matches.
const PAYLOAD_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Outcome of one Minion's part in the exchange, written to its header.
struct Outcome {
    status: u64,
    errors: u64,
    completed: u64,
    failed_iteration: u64,
}

/// One direction of transmission and one of reception, as seen by a Minion.
struct Endpoint {
    destination: CacheDest,
    send_flag: usize,
    send_data: usize,
    receive_flag: usize,
    receive_data: usize,
    payload_words: usize,
    epoch: u64,
    timeout_cycles: u64,
}

/// Expected value of payload word `index` of forward message `iteration`.
#[inline(always)]
fn payload_word(epoch: u64, iteration: u64, index: u64) -> u64 {
    ((epoch << 40) ^ (iteration << 20) ^ index) ^ PAYLOAD_SALT
}

impl Endpoint {
    /// Flag value announcing message `iteration`.
    #[inline(always)]
    fn flag_value(&self, iteration: u64) -> u64 {
        (self.epoch << 32) | iteration
    }

    /// Payload length in bytes.
    #[inline(always)]
    fn payload_bytes(&self) -> usize {
        self.payload_words * WORD_BYTES
    }

    /// Writes back the payload already stored in the send area, then writes
    /// and writes back the send flag for `iteration`.
    ///
    /// # Safety
    /// The send areas must lie within the channel buffer and be written by
    /// this Minion only.
    #[inline(always)]
    unsafe fn publish(&self, iteration: u64) {
        unsafe {
            if self.payload_words != 0 {
                // Completes before the flag store below is issued, so the
                // payload reaches the destination level first.
                cache_writeback_to(self.destination, self.send_data, self.payload_bytes());
            }
            write_volatile(self.send_flag as *mut u64, self.flag_value(iteration));
            cache_writeback_to(self.destination, self.send_flag, CACHE_LINE);
        }
    }

    /// Polls the receive flag until it announces `iteration`; returns `false`
    /// if [`Endpoint::timeout_cycles`] elapse first.
    ///
    /// # Safety
    /// The receive flag must lie within the channel buffer and must not be
    /// written by this Minion.
    #[inline(always)]
    unsafe fn wait_for(&self, iteration: u64) -> bool {
        let expected = self.flag_value(iteration);
        let start = timestamp();
        loop {
            // SAFETY: the line is never dirty in this Minion's cache, so the
            // invalidation discards no data; see the function contract.
            unsafe {
                cache_invalidate_to(self.destination, self.receive_flag, CACHE_LINE);
                if read_volatile(self.receive_flag as *const u64) == expected {
                    return true;
                }
            }
            if timestamp().wrapping_sub(start) > self.timeout_cycles {
                return false;
            }
        }
    }

    /// Invalidates the receive payload so that the following reads fetch the
    /// producer's data from the destination level.
    ///
    /// # Safety
    /// As for [`Endpoint::wait_for`], applied to the receive payload.
    #[inline(always)]
    unsafe fn invalidate_received(&self) {
        if self.payload_words != 0 {
            unsafe {
                cache_invalidate_to(self.destination, self.receive_data, self.payload_bytes())
            };
        }
    }

    /// Writes this Minion's flag and payload lines back to DDR, leaving no
    /// dirty line of the buffer in any cache.
    ///
    /// # Safety
    /// As for [`Endpoint::publish`].
    unsafe fn retire(&self) {
        unsafe {
            cache_writeback(self.send_data, self.payload_bytes());
            cache_writeback(self.send_flag, CACHE_LINE);
        }
    }
}

/// Ping side: sends each message, awaits the reply and records the timings.
///
/// # Safety
/// `endpoint` must describe the forward direction for sending; `samples`
/// must address `iterations` samples of the results area.
unsafe fn run_ping(endpoint: &Endpoint, iterations: u64, samples: usize) -> Outcome {
    let mut outcome = Outcome {
        status: CHANNEL_STATUS_OK,
        errors: 0,
        completed: 0,
        failed_iteration: 0,
    };
    let send = endpoint.send_data as *mut u64;
    let receive = endpoint.receive_data as *const u64;
    for iteration in 1..=iterations {
        let start = timestamp();
        // SAFETY: indices are below payload_words, within the send and
        // receive payload areas; see the function contract.
        unsafe {
            for index in 0..endpoint.payload_words {
                write_volatile(
                    send.add(index),
                    payload_word(endpoint.epoch, iteration, index as u64),
                );
            }
            endpoint.publish(iteration);
        }
        let sent = timestamp();
        if !unsafe { endpoint.wait_for(iteration) } {
            outcome.status = CHANNEL_STATUS_TIMEOUT;
            outcome.failed_iteration = iteration;
            break;
        }
        unsafe {
            endpoint.invalidate_received();
            for index in 0..endpoint.payload_words {
                let expected = !payload_word(endpoint.epoch, iteration, index as u64);
                if read_volatile(receive.add(index)) != expected {
                    outcome.errors += 1;
                }
            }
        }
        let finish = timestamp();

        let sample = (samples
            + (iteration as usize - 1) * CHANNEL_SAMPLE_WORDS as usize * WORD_BYTES)
            as *mut u64;
        // SAFETY: sample `iteration - 1` lies within the results area.
        unsafe {
            write_volatile(sample, finish.wrapping_sub(start));
            write_volatile(sample.add(1), sent.wrapping_sub(start));
        }
        outcome.completed = iteration;
    }
    outcome
}

/// Pong side: awaits each message, verifies it and replies with its
/// complement.
///
/// # Safety
/// `endpoint` must describe the reply direction for sending.
unsafe fn run_pong(endpoint: &Endpoint, iterations: u64) -> Outcome {
    let mut outcome = Outcome {
        status: CHANNEL_STATUS_OK,
        errors: 0,
        completed: 0,
        failed_iteration: 0,
    };
    let send = endpoint.send_data as *mut u64;
    let receive = endpoint.receive_data as *const u64;
    for iteration in 1..=iterations {
        if !unsafe { endpoint.wait_for(iteration) } {
            outcome.status = CHANNEL_STATUS_TIMEOUT;
            outcome.failed_iteration = iteration;
            break;
        }
        // SAFETY: indices are below payload_words, within the payload areas.
        unsafe {
            endpoint.invalidate_received();
            for index in 0..endpoint.payload_words {
                let value = read_volatile(receive.add(index));
                if value != payload_word(endpoint.epoch, iteration, index as u64) {
                    outcome.errors += 1;
                }
                write_volatile(send.add(index), !value);
            }
            endpoint.publish(iteration);
        }
        outcome.completed = iteration;
    }
    outcome
}

/// Writes `outcome` to the header line at `header` and writes it back to DDR.
///
/// # Safety
/// `header` must address a cache line of the results area written by this
/// Minion only.
unsafe fn write_header(header: usize, outcome: &Outcome) {
    let words = header as *mut u64;
    unsafe {
        write_volatile(words.add(CHANNEL_HEADER_STATUS), outcome.status);
        write_volatile(words.add(CHANNEL_HEADER_ERRORS), outcome.errors);
        write_volatile(words.add(CHANNEL_HEADER_COMPLETED), outcome.completed);
        write_volatile(
            words.add(CHANNEL_HEADER_FAILED_ITERATION),
            outcome.failed_iteration,
        );
        cache_writeback(header, CACHE_LINE);
    }
}

/// Maps [`ChannelBenchArgs::cache_level`] to a destination, rejecting L2 for
/// Minions in different shires (which share no L2).
fn destination(args: &ChannelBenchArgs) -> Option<CacheDest> {
    match args.cache_level {
        1 if args.ping_shire == args.pong_shire => Some(CacheDest::L2),
        2 => Some(CacheDest::L3),
        3 => Some(CacheDest::Mem),
        _ => None,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // SAFETY: firmware staged a valid ChannelBenchArgs at args_ptr before launch.
    let args: &ChannelBenchArgs = unsafe { ChannelBenchArgs::from_ptr(args_ptr as *const u8) };

    let hart = hart_id();
    if hart & 1 != 0 {
        return 0;
    }
    let shire = hart >> 6;
    let minion = (hart & 63) >> 1;
    let is_ping = shire == args.ping_shire && minion == args.ping_minion;
    let is_pong = shire == args.pong_shire && minion == args.pong_minion;
    if !is_ping && !is_pong {
        return 0;
    }

    let results = args.results as usize;
    let header = results
        + if is_ping {
            CHANNEL_PING_HEADER_OFFSET
        } else {
            CHANNEL_PONG_HEADER_OFFSET
        } as usize;

    let valid_arguments = !(is_ping && is_pong)
        && args.ping_minion < MINIONS_PER_SHIRE
        && args.pong_minion < MINIONS_PER_SHIRE
        && args.message_bytes.is_multiple_of(WORD_BYTES as u64)
        && args.iterations < MAX_ITERATIONS
        && args.buffer.is_multiple_of(CACHE_LINE as u64)
        && args.results.is_multiple_of(CACHE_LINE as u64);
    let destination = match destination(args) {
        Some(destination) if valid_arguments => destination,
        _ => {
            let outcome = Outcome {
                status: CHANNEL_STATUS_BAD_ARGUMENTS,
                errors: 0,
                completed: 0,
                failed_iteration: 0,
            };
            // SAFETY: the header line lies within the results area and is
            // written by this Minion only.
            unsafe { write_header(header, &outcome) };
            return 0;
        }
    };

    let buffer = args.buffer as usize;
    let forward_flag = buffer + CHANNEL_FORWARD_FLAG_OFFSET as usize;
    let reply_flag = buffer + CHANNEL_REPLY_FLAG_OFFSET as usize;
    let forward_data = buffer + CHANNEL_FORWARD_DATA_OFFSET as usize;
    let reply_data = buffer + channel_reply_data_offset(args.message_bytes) as usize;
    let (send_flag, send_data, receive_flag, receive_data) = if is_ping {
        (forward_flag, forward_data, reply_flag, reply_data)
    } else {
        (reply_flag, reply_data, forward_flag, forward_data)
    };
    let endpoint = Endpoint {
        destination,
        send_flag,
        send_data,
        receive_flag,
        receive_data,
        payload_words: args.message_bytes as usize / WORD_BYTES,
        epoch: u64::from(args.epoch),
        timeout_cycles: args.timeout_cycles,
    };
    let iterations = u64::from(args.iterations);

    // SAFETY: the host allocated the buffer and results areas with the sizes
    // given by channel_buffer_bytes and channel_results_bytes; each area
    // written here is written by this Minion alone.
    unsafe {
        let outcome = if is_ping {
            let samples = results + CHANNEL_SAMPLES_OFFSET as usize;
            let outcome = run_ping(&endpoint, iterations, samples);
            let sample_bytes =
                (channel_results_bytes(args.iterations) - CHANNEL_SAMPLES_OFFSET) as usize;
            cache_writeback(samples, sample_bytes);
            outcome
        } else {
            run_pong(&endpoint, iterations)
        };
        endpoint.retire();
        write_header(header, &outcome);
    }

    0
}

// ---------------------------------------------------------------------------
// Minimal runtime support
// ---------------------------------------------------------------------------

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        unsafe { core::arch::asm!("wfi") };
    }
}
