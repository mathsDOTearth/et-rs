//! Pure-Rust decoder for the et-trace buffer layout.
//!
//! This is a safe reimplementation of the reference C `Trace_Decode` iterator
//! (`esperanto/et-trace/decoder.h`) over the layout in
//! `esperanto/et-trace/layout.h`. It parses a linear trace buffer, transparently
//! walking sub-buffer partitions, and yields typed entries. Unlike the C
//! decoder it never dereferences unvalidated pointers: every field access is
//! bounds-checked against the buffer, so a malformed or truncated buffer
//! terminates iteration cleanly rather than reading out of bounds.
//!
//! ```no_run
//! # fn decode(bytes: &[u8]) {
//! use et_soc1::trace::{TraceBuffer, DecodedEntry};
//! let tb = TraceBuffer::parse(bytes).expect("valid trace header");
//! for entry in tb.entries() {
//!     if let DecodedEntry::String(s) = entry.decoded() {
//!         println!("[hart {}] {}", entry.hart_id, s);
//!     }
//! }
//! # }
//! ```

use crate::ffi::trace as ffi;
use std::borrow::Cow;

/// et-trace magic header (`TRACE_MAGIC_HEADER`).
pub const MAGIC_HEADER: u32 = ffi::TRACE_MAGIC_HEADER;
/// Trace layout major version implemented by this decoder.
pub const VERSION_MAJOR: u16 = ffi::TRACE_VERSION_MAJOR as u16;
/// Trace layout minor version implemented by this decoder.
pub const VERSION_MINOR: u16 = ffi::TRACE_VERSION_MINOR as u16;

/// Size of the standard buffer header (`trace_buffer_std_header_t`). The struct
/// is `aligned(64)`, so the first entry begins 64 bytes into the buffer.
const STD_HEADER_SIZE: usize = 64;
/// Size of a per-entry header (`trace_entry_header_t`).
const ENTRY_HEADER_SIZE: usize = 16;
/// Size of a sub-buffer size header (`trace_buffer_size_header_t`).
const SIZE_HEADER_SIZE: usize = 4;

// Cross-check the hand-encoded offsets against the ABI bindgen observed.
const _: () = assert!(core::mem::size_of::<ffi::trace_buffer_std_header_t>() == STD_HEADER_SIZE);
const _: () = assert!(core::mem::size_of::<ffi::trace_entry_header_t>() == ENTRY_HEADER_SIZE);
const _: () = assert!(core::mem::size_of::<ffi::trace_buffer_size_header_t>() == SIZE_HEADER_SIZE);

/// Errors returned when parsing a trace buffer header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceError {
    /// The buffer is shorter than the standard header.
    TooShort,
    /// The magic header did not match [`MAGIC_HEADER`].
    BadMagic(u32),
    /// The buffer's layout version is not supported by this decoder.
    IncompatibleVersion {
        /// Major version found in the buffer.
        major: u16,
        /// Minor version found in the buffer.
        minor: u16,
    },
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceError::TooShort => write!(f, "trace buffer shorter than its header"),
            TraceError::BadMagic(m) => write!(f, "bad trace magic 0x{m:08x}"),
            TraceError::IncompatibleVersion { major, minor } => {
                write!(f, "incompatible trace layout version {major}.{minor}")
            }
        }
    }
}

impl std::error::Error for TraceError {}

/// The decoded standard header of a trace buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Layout version (major, minor, patch).
    pub version: (u16, u16, u16),
    /// Buffer type (`trace_buffer_type`: MM, CM, SP, ...).
    pub buffer_type: u16,
    /// Bytes of valid data in the primary partition, measured from the buffer
    /// start (thus inclusive of the 64-byte header).
    pub data_size: u32,
    /// Size of each sub-buffer partition, in bytes.
    pub sub_buffer_size: u32,
    /// Number of sub-buffer partitions.
    pub sub_buffer_count: u16,
}

/// A parsed, borrowed trace buffer ready for iteration.
#[derive(Clone, Copy, Debug)]
pub struct TraceBuffer<'a> {
    data: &'a [u8],
    header: Header,
}

impl<'a> TraceBuffer<'a> {
    /// Validate the standard header and wrap the buffer for decoding.
    pub fn parse(data: &'a [u8]) -> Result<Self, TraceError> {
        if data.len() < STD_HEADER_SIZE {
            return Err(TraceError::TooShort);
        }
        let magic = rd_u32(data, 0);
        if magic != MAGIC_HEADER {
            return Err(TraceError::BadMagic(magic));
        }
        let major = rd_u16(data, 4);
        let minor = rd_u16(data, 6);
        let patch = rd_u16(data, 8);
        // Semantic-versioning compatibility: identical major, and the buffer's
        // minor no newer than the decoder's.
        if major != VERSION_MAJOR || minor > VERSION_MINOR {
            return Err(TraceError::IncompatibleVersion { major, minor });
        }
        let header = Header {
            version: (major, minor, patch),
            buffer_type: rd_u16(data, 10),
            data_size: rd_u32(data, 12),
            sub_buffer_size: rd_u32(data, 16),
            sub_buffer_count: rd_u16(data, 20),
        };
        Ok(TraceBuffer { data, header })
    }

    /// The decoded standard header.
    pub fn header(&self) -> Header {
        self.header
    }

    /// Iterate the entries in the buffer, in stream order across all populated
    /// sub-buffer partitions.
    pub fn entries(&self) -> Entries<'a> {
        Entries {
            data: self.data,
            segments: self.segments(),
            segment: 0,
            offset: 0,
            started: false,
        }
    }

    /// Compute the `[first_entry, end)` byte ranges of every populated partition.
    ///
    /// Partition 0 spans `[64, data_size)`; further partitions each begin with a
    /// 4-byte size header at `i * sub_buffer_size` and span `[+4, +size)`.
    /// Empty partitions are omitted, matching `decode_next_valid_sub_buffer`.
    fn segments(&self) -> Vec<Segment> {
        let len = self.data.len();
        let mut segs = Vec::new();

        let data_end = (self.header.data_size as usize).min(len);
        if data_end > STD_HEADER_SIZE {
            segs.push(Segment {
                start: STD_HEADER_SIZE,
                end: data_end,
            });
        }

        let count = self.header.sub_buffer_count as usize;
        let stride = self.header.sub_buffer_size as usize;
        if count > 1 && stride >= SIZE_HEADER_SIZE {
            for i in 1..count {
                let base = i.saturating_mul(stride);
                if base + SIZE_HEADER_SIZE > len {
                    break;
                }
                let sub_size = rd_u32(self.data, base) as usize;
                if sub_size > SIZE_HEADER_SIZE {
                    let end = base.saturating_add(sub_size).min(len);
                    segs.push(Segment {
                        start: base + SIZE_HEADER_SIZE,
                        end,
                    });
                }
            }
        }
        segs
    }
}

/// A `[first_entry, end)` byte range within one populated partition.
#[derive(Clone, Copy, Debug)]
struct Segment {
    start: usize,
    end: usize,
}

/// Iterator over the entries of a [`TraceBuffer`].
#[derive(Clone, Debug)]
pub struct Entries<'a> {
    data: &'a [u8],
    segments: Vec<Segment>,
    segment: usize,
    offset: usize,
    started: bool,
}

impl<'a> Iterator for Entries<'a> {
    type Item = Entry<'a>;

    fn next(&mut self) -> Option<Entry<'a>> {
        loop {
            let seg = *self.segments.get(self.segment)?;
            if !self.started {
                self.offset = seg.start;
                self.started = true;
            }
            // Exhausted this partition: advance to the next populated one.
            if self.offset + ENTRY_HEADER_SIZE > seg.end {
                self.segment += 1;
                self.started = false;
                continue;
            }

            let off = self.offset;
            let cycle = rd_u64(self.data, off);
            let payload_size = rd_u32(self.data, off + 8) as usize;
            let hart_id = rd_u16(self.data, off + 12);
            let raw_type = rd_u16(self.data, off + 14);

            let payload_start = off + ENTRY_HEADER_SIZE;
            let payload_end = payload_start.saturating_add(payload_size);
            // A payload running past the partition or buffer marks corruption;
            // terminate rather than fabricate an entry.
            if payload_end > seg.end || payload_end > self.data.len() {
                return None;
            }

            self.offset = payload_end;
            return Some(Entry {
                cycle,
                hart_id,
                raw_type,
                payload: &self.data[payload_start..payload_end],
            });
        }
    }
}

/// A single decoded trace entry with its common header fields and raw payload.
#[derive(Clone, Copy, Debug)]
pub struct Entry<'a> {
    /// Device cycle counter at which the entry was recorded.
    pub cycle: u64,
    /// Hart that produced the entry (when applicable).
    pub hart_id: u16,
    /// Raw entry type (`trace_type`).
    pub raw_type: u16,
    /// Entry payload following the 16-byte common header.
    pub payload: &'a [u8],
}

impl<'a> Entry<'a> {
    /// The entry type as a typed enum.
    pub fn trace_type(&self) -> TraceType {
        TraceType::from_raw(self.raw_type)
    }

    /// If this is a string entry, borrow it as text (up to the first NUL,
    /// lossily decoded as UTF-8).
    pub fn as_str(&self) -> Option<Cow<'a, str>> {
        if self.trace_type() != TraceType::String {
            return None;
        }
        Some(decode_cstr(self.payload))
    }

    /// Decode the payload into a typed [`DecodedEntry`].
    pub fn decoded(&self) -> DecodedEntry<'a> {
        match self.trace_type() {
            TraceType::String => DecodedEntry::String(decode_cstr(self.payload)),
            TraceType::ValueU64 => scalar(self.payload, 8)
                .map(|(tag, v)| DecodedEntry::ValueU64 { tag, value: v })
                .unwrap_or(DecodedEntry::Malformed),
            TraceType::ValueU32 => scalar(self.payload, 4)
                .map(|(tag, v)| DecodedEntry::ValueU32 {
                    tag,
                    value: v as u32,
                })
                .unwrap_or(DecodedEntry::Malformed),
            TraceType::ValueU16 => scalar(self.payload, 2)
                .map(|(tag, v)| DecodedEntry::ValueU16 {
                    tag,
                    value: v as u16,
                })
                .unwrap_or(DecodedEntry::Malformed),
            TraceType::ValueU8 => scalar(self.payload, 1)
                .map(|(tag, v)| DecodedEntry::ValueU8 {
                    tag,
                    value: v as u8,
                })
                .unwrap_or(DecodedEntry::Malformed),
            _ => DecodedEntry::Other {
                trace_type: self.trace_type(),
                payload: self.payload,
            },
        }
    }
}

/// A typed decoding of an [`Entry`]'s payload.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodedEntry<'a> {
    /// A textual kernel print (`TRACE_TYPE_STRING`).
    String(Cow<'a, str>),
    /// A tagged 64-bit value (`TRACE_TYPE_VALUE_U64`).
    ValueU64 {
        /// User tag associated with the value.
        tag: u32,
        /// The logged value.
        value: u64,
    },
    /// A tagged 32-bit value.
    ValueU32 { tag: u32, value: u32 },
    /// A tagged 16-bit value.
    ValueU16 { tag: u32, value: u16 },
    /// A tagged 8-bit value.
    ValueU8 { tag: u32, value: u8 },
    /// Any other entry type, left as raw payload bytes.
    Other {
        /// The entry type.
        trace_type: TraceType,
        /// The undecoded payload.
        payload: &'a [u8],
    },
    /// The payload was too short for its declared type.
    Malformed,
}

/// Entry payload types (`enum trace_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceType {
    String,
    PmcCounter,
    PmcCountersCompute,
    PmcCountersMemory,
    PmcCountersSc,
    PmcCountersMs,
    ValueU64,
    ValueU32,
    ValueU16,
    ValueU8,
    ValueFloat,
    Memory,
    Exception,
    CmdStatus,
    PowerStatus,
    CustomEvent,
    UserProfileEvent,
    /// A type this decoder does not model, carrying its raw discriminant.
    Unknown(u16),
}

impl TraceType {
    /// Map a raw `trace_type` discriminant to a typed variant.
    pub fn from_raw(raw: u16) -> Self {
        use ffi::trace_type::*;
        match raw as ffi::trace_type::Type {
            TRACE_TYPE_STRING => TraceType::String,
            TRACE_TYPE_PMC_COUNTER => TraceType::PmcCounter,
            TRACE_TYPE_PMC_COUNTERS_COMPUTE => TraceType::PmcCountersCompute,
            TRACE_TYPE_PMC_COUNTERS_MEMORY => TraceType::PmcCountersMemory,
            TRACE_TYPE_PMC_COUNTERS_SC => TraceType::PmcCountersSc,
            TRACE_TYPE_PMC_COUNTERS_MS => TraceType::PmcCountersMs,
            TRACE_TYPE_VALUE_U64 => TraceType::ValueU64,
            TRACE_TYPE_VALUE_U32 => TraceType::ValueU32,
            TRACE_TYPE_VALUE_U16 => TraceType::ValueU16,
            TRACE_TYPE_VALUE_U8 => TraceType::ValueU8,
            TRACE_TYPE_VALUE_FLOAT => TraceType::ValueFloat,
            TRACE_TYPE_MEMORY => TraceType::Memory,
            TRACE_TYPE_EXCEPTION => TraceType::Exception,
            TRACE_TYPE_CMD_STATUS => TraceType::CmdStatus,
            TRACE_TYPE_POWER_STATUS => TraceType::PowerStatus,
            TRACE_TYPE_CUSTOM_EVENT => TraceType::CustomEvent,
            TRACE_TYPE_USER_PROFILE_EVENT => TraceType::UserProfileEvent,
            _ => TraceType::Unknown(raw),
        }
    }
}

/// Decode a scalar value entry: a 4-byte tag followed by the value.
fn scalar(payload: &[u8], value_bytes: usize) -> Option<(u32, u64)> {
    if payload.len() < 4 + value_bytes {
        return None;
    }
    let tag = rd_u32(payload, 0);
    let mut v = [0u8; 8];
    v[..value_bytes].copy_from_slice(&payload[4..4 + value_bytes]);
    Some((tag, u64::from_le_bytes(v)))
}

/// Interpret a payload as a NUL-terminated C string, lossily as UTF-8.
fn decode_cstr(payload: &[u8]) -> Cow<'_, str> {
    let end = payload
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(payload.len());
    String::from_utf8_lossy(&payload[..end])
}

// Little-endian field readers. Callers guarantee the offsets are in bounds; the
// iterator and parser bounds-check before calling.
#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline]
fn rd_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}
