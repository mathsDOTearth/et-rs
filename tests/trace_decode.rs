//! Integration tests for the pure-Rust et-trace decoder, driven against
//! synthetic buffers assembled to the layout in `et-trace/layout.h`.

use et_soc1::trace::{TraceBuffer, TraceError, TraceType};

const MAGIC: u32 = 0x7654_3210;
const STD_HEADER: usize = 64;

/// Encode a single string entry (16-byte header + NUL-terminated, 8-aligned body).
fn push_string_entry(buf: &mut Vec<u8>, cycle: u64, hart: u16, text: &str) {
    let mut payload = text.as_bytes().to_vec();
    payload.push(0);
    while !payload.len().is_multiple_of(8) {
        payload.push(0);
    }
    buf.extend_from_slice(&cycle.to_le_bytes());
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(&hart.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // TRACE_TYPE_STRING
    buf.extend_from_slice(&payload);
}

fn write_std_header(
    buf: &mut [u8],
    buffer_type: u16,
    data_size: u32,
    sub_size: u32,
    sub_count: u16,
) {
    buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    buf[4..6].copy_from_slice(&0u16.to_le_bytes()); // major
    buf[6..8].copy_from_slice(&6u16.to_le_bytes()); // minor
    buf[8..10].copy_from_slice(&0u16.to_le_bytes()); // patch
    buf[10..12].copy_from_slice(&buffer_type.to_le_bytes());
    buf[12..16].copy_from_slice(&data_size.to_le_bytes());
    buf[16..20].copy_from_slice(&sub_size.to_le_bytes());
    buf[20..22].copy_from_slice(&sub_count.to_le_bytes());
}

/// Build a single-partition trace buffer holding the given string entries.
fn build_single(entries: &[(u64, u16, &str)]) -> Vec<u8> {
    let mut buf = vec![0u8; STD_HEADER];
    for &(cycle, hart, text) in entries {
        push_string_entry(&mut buf, cycle, hart, text);
    }
    let data_size = buf.len() as u32;
    write_std_header(&mut buf, 1 /* CM */, data_size, 0, 0);
    buf
}

#[test]
fn decodes_string_entries_in_order() {
    let inputs = [
        (10u64, 0u16, "Hello World from hart 0"),
        (20, 1, "Hello World from hart 1"),
        (
            30,
            42,
            "a longer line that crosses the eight byte alignment boundary",
        ),
    ];
    let buf = build_single(&inputs);
    let tb = TraceBuffer::parse(&buf).expect("valid header");
    assert_eq!(tb.header().buffer_type, 1);

    let decoded: Vec<_> = tb
        .entries()
        .map(|e| {
            (
                e.cycle,
                e.hart_id,
                e.trace_type(),
                e.as_str().unwrap().into_owned(),
            )
        })
        .collect();

    assert_eq!(decoded.len(), inputs.len());
    for (got, want) in decoded.iter().zip(inputs.iter()) {
        assert_eq!(got.0, want.0);
        assert_eq!(got.1, want.1);
        assert_eq!(got.2, TraceType::String);
        assert_eq!(got.3, want.2);
    }
}

#[test]
fn empty_buffer_yields_no_entries() {
    let buf = build_single(&[]);
    let tb = TraceBuffer::parse(&buf).expect("valid header");
    assert_eq!(tb.entries().count(), 0);
}

#[test]
fn rejects_bad_magic() {
    let mut buf = build_single(&[(1, 0, "x")]);
    buf[0] ^= 0xFF;
    assert!(matches!(
        TraceBuffer::parse(&buf),
        Err(TraceError::BadMagic(_))
    ));
}

#[test]
fn rejects_short_buffer() {
    assert!(matches!(
        TraceBuffer::parse(&[0u8; 8]),
        Err(TraceError::TooShort)
    ));
}

#[test]
fn truncated_payload_terminates_cleanly() {
    // A buffer whose header claims more data than is present must not panic or
    // read out of bounds; iteration simply stops.
    let mut buf = build_single(&[(1, 0, "abc"), (2, 0, "def")]);
    let real_len = buf.len();
    // Claim 32 extra bytes of data that do not exist.
    write_std_header(&mut buf, 1, (real_len + 32) as u32, 0, 0);
    let tb = TraceBuffer::parse(&buf).expect("valid header");
    // Both real entries decode; the phantom tail is ignored.
    assert_eq!(tb.entries().count(), 2);
}

#[test]
fn decodes_across_sub_buffers() {
    let stride = 256usize;
    let count = 3u16;
    let mut buf = vec![0u8; stride * count as usize];

    // Partition 0: standard header then two entries.
    let mut part0 = vec![0u8; STD_HEADER];
    push_string_entry(&mut part0, 1, 0, "p0-a");
    push_string_entry(&mut part0, 2, 0, "p0-b");
    let part0_data = part0.len() as u32; // absolute end of partition-0 data
    buf[..part0.len()].copy_from_slice(&part0);
    write_std_header(&mut buf, 1, part0_data, stride as u32, count);

    // Partition 1: 4-byte size header then one entry.
    {
        let base = stride;
        let mut body = vec![0u8; 4];
        push_string_entry(&mut body, 3, 1, "p1-a");
        let sub_size = body.len() as u32; // includes the 4-byte size header
        body[0..4].copy_from_slice(&sub_size.to_le_bytes());
        buf[base..base + body.len()].copy_from_slice(&body);
    }

    // Partition 2: left empty (size header data_size stays 0), must be skipped.

    let tb = TraceBuffer::parse(&buf).expect("valid header");
    let strings: Vec<String> = tb
        .entries()
        .filter_map(|e| e.as_str().map(|s| s.into_owned()))
        .collect();
    assert_eq!(strings, vec!["p0-a", "p0-b", "p1-a"]);
}
