//! Raw FFI bindings for the SDK ABI, vendored into the source tree.
//!
//! Two independent binding sets are used: [`ops`] covers the character-device
//! uapi (`et_ioctl.h`) together with the device-ops RPC message ABI, and
//! [`trace`] covers the et-trace buffer layout. They are kept in separate
//! modules because the two SDK headers define conflicting `enum trace_buffer_type`
//! tags.
//!
//! The bindings are committed as `src/bindings_ops.rs` / `src/bindings_trace.rs`
//! so the crate builds without the SDK. Regenerate them from the headers with
//! `cargo build --features regenerate-bindings` (see `build.rs`).
//!
//! Everything here is `unsafe` to use directly; the safe wrappers live in the
//! parent modules.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

/// Driver uapi (`et_ioctl.h`) and device-ops RPC message ABI.
pub mod ops {
    include!("bindings_ops.rs");
}

/// et-trace buffer layout (`esperanto/et-trace/layout.h`).
pub mod trace {
    include!("bindings_trace.rs");
}
