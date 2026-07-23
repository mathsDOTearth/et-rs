/*
 * Aggregated header consumed by bindgen to generate the et-trace buffer layout.
 *
 * Only the passive layout definitions are pulled in. The decoder itself
 * (et-trace/decoder.h under ET_TRACE_DECODER_IMPL) is intentionally omitted and
 * reimplemented in safe Rust in the `trace` module.
 */
#include <esperanto/et-trace/layout.h>
