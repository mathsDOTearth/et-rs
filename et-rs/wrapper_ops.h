/*
 * Aggregated header consumed by bindgen to generate the driver/device-ops ABI.
 *
 * This translation unit deliberately excludes <esperanto/et-trace/layout.h>:
 * both et_ioctl.h and the trace layout header define an `enum trace_buffer_type`
 * with divergent enumerators, so pulling them into a single translation unit
 * would be a C redefinition error. The trace layout is generated separately via
 * wrapper_trace.h.
 */
#include <et_ioctl.h>
#include <esperanto/device-apis/device_apis_message_types.h>
#include <esperanto/device-apis/operations-api/device_ops_api_spec.h>
#include <esperanto/device-apis/operations-api/device_ops_api_rpc_types.h>
