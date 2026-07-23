/*
 * C ABI shim over the Esperanto device-layer's software-emulator backend.
 *
 * This exposes just enough of dev::IDeviceLayer (the SysEmu implementation) for
 * the Rust `FfiTransport` to drive an emulated ET-SoC-1 device with exactly the
 * same command wire format the crate sends to real hardware. All C++ types
 * (unique_ptr, std::vector, exceptions) are confined behind this boundary.
 */
#ifndef ET_EMU_SHIM_H
#define ET_EMU_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to an emulated device layer bound to device index 0. */
typedef struct et_emu_dev et_emu_dev;

/*
 * Create a single-device software-emulator layer. `sdk_prefix` is the SDK
 * install root (e.g. "/opt/et"); firmware ELFs and the sys_emu binary are
 * resolved beneath it. `run_dir` is a writable directory for emulator logs.
 * On failure returns NULL and writes a message into `errbuf`.
 */
et_emu_dev *et_emu_create(const char *sdk_prefix, const char *run_dir,
                          char *errbuf, size_t errlen);

void et_emu_destroy(et_emu_dev *dev);

/* Message of the most recent failure on the calling thread (empty if none). */
const char *et_emu_last_error(void);

uint64_t et_emu_dram_base(et_emu_dev *dev);
uint64_t et_emu_dram_size(et_emu_dev *dev);
uint64_t et_emu_dma_max_elem_size(et_emu_dev *dev);
uint32_t et_emu_dma_max_elem_count(et_emu_dev *dev);
int et_emu_dma_alignment_bits(et_emu_dev *dev);
uint32_t et_emu_sq_count(et_emu_dev *dev);
uint32_t et_emu_sq_max_msg(et_emu_dev *dev);

/* Push a command. Returns 1 if sent, 0 if the queue was full, -1 on error. */
int et_emu_push_sq(et_emu_dev *dev, uint16_t sq_index, const uint8_t *cmd,
                   size_t size, uint8_t is_dma);

/*
 * Pop one response into `out` (capacity `out_cap`). Returns 1 and sets
 * `*out_len` if a response was received, 0 if none was available, -1 on error
 * (including a response larger than `out_cap`).
 */
int et_emu_pop_cq(et_emu_dev *dev, uint8_t *out, size_t out_cap,
                  size_t *out_len);

/* Block until the completion (1) or submission (via sq variant) queue is ready,
 * or `timeout_ms` elapses. Returns 1 if ready, 0 on timeout, -1 on error. */
int et_emu_wait_cq(et_emu_dev *dev, uint32_t timeout_ms);
int et_emu_wait_sq(et_emu_dev *dev, uint32_t timeout_ms);

/* Allocate/free a DMA-capable host buffer. The returned pointer is valid both
 * for host access and as the DMA endpoint address the emulator dereferences. */
void *et_emu_alloc_dma(et_emu_dev *dev, size_t size, int writeable);
void et_emu_free_dma(et_emu_dev *dev, void *buf);

/* Firmware image update. Returns bytes written, or -1 on error. */
long et_emu_fw_update(et_emu_dev *dev, const uint8_t *img, size_t size);

/* Firmware trace buffer size / extraction for a TraceBufferType. */
long et_emu_trace_size(et_emu_dev *dev, uint8_t trace_type);
long et_emu_extract_trace(et_emu_dev *dev, uint8_t trace_type, uint8_t *out,
                          size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* ET_EMU_SHIM_H */
