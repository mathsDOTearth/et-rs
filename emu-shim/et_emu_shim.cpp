// C ABI shim over dev::IDeviceLayer's software-emulator backend. See the header
// for the contract. Everything here runs against device index 0 of a
// single-device SysEmu layer.

#include "et_emu_shim.h"

#include <device-layer/IDeviceLayer.h>
#include <sw-sysemu/SysEmuOptions.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

// Most recent exception message, surfaced to the Rust layer for diagnostics.
thread_local std::string g_last_error;

// The single emulated device index this shim manages.
constexpr int kDevice = 0;

// Enabled compute shires: the low 33 bits, matching the SDK test drive.
constexpr uint64_t kMinionShiresMask = 0x1FFFFFFFFull;

std::string join(const std::string &prefix, const char *rel) {
  return prefix + rel;
}

} // namespace

struct et_emu_dev {
  std::unique_ptr<dev::IDeviceLayer> layer;
};

extern "C" {

et_emu_dev *et_emu_create(const char *sdk_prefix, const char *run_dir,
                          char *errbuf, size_t errlen) {
  try {
    const std::string prefix = sdk_prefix;
    emu::SysEmuOptions opt;
    opt.bootromTrampolineToBL2ElfPath =
        join(prefix, "/lib/esperanto-fw/BootromTrampolineToBL2/"
                     "BootromTrampolineToBL2.elf");
    opt.spBL2ElfPath =
        join(prefix, "/lib/esperanto-fw/ServiceProcessorBL2/fast-boot/"
                     "ServiceProcessorBL2_fast-boot.elf");
    opt.machineMinionElfPath =
        join(prefix, "/lib/esperanto-fw/MachineMinion/MachineMinion.elf");
    opt.masterMinionElfPath =
        join(prefix, "/lib/esperanto-fw/MasterMinion/MasterMinion.elf");
    opt.workerMinionElfPath =
        join(prefix, "/lib/esperanto-fw/WorkerMinion/WorkerMinion.elf");
    opt.executablePath = join(prefix, "/bin/sys_emu");
    opt.runDir = run_dir;
    opt.maxCycles = std::numeric_limits<uint64_t>::max();
    opt.minionShiresMask = kMinionShiresMask;
    const std::string rd = run_dir;
    opt.puUart0Path = rd + "/pu_uart0_tx.log";
    opt.puUart1Path = rd + "/pu_uart1_tx.log";
    opt.spUart0Path = rd + "/spio_uart0_tx.log";
    opt.spUart1Path = rd + "/spio_uart1_tx.log";
    opt.startGdb = false;

    auto dev = std::make_unique<et_emu_dev>();
    dev->layer = dev::IDeviceLayer::createSysEmuDeviceLayer(opt, 1);
    if (!dev->layer) {
      std::snprintf(errbuf, errlen, "createSysEmuDeviceLayer returned null");
      return nullptr;
    }
    return dev.release();
  } catch (const std::exception &e) {
    std::snprintf(errbuf, errlen, "%s", e.what());
    return nullptr;
  } catch (...) {
    std::snprintf(errbuf, errlen, "unknown exception creating device layer");
    return nullptr;
  }
}

void et_emu_destroy(et_emu_dev *dev) { delete dev; }

const char *et_emu_last_error(void) { return g_last_error.c_str(); }

uint64_t et_emu_dram_base(et_emu_dev *dev) {
  return dev->layer->getDramBaseAddress(kDevice);
}

uint64_t et_emu_dram_size(et_emu_dev *dev) {
  return dev->layer->getDramSize(kDevice);
}

uint64_t et_emu_dma_max_elem_size(et_emu_dev *dev) {
  return dev->layer->getDmaInfo(kDevice).maxElementSize_;
}

uint32_t et_emu_dma_max_elem_count(et_emu_dev *dev) {
  return static_cast<uint32_t>(dev->layer->getDmaInfo(kDevice).maxElementCount_);
}

int et_emu_dma_alignment_bits(et_emu_dev *dev) {
  return dev->layer->getDmaAlignment();
}

uint32_t et_emu_sq_count(et_emu_dev *dev) {
  return static_cast<uint32_t>(dev->layer->getSubmissionQueuesCount(kDevice));
}

uint32_t et_emu_sq_max_msg(et_emu_dev *dev) {
  return static_cast<uint32_t>(
      dev->layer->getSubmissionQueueSizeMasterMinion(kDevice));
}

int et_emu_push_sq(et_emu_dev *dev, uint16_t sq_index, const uint8_t *cmd,
                   size_t size, uint8_t is_dma) {
  try {
    // sendCommandMasterMinion takes a mutable buffer; copy the command in.
    std::vector<std::byte> buf(size);
    std::memcpy(buf.data(), cmd, size);
    dev::CmdFlagMM flags;
    flags.isDma_ = is_dma != 0;
    bool sent = dev->layer->sendCommandMasterMinion(kDevice, sq_index,
                                                    buf.data(), size, flags);
    return sent ? 1 : 0;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown exception";
    return -1;
  }
}

int et_emu_pop_cq(et_emu_dev *dev, uint8_t *out, size_t out_cap,
                  size_t *out_len) {
  try {
    std::vector<std::byte> rsp;
    bool got = dev->layer->receiveResponseMasterMinion(kDevice, rsp);
    if (!got) {
      return 0;
    }
    if (rsp.size() > out_cap) {
      return -1;
    }
    std::memcpy(out, rsp.data(), rsp.size());
    *out_len = rsp.size();
    return 1;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown exception";
    return -1;
  }
}

int et_emu_wait_cq(et_emu_dev *dev, uint32_t timeout_ms) {
  try {
    uint64_t sq_bitmap = 0;
    bool cq_available = false;
    dev->layer->waitForEpollEventsMasterMinion(
        kDevice, sq_bitmap, cq_available, std::chrono::milliseconds(timeout_ms));
    return cq_available ? 1 : 0;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown exception";
    return -1;
  }
}

int et_emu_wait_sq(et_emu_dev *dev, uint32_t timeout_ms) {
  try {
    uint64_t sq_bitmap = 0;
    bool cq_available = false;
    dev->layer->waitForEpollEventsMasterMinion(
        kDevice, sq_bitmap, cq_available, std::chrono::milliseconds(timeout_ms));
    return sq_bitmap != 0 ? 1 : 0;
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown exception";
    return -1;
  }
}

void *et_emu_alloc_dma(et_emu_dev *dev, size_t size, int writeable) {
  try {
    return dev->layer->allocDmaBuffer(kDevice, size, writeable != 0);
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return nullptr;
  } catch (...) {
    g_last_error = "unknown exception";
    return nullptr;
  }
}

void et_emu_free_dma(et_emu_dev *dev, void *buf) {
  try {
    dev->layer->freeDmaBuffer(buf);
  } catch (...) {
    // Best-effort free.
  }
}

long et_emu_fw_update(et_emu_dev *dev, const uint8_t *img, size_t size) {
  try {
    std::vector<unsigned char> image(img, img + size);
    return dev->layer->updateFirmwareImage(kDevice, image);
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown exception";
    return -1;
  }
}

long et_emu_trace_size(et_emu_dev *dev, uint8_t trace_type) {
  try {
    return static_cast<long>(dev->layer->getTraceBufferSizeMasterMinion(
        kDevice, static_cast<dev::TraceBufferType>(trace_type)));
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown exception";
    return -1;
  }
}

long et_emu_extract_trace(et_emu_dev *dev, uint8_t trace_type, uint8_t *out,
                          size_t out_cap) {
  try {
    std::vector<std::byte> buf;
    bool got = dev->layer->getTraceBufferServiceProcessor(
        kDevice, static_cast<dev::TraceBufferType>(trace_type), buf);
    if (!got) {
      return 0;
    }
    if (buf.size() > out_cap) {
      return -1;
    }
    std::memcpy(out, buf.data(), buf.size());
    return static_cast<long>(buf.size());
  } catch (const std::exception &e) {
    g_last_error = e.what();
    return -1;
  } catch (...) {
    g_last_error = "unknown exception";
    return -1;
  }
}

} // extern "C"
