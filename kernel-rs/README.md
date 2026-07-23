# hello-kernel-rs

The ET-SoC-1 "hello world" **compute kernel written in pure Rust** — the
device-side counterpart to the host `et_soc1` crate, and a drop-in replacement
for the SDK test drive's C `hello.c`.

Every hart writes `"Hello World from hart N"` into its U-mode trace buffer and
returns to firmware. There is **no C dependency**: it reimplements `get_hart_id`
(the `hartid` CSR) and the `et_printf` / `Trace_String` trace write directly. See
`src/main.rs` for how a hart reaches its trace control block.

## Build

It cross-compiles to the compute harts (RV64IMAC); the target, code model
(`medium` = medany, required for the fixed high link address) and linker script
are set in `.cargo/config.toml`:

```bash
rustup target add riscv64imac-unknown-none-elf   # once
cargo build --release
# -> target/riscv64imac-unknown-none-elf/release/hello-rs   (an ELF)
```

## Run

Load and launch it with the host crate's examples, exactly like the C kernel:

```bash
# Software emulator (no hardware), from the repository root:
cargo run --features emu --example hello_sysemu -- \
    kernel-rs/target/riscv64imac-unknown-none-elf/release/hello-rs

# Real hardware:
cargo run --example hello -- \
    kernel-rs/target/riscv64imac-unknown-none-elf/release/hello-rs
```

Expected: 64 decoded `Hello World from hart N` lines.

## How it works

* **Entry/exit** (`_start`, in `global_asm!`): initialise `gp`, call
  `entry_point`, then `ecall` with `SYSCALL_RETURN_FROM_KERNEL` /
  `KERNEL_RETURN_SUCCESS`. Firmware sets the stack pointer before entry.
* **`get_hart_id`**: `csrr` of the custom `hartid` CSR (`0xCD0`).
* **Trace write**: firmware populates a per-hart control block at
  `0x8004F23000 + hart_id * 64` holding the trace slice's base and write offset.
  Logging bumps the offset and writes a `trace_string_t` (16-byte header +
  NUL-terminated, 8-byte-aligned string), just as the SDK's `Trace_String` does.
  Firmware finalises the buffer's size headers on return; the host DMAs it back
  and decodes it with `et_soc1::trace`.
* No `.bss` (firmware does not zero it; the linker script asserts this), no heap,
  no unwinding (`panic = "abort"`).

It is a separate cargo package from the host crate because it targets RISC-V
bare-metal, not the host.
