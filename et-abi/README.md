# et-abi

Shared host/device ABI types for the Esperanto ET-SoC-1, used by the
[`et-rs`](https://github.com/mathsDOTearth/et-rs) host driver and the
[`et-k-rs`](https://github.com/mathsDOTearth/et-rs) device-side kernel library.

A kernel launch passes its arguments by pointer: the host stages an argument
struct in device memory and the firmware delivers its address to the kernel (in
register `a0`). This crate defines those argument structs **once**, so the host
launcher and the device kernel cannot drift on field order, sizes, or padding.

Because both the host (x86-64) and the device (RV64) are little-endian, the
in-memory `#[repr(C)]` layout is the wire layout: the host takes the struct's
bytes with [`DeviceArgs::as_bytes`] and the kernel reinterprets the pointer with
[`DeviceArgs::from_ptr`]. No serialisation step is involved.

The crate is `no_std` with no dependencies, so it builds for the host and for the
`riscv64imac-unknown-none-elf` device target alike.

```rust
use et_abi::{DeviceArgs, ReduceArgs};

// Host: build the args the kernel will read.
let args = ReduceArgs { input: 0x8005_0000, out: 0x8006_0000, n: 1024, n_harts: 64 };
let bytes: &[u8] = args.as_bytes();

// Device: recover them from the pointer the firmware passed in a0.
// let args = unsafe { ReduceArgs::from_ptr(a0 as *const u8) };
```

It also holds the architectural constants both sides must agree on:
`CACHE_LINE` (the per-hart output stride, so host padding and device writes
match and avoid false sharing), `HARTS_PER_SHIRE`, and
`HARTS_PER_NEIGHBOURHOOD`.

`CachePadded<T>` is a `#[repr(align(64))]` wrapper that places `T` on its own
cache line. Use it for per-hart output cells on the software-coherent ET-SoC-1
to prevent false-sharing corruption structurally, without explicit cache
operations.

## Tensor-extension ABI

The sGEMM kernel and its host launcher share `GemmArgs`, a second argument
struct for single-precision GEMM:

```rust
use et_abi::{DeviceArgs, GemmArgs};

let args = GemmArgs {
    a:        a_device_addr,
    b:        b_device_addr,
    c:        c_device_addr,
    n_shires: 1,
    m: 64, n: 64, k: 64,
    lda: 256, ldb: 256, ldc: 256,  // row strides in bytes; multiple of 64
    alpha: 1.0, beta: 0.0,
};
```

`GemmArgs` is 64 bytes exactly (4 x u64 for pointers / shire count, 8 x
u32/f32 for dimensions, strides, and scaling). A compile-time assertion
(`assert!(size_of::<GemmArgs>() == 64)`) guards the layout.

The crate also exports the tensor-extension architectural constants:

| Constant | Value | Meaning |
|---|---|---|
| `TENSOR_ALIGN` | 64 | Minimum alignment (bytes) for tensor load/store addresses and leading dimensions. |
| `GEMM_TILE_M` | 16 | sGEMM output tile rows. |
| `GEMM_TILE_K` | 16 | sGEMM inner-dimension tile size. |
| `GEMM_TILE_N` | 16 | sGEMM output tile columns; partial last-column tiles are handled transparently. |
| `MINIONS_PER_SHIRE` | 32 | Minion cores per compute shire; used to compute the cyclic tile step. |

## Thanks

Thanks to AiNEKKO https://nekko.ai/ and AI Foundry https://aifoundry.org/ for allowing me 
time on their community ET-SoC-1 servers to develop this code.

The ET-SoC-1 ET Platform SDK and software emulator can be found on their GitHub: https://github.com/aifoundry-org/et-platform

## Licence

Apache-2.0, matching the ET Platform SDK headers this crate binds to.  
ET-SoC-1 ET Platform API is under the Apache 2 License.
