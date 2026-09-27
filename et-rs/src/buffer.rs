//! Typed device-memory buffers layered over the byte-oriented DMA primitives.
//!
//! [`Device::alloc`] plus [`Device::memcpy_h2d`]/[`Device::memcpy_d2h`] are the
//! low-level interface: they deal in raw addresses and `&[u8]`, so call sites
//! reinterpret their data with `unsafe` slice casts and track sizes by hand.
//! The types here restore that type information. A [`DeviceBuffer<T>`] is a
//! lightweight handle (device address, element count, element type); the owning
//! [`Device`] uploads and downloads whole typed slices, so no byte arithmetic or
//! `unsafe` reinterpretation appears at the call site.
//!
//! [`PaddedArray<T>`] additionally lays out one element per cache line, which the
//! ET-SoC-1's software-managed coherence requires when distinct harts write
//! distinct elements: without the padding, two harts sharing a cache line
//! corrupt each other's writes silently (false sharing). It pairs with the
//! device-side `Grid::output_cell`, which writes each hart's result at
//! `base + hart * CACHE_LINE`.

use std::marker::PhantomData;
use std::mem::size_of;

use et_abi::CACHE_LINE;
pub use et_abi::DevicePod;

use crate::device::{Device, DeviceRegion, DmaOptions};
use crate::error::{Error, Result};
use crate::transport::Transport;

/// Reinterpret a POD slice as its byte representation.
fn as_bytes<E: DevicePod>(data: &[E]) -> &[u8] {
    // SAFETY: `E: DevicePod` is POD, so the `size_of_val` bytes backing `data`
    // are a valid byte representation of it.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// Mutable byte view of a slice of device POD values.
fn as_bytes_mut<E: DevicePod>(data: &mut [E]) -> &mut [u8] {
    // SAFETY: as in `as_bytes`; additionally, any byte pattern written through
    // the view is a valid `E`, because `E: DevicePod` admits every bit pattern.
    unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, std::mem::size_of_val(data))
    }
}

/// A host vector of `n` zero-initialised values of `E`, ready to receive a DMA.
fn zeroed_vec<E: DevicePod>(n: usize) -> Result<Vec<E>> {
    byte_len_of::<E>(n)?;
    let mut out: Vec<E> = Vec::with_capacity(n);
    // SAFETY: capacity for `n` values is reserved, the bytes are initialised to
    // zero before the length is set, and all-zero is a valid `E: DevicePod`.
    unsafe {
        std::ptr::write_bytes(out.as_mut_ptr(), 0, n);
        out.set_len(n);
    }
    Ok(out)
}

/// Size in bytes of `n` values of `E`, or [`Error::Limit`] on overflow.
fn byte_len_of<E>(n: usize) -> Result<usize> {
    n.checked_mul(size_of::<E>()).ok_or_else(|| {
        Error::Limit(format!(
            "{n} elements of {} bytes overflow usize",
            size_of::<E>()
        ))
    })
}

/// A typed handle to a contiguous device array of `len` values of `T`.
///
/// Created by [`Device::alloc_array`] or [`Device::upload`]. The handle is
/// `Copy` and does not own the DRAM (there is no device-side free); it simply
/// records where the array lives and how to interpret it.
#[derive(Clone, Copy, Debug)]
pub struct DeviceBuffer<T> {
    region: DeviceRegion,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: DevicePod> DeviceBuffer<T> {
    /// Device address of the first element (use as a kernel argument).
    pub fn addr(&self) -> u64 {
        self.region.addr
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size of the buffer in bytes.
    pub fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// The underlying untyped region.
    pub fn region(&self) -> DeviceRegion {
        self.region
    }
}

/// A device array in which each element occupies a full cache line.
///
/// Use this for per-hart outputs so distinct harts never share a cache line
/// (see the module docs). `T` must be no larger than one cache line.
#[derive(Clone, Copy, Debug)]
pub struct PaddedArray<T> {
    region: DeviceRegion,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: DevicePod> PaddedArray<T> {
    /// Device address of element 0 (element `i` lives at `addr + i * CACHE_LINE`).
    pub fn addr(&self) -> u64 {
        self.region.addr
    }

    /// Number of elements (cache lines).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the array holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The stride between consecutive elements, in bytes (one cache line).
    pub fn stride(&self) -> usize {
        CACHE_LINE
    }

    /// The underlying untyped region.
    pub fn region(&self) -> DeviceRegion {
        self.region
    }
}

impl<Tr: Transport> Device<Tr> {
    /// Allocate an uninitialised device array of `n` values of `E`.
    pub fn alloc_array<E: DevicePod>(&self, n: usize) -> Result<DeviceBuffer<E>> {
        let region = self.alloc(byte_len_of::<E>(n)? as u64)?;
        Ok(DeviceBuffer {
            region,
            len: n,
            _marker: PhantomData,
        })
    }

    /// Allocate a device array and upload `data` into it (host -> device).
    pub fn upload<E: DevicePod>(&self, data: &[E]) -> Result<DeviceBuffer<E>> {
        let buf = self.alloc_array::<E>(data.len())?;
        self.write_buffer(&buf, data)?;
        Ok(buf)
    }

    /// Copy `data` into an existing device buffer (host -> device).
    pub fn write_buffer<E: DevicePod>(&self, buf: &DeviceBuffer<E>, data: &[E]) -> Result<()> {
        if data.len() > buf.len() {
            return Err(Error::Limit(format!(
                "write of {} elements exceeds buffer capacity {}",
                data.len(),
                buf.len()
            )));
        }
        self.memcpy_h2d(as_bytes(data), buf.addr())
    }

    /// Download a device buffer into a host `Vec` (device -> host).
    pub fn download<E: DevicePod>(&self, buf: &DeviceBuffer<E>) -> Result<Vec<E>> {
        let mut out = zeroed_vec::<E>(buf.len())?;
        self.memcpy_d2h(buf.addr(), as_bytes_mut(&mut out))?;
        Ok(out)
    }

    /// Allocate a cache-line-padded device array of `n` values of `E`.
    ///
    /// # Panics
    /// If `E` is larger than one cache line.
    pub fn alloc_padded<E: DevicePod>(&self, n: usize) -> Result<PaddedArray<E>> {
        assert!(
            size_of::<E>() <= CACHE_LINE,
            "padded element ({} bytes) exceeds one cache line ({CACHE_LINE} bytes)",
            size_of::<E>()
        );
        let region = self.alloc(byte_len_of::<[u8; CACHE_LINE]>(n)? as u64)?;
        Ok(PaddedArray {
            region,
            len: n,
            _marker: PhantomData,
        })
    }

    /// Download a cache-line-padded array, extracting one `E` from each line.
    pub fn download_padded<E: DevicePod>(&self, arr: &PaddedArray<E>) -> Result<Vec<E>> {
        let mut raw = vec![0u8; arr.len() * CACHE_LINE];
        self.memcpy_d2h(arr.addr(), &mut raw)?;
        let mut out = Vec::with_capacity(arr.len());
        for i in 0..arr.len() {
            let off = i * CACHE_LINE;
            // SAFETY: `E: DevicePod` is valid for any bit pattern; `off .. off +
            // size_of::<E>()` lies within `raw` (guaranteed by the allocation of
            // `len * CACHE_LINE` and `size_of::<E>() <= CACHE_LINE`). The offset
            // need not be aligned, hence `read_unaligned`.
            out.push(unsafe { std::ptr::read_unaligned(raw.as_ptr().add(off) as *const E) });
        }
        Ok(out)
    }

    /// Fill every byte of `region` with `value` (host -> device).
    ///
    /// Test programs use this to prefill an output buffer with a sentinel (for
    /// example `0xFF`, a NaN pattern for `f32`) before a launch, so that a kernel
    /// which fails to write a result cannot pass on data left in DRAM by an
    /// earlier run: allocation restarts at the same addresses and does not
    /// clear memory.
    pub fn fill(&self, region: DeviceRegion, value: u8) -> Result<()> {
        let len = usize::try_from(region.size).map_err(|_| {
            Error::Limit(format!(
                "fill of {} bytes exceeds the host address space",
                region.size
            ))
        })?;
        self.memcpy_h2d(&vec![value; len], region.addr)
    }

    /// Upload a typed slice to device address `dst`, without allocating a new
    /// [`DeviceBuffer`].
    ///
    /// Use when `dst` is an address already held externally, or when writing
    /// into a sub-region of a larger device allocation. Equivalent to
    /// [`Device::memcpy_h2d`] but accepts a typed slice rather than `&[u8]`.
    ///
    /// For concurrent DMA via a specific submission queue, use
    /// [`Device::upload_slice_opts`].
    pub fn upload_slice<E: DevicePod>(&self, data: &[E], dst: u64) -> Result<()> {
        self.memcpy_h2d(as_bytes(data), dst)
    }

    /// Upload a typed slice to device address `dst` using explicit [`DmaOptions`].
    ///
    /// Equivalent to [`Device::upload_slice`] but routes the DMA command
    /// through `opts`, enabling SQ selection and optional timeout override.
    pub fn upload_slice_opts<E: DevicePod>(
        &self,
        data: &[E],
        dst: u64,
        opts: &DmaOptions,
    ) -> Result<()> {
        self.memcpy_h2d_opts(as_bytes(data), dst, opts)
    }

    /// Download `n` values of `E` from device address `src` into a host `Vec`.
    ///
    /// The untracked counterpart to [`Device::upload_slice`]: downloads from a
    /// raw device address without a [`DeviceBuffer`] handle. Callers are
    /// responsible for ensuring `src` is valid and `n` does not exceed the
    /// allocated region.
    ///
    /// For concurrent DMA, use [`Device::download_slice_opts`].
    pub fn download_slice<E: DevicePod>(&self, src: u64, n: usize) -> Result<Vec<E>> {
        let mut out = zeroed_vec::<E>(n)?;
        self.memcpy_d2h(src, as_bytes_mut(&mut out))?;
        Ok(out)
    }

    /// Download `n` values of `E` from device address `src` using explicit
    /// [`DmaOptions`].
    ///
    /// Equivalent to [`Device::download_slice`] but routes the DMA command
    /// through `opts`.
    pub fn download_slice_opts<E: DevicePod>(
        &self,
        src: u64,
        n: usize,
        opts: &DmaOptions,
    ) -> Result<Vec<E>> {
        let mut out = zeroed_vec::<E>(n)?;
        self.memcpy_d2h_opts(src, as_bytes_mut(&mut out), opts)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pod_bytes_are_little_endian() {
        let v = [1u32, 0x0203_0405u32];
        let b = as_bytes(&v);
        assert_eq!(b.len(), 8);
        assert_eq!(&b[0..4], &1u32.to_le_bytes());
        assert_eq!(&b[4..8], &0x0203_0405u32.to_le_bytes());
    }
}
