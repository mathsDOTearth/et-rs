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

use crate::device::{Device, DeviceRegion};
use crate::error::{Error, Result};
use crate::transport::Transport;

/// Types that may be copied verbatim between host and device memory.
///
/// # Safety
/// An implementor must be "plain old data": a scalar or a `#[repr(C)]` struct of
/// such, with no padding bytes and valid for every bit pattern. Then
/// reinterpreting a value (or slice) as bytes and back is always sound, which is
/// what the upload/download paths rely on. The provided implementations cover
/// the integer and floating scalars; implement it for your own `#[repr(C)]` POD
/// structs to store them in a [`DeviceBuffer`].
pub unsafe trait DevicePod: Copy + 'static {}

macro_rules! impl_device_pod {
    ($($t:ty),* $(,)?) => { $( unsafe impl DevicePod for $t {} )* };
}
impl_device_pod!(
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64, usize, isize
);

/// Reinterpret a POD slice as its byte representation.
fn as_bytes<E: DevicePod>(data: &[E]) -> &[u8] {
    // SAFETY: `E: DevicePod` is POD, so the `size_of_val` bytes backing `data`
    // are a valid byte representation of it.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
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
        let region = self.alloc((n * size_of::<E>()) as u64)?;
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
        let mut out: Vec<E> = Vec::with_capacity(buf.len());
        // SAFETY: capacity for `buf.len()` values of `E` is reserved; `E: DevicePod`
        // is valid for any bit pattern, so once `memcpy_d2h` has filled exactly
        // `byte_len` bytes the elements are initialised and the length is set.
        // On error `out` drops with length 0, freeing the buffer without touching
        // uninitialised memory.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, buf.byte_len());
            self.memcpy_d2h(buf.addr(), dst)?;
            out.set_len(buf.len());
        }
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
        let region = self.alloc((n * CACHE_LINE) as u64)?;
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
