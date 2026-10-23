//! Component planes kept for reuse by the next decode on the same thread.
//!
//! A `decode()` to RGB sizes one plane per component -- ~3.1 MB for a 1080p
//! 4:2:0 frame -- and drops them as soon as the image is assembled. Callers
//! that construct a fresh `Decoder` per image (the common shape: a thumbnailer,
//! an MJPEG frame loop) therefore paid for those pages to be faulted in again
//! on every decode. Under an allocator that recycles freed memory (rusty_alloc,
//! mimalloc) that was nearly free; under the Windows system heap a 1080p frame
//! paid +0.43 ms for it, enough to put the whole decode 1.17x behind zune-jpeg
//! on one core. Keeping the planes here makes the cost independent of the
//! allocator.
//!
//! The pool is bounded (see `MAX_PLANES` / `MAX_BYTES`): after an unusually
//! large image it keeps nothing rather than pinning that memory, and
//! `RUSTY_JPEG_PLANE_POOL=0` turns it off. Planes are refilled, never trusted:
//! the decoder overwrites every block before a plane is read (see
//! `ImmediateWorker::start_immediate`), so reuse cannot leak one image into the
//! next. Without `std` there is no thread-local storage and this is a no-op.

use alloc::vec::Vec;

/// At most this many planes are kept: one image's worth (Y, Cb, Cr, and a
/// fourth for CMYK/YCCK).
#[cfg(feature = "std")]
const MAX_PLANES: usize = 4;
/// ...and at most this many bytes of them (a 4K 4:4:4 frame is ~25 MB).
#[cfg(feature = "std")]
const MAX_BYTES: usize = 64 << 20;

#[cfg(feature = "std")]
std::thread_local! {
    static POOL: core::cell::RefCell<Vec<Vec<u8>>> = const { core::cell::RefCell::new(Vec::new()) };
}

#[cfg(feature = "std")]
fn enabled() -> bool {
    crate::cached_knob!(bool, {
        crate::knob("RUSTY_JPEG_PLANE_POOL").as_deref() != Some("0")
    })
}

/// The pooled plane whose capacity fits `len` most tightly, if any.
pub(crate) fn take(len: usize) -> Option<Vec<u8>> {
    #[cfg(feature = "std")]
    {
        if !enabled() {
            return None;
        }
        POOL.try_with(|pool| {
            let mut pool = pool.borrow_mut();
            let best = pool
                .iter()
                .enumerate()
                .filter(|(_, plane)| plane.capacity() >= len)
                .min_by_key(|(_, plane)| plane.capacity())
                .map(|(i, _)| i)?;
            Some(pool.swap_remove(best))
        })
        .ok()
        .flatten()
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = len;
        None
    }
}

/// Hand planes back after a decode. Keeps them only within the bounds.
pub(crate) fn give_back(planes: Vec<Vec<u8>>) {
    #[cfg(feature = "std")]
    {
        if !enabled() {
            return;
        }
        let _ = POOL.try_with(|pool| {
            let mut pool = pool.borrow_mut();
            for plane in planes {
                if plane.capacity() == 0 {
                    continue;
                }
                let held: usize = pool.iter().map(Vec::capacity).sum();
                if pool.len() >= MAX_PLANES || held + plane.capacity() > MAX_BYTES {
                    // Full, or this image was large: keep the newest set
                    // rather than growing.
                    pool.clear();
                    if plane.capacity() > MAX_BYTES {
                        continue;
                    }
                }
                pool.push(plane);
            }
        });
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = planes;
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn best_fit_and_bounds() {
        give_back(vec![
            Vec::with_capacity(100),
            Vec::with_capacity(10),
            Vec::with_capacity(50),
        ]);
        let a = take(40).unwrap();
        assert!(
            a.capacity() >= 40 && a.capacity() < 100,
            "best fit, got {}",
            a.capacity()
        );
        assert!(take(1000).is_none());
        // An oversized plane is never kept.
        give_back(vec![Vec::with_capacity(MAX_BYTES + 1)]);
        assert!(take(MAX_BYTES).is_none());
        // Drain.
        while take(0).is_some() {}
    }
}
