mod immediate;
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
mod multithreaded;
#[cfg(all(not(target_arch = "wasm32"), feature = "rayon"))]
mod rayon;

use crate::decode::decoder::{choose_color_convert_func, ColorTransform};
use crate::decode::error::Result;
use crate::decode::parser::{Component, Dimensions};
use crate::decode::upsampler::Upsampler;

use alloc::sync::Arc;
#[cfg(feature = "platform_independent")]
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ops::Range;

pub struct RowData {
    pub index: usize,
    pub component: Component,
    pub quantization_table: Arc<[u16; 64]>,
    /// A previously-used output plane to refill instead of allocating.
    ///
    /// Sizing a fresh output plane measured **66.7% of whole-frame decode** —
    /// 3.1 MB per 1080p frame at an effective 522 MB/s, which is first-touch
    /// page-fault cost, not memset (touching the same pages a second time cost
    /// 1.2%). Handing back an already-faulted allocation removes it. FFmpeg
    /// solves the same problem with `AVBufferPool`.
    pub recycled: Option<Vec<u8>>,
}

pub trait Worker {
    fn start(&mut self, row_data: RowData) -> Result<()>;
    fn append_row(&mut self, row: (usize, Vec<i16>)) -> Result<()>;
    fn get_result(&mut self, index: usize) -> Result<Vec<u8>>;

    /// Hand back a coefficient buffer the worker has finished with, so the
    /// caller can refill it instead of allocating a new one.
    ///
    /// `decode_scan` otherwise allocates and zeroes a fresh `Vec<i16>` for every
    /// MCU row of every component — about 6.3 MB per 1080p frame. A worker that
    /// consumes rows synchronously can simply return the buffer; one that ships
    /// them to another thread cannot, so the default is `None` and the caller
    /// falls back to allocating.
    fn reclaim_buffer(&mut self) -> Option<Vec<i16>> {
        None
    }
    /// Default implementation for spawning multiple tasks.
    fn append_rows(&mut self, row: &mut dyn Iterator<Item = (usize, Vec<i16>)>) -> Result<()> {
        for item in row {
            self.append_row(item)?;
        }
        Ok(())
    }

    /// Concrete access to the synchronous worker, when this IS one.
    ///
    /// `fused_block` is called once per block -- ~49k times per 1080p frame --
    /// and through `&mut dyn Worker` every one of those is an indirect call that
    /// also blocks inlining of the transform dispatch into the decode loop.
    /// Resolving the concrete type ONCE per scan makes the whole fused body a
    /// static call the optimizer can see through.
    fn as_immediate(&mut self) -> Option<&mut immediate::ImmediateWorker> {
        None
    }

    /// Whether this worker can inverse-transform a block the moment it is
    /// decoded, instead of accumulating a whole MCU row of coefficients first.
    ///
    /// The row buffer costs a full write -> zero -> read round trip of **18.8 MB
    /// per 1080p frame**, through a ~60 KB-per-row buffer that does not fit in
    /// L1. It exists only so a row can be shipped to another THREAD; a worker
    /// that consumes blocks synchronously has no use for it and can keep the
    /// block in L1 from entropy decode straight through the IDCT.
    fn supports_fused(&self) -> bool {
        false
    }
}

#[allow(dead_code)]
/// Which worker the last decode actually instantiated.
///
/// `set_single_threaded` used to be silently ignored for baseline images: the
/// scan-loop call site hardcoded `Multithreaded`, and because the worker is
/// cached on first use it won. The flag looked live -- the profile barely moved
/// -- while the immediate worker, and with it `reclaim_buffer`, never ran.
/// This counter is what makes that observable from a test.
#[cfg(feature = "std")]
static LAST_WORKER_WAS_IMMEDIATE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "std")]
fn record_worker_choice(prefer: &PreferWorkerKind) {
    LAST_WORKER_WAS_IMMEDIATE.store(
        matches!(prefer, PreferWorkerKind::Immediate),
        core::sync::atomic::Ordering::Relaxed,
    );
}

#[cfg(not(feature = "std"))]
fn record_worker_choice(_prefer: &PreferWorkerKind) {}

/// Whether the most recently created worker was the single-threaded one.
#[cfg(feature = "std")]
pub fn last_worker_was_immediate() -> bool {
    LAST_WORKER_WAS_IMMEDIATE.load(core::sync::atomic::Ordering::Relaxed)
}

#[derive(Debug)]
pub enum PreferWorkerKind {
    Immediate,
    Multithreaded,
}

#[derive(Default)]
pub struct WorkerScope {
    inner: core::cell::RefCell<Option<WorkerScopeInner>>,
}

// The immediate worker embeds its plane buffers while the threaded one holds
// channel handles, so the variants differ in size by design. Boxing the large
// variant would add an allocation to the path this crate spends most of its
// time in.
#[allow(clippy::large_enum_variant)]
enum WorkerScopeInner {
    #[cfg(all(not(target_arch = "wasm32"), feature = "rayon"))]
    Rayon(Box<rayon::Scoped>),
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    Multithreaded(multithreaded::MpscWorker),
    Immediate(immediate::ImmediateWorker),
}

impl WorkerScope {
    pub fn with<T>(with: impl FnOnce(&Self) -> T) -> T {
        with(&WorkerScope {
            inner: RefCell::default(),
        })
    }

    pub fn get_or_init_worker<T>(
        &self,
        prefer: PreferWorkerKind,
        f: impl FnOnce(&mut dyn Worker) -> T,
    ) -> T {
        let mut inner = self.inner.borrow_mut();
        if inner.is_none() {
            record_worker_choice(&prefer);
        }
        let inner = inner.get_or_insert_with(move || match prefer {
            #[cfg(all(not(target_arch = "wasm32"), feature = "rayon"))]
            PreferWorkerKind::Multithreaded => WorkerScopeInner::Rayon(Default::default()),
            #[allow(unreachable_patterns)]
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            PreferWorkerKind::Multithreaded => WorkerScopeInner::Multithreaded(Default::default()),
            _ => WorkerScopeInner::Immediate(Default::default()),
        });

        f(match &mut *inner {
            #[cfg(all(not(target_arch = "wasm32"), feature = "rayon"))]
            WorkerScopeInner::Rayon(worker) => worker.as_mut(),
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            WorkerScopeInner::Multithreaded(worker) => worker,
            WorkerScopeInner::Immediate(worker) => worker,
        })
    }
}

/// Whether `color_transform`'s converter reads exactly `output_width` samples
/// per component, so a full-resolution plane row can be passed to it in place
/// (see `Upsampler::upsample_and_interleave_rows`).
pub(crate) fn reads_exact_width(color_transform: ColorTransform) -> bool {
    matches!(color_transform, ColorTransform::YCbCr | ColorTransform::RGB)
}

/// How many cores this process may actually run on.
///
/// `available_parallelism` honours the CPU affinity mask on Linux, but not on
/// Windows, where it reports every logical processor even when the process
/// is confined to one. Threading a decode then only time-slices that core:
/// measured with the default (threaded) decoder pinned to one core, 1080p
/// 4:2:0 ran 1.35-1.48x slower than the same decoder single-threaded. So on
/// Windows the affinity mask is consulted too (per call: a process can be
/// re-pinned while running).
pub(crate) fn usable_cores() -> usize {
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        #[cfg(all(windows, not(feature = "platform_independent")))]
        {
            if let Some(allowed) = windows_affinity_cores() {
                return cores.min(allowed).max(1);
            }
        }
        cores
    }
    #[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
    {
        1
    }
}

/// The number of processors in this process's affinity mask (within its
/// processor group), or `None` if Windows will not say.
#[cfg(all(windows, feature = "std", not(feature = "platform_independent")))]
fn windows_affinity_cores() -> Option<usize> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessAffinityMask(process: isize, mask: *mut usize, system: *mut usize) -> i32;
    }
    let (mut process_mask, mut system_mask) = (0usize, 0usize);
    // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no closing;
    // both out-pointers are valid for the duration of the call.
    #[allow(unsafe_code)]
    let ok =
        unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process_mask, &mut system_mask) };
    (ok != 0 && process_mask != 0).then(|| process_mask.count_ones() as usize)
}

/// How many threads the RGB output stage (upsample + colour convert) uses.
///
/// One band per ~256K output pixels -- a 1080p frame's output stage is ~1 ms
/// serial, while spawning a scoped thread costs tens of microseconds -- capped
/// by the machine and by `MAX_OUTPUT_THREADS`, past which the stage is bound
/// by memory bandwidth rather than cores.
fn output_bands(width: usize, height: usize) -> usize {
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    {
        const PIXELS_PER_BAND: usize = 256 * 1024;
        let max = crate::cached_knob!(usize, {
            crate::knob("RUSTY_JPEG_OUTPUT_THREADS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(MAX_OUTPUT_THREADS)
        });
        (width * height / PIXELS_PER_BAND).clamp(1, usable_cores().min(max).max(1))
    }
    #[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
    {
        let _ = (width, height);
        1
    }
}

/// Measured on 1080p and 2000x1250 (24-thread laptop): 4 threads beat 8 on
/// both, min and median. Override with `RUSTY_JPEG_OUTPUT_THREADS`.
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
const MAX_OUTPUT_THREADS: usize = 4;

pub fn compute_image_parallel(
    components: &[Component],
    data: Vec<Vec<u8>>,
    output_size: Dimensions,
    color_transform: ColorTransform,
    parallel: bool,
) -> Result<Vec<u8>> {
    #[cfg(all(not(target_arch = "wasm32"), feature = "rayon"))]
    {
        let _ = parallel;
        return rayon::compute_image_parallel(components, data, output_size, color_transform);
    }

    #[allow(unreachable_code)]
    {
        let color_convert_func = choose_color_convert_func(components.len(), color_transform)?;
        let upsampler = Upsampler::new(components, output_size.width, output_size.height)?;
        let line_size = output_size.width as usize * components.len();
        let height = output_size.height as usize;
        let len = line_size * height;
        let width = output_size.width as usize;
        let direct = reads_exact_width(color_transform);
        let bands = if parallel {
            output_bands(width, height)
        } else {
            1
        };

        // Not zero-filled: every byte is written below (see
        // `upsample_and_interleave_rows_uninit`).
        #[cfg(not(feature = "platform_independent"))]
        {
            let mut image: Vec<u8> = Vec::with_capacity(len);
            in_bands(
                &mut image.spare_capacity_mut()[..len],
                line_size,
                bands,
                |rows, lines| {
                    upsampler.upsample_and_interleave_rows_uninit(
                        &data,
                        rows,
                        width,
                        lines,
                        color_convert_func,
                        direct,
                    )
                },
            );
            // SAFETY: the spare capacity passed to `in_bands` is exactly
            // `height` rows of `line_size` bytes; `in_bands` hands every row to
            // exactly one call, and `upsample_and_interleave_rows_uninit`
            // writes every byte of every row it is given (each row is rendered
            // to a buffer of `line_size` bytes and copied out whole). So all
            // `len` bytes are initialised.
            #[allow(unsafe_code)]
            unsafe {
                image.set_len(len)
            };
            Ok(image)
        }

        // `platform_independent` forbids `unsafe`, so it pays for the zeroing.
        #[cfg(feature = "platform_independent")]
        {
            let mut image = vec![0u8; len];
            in_bands(&mut image, line_size, bands, |rows, lines| {
                upsampler.upsample_and_interleave_rows(
                    &data,
                    rows,
                    width,
                    lines,
                    color_convert_func,
                    direct,
                )
            });
            Ok(image)
        }
    }
}

/// Render an image of `line_size`-byte rows into `out` in `bands` row bands,
/// one thread each (the calling thread takes the first); `render` is given a
/// row range and exactly the slice of `out` those rows occupy. Every row is
/// rendered exactly once whatever the band count, and each output row
/// depends only on the finished planes, so the bytes cannot depend on it.
fn in_bands<T: Send>(
    out: &mut [T],
    line_size: usize,
    bands: usize,
    render: impl Fn(Range<usize>, &mut [T]) + Sync,
) {
    let height = out.len().checked_div(line_size).unwrap_or(0);
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    if bands > 1 {
        let rows_per_band = height.div_ceil(bands);
        let render = &render;
        std::thread::scope(|scope| {
            let mut chunks = out.chunks_mut(line_size * rows_per_band).enumerate();
            let first = chunks.next();
            for (band, lines) in chunks {
                let start = band * rows_per_band;
                scope.spawn(move || render(start..start + lines.len() / line_size, lines));
            }
            if let Some((_, lines)) = first {
                render(0..lines.len() / line_size, lines);
            }
        });
        return;
    }
    let _ = bands;
    render(0..height, out);
}
