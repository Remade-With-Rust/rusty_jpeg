#![allow(unsafe_code)]

//! Vector kernels for the decoder.
//!
//! **Contract: every kernel here is a bit-exact twin of its scalar oracle.** A
//! decode must produce the same pixels on every CPU, so no kernel may round
//! differently from the scalar code it replaces -- an IDCT or colour kernel
//! that is merely "close" makes the output depend on which machine ran it.
//! The oracle tests (`idct_kernels_match_scalar`, `avx2_pair_matches_scalar`,
//! `ycbcr_kernels_match_scalar`, `simd_decode_matches_scalar_decode`) assert
//! equality, never a tolerance.
//!
//! On x86 the kernels are selected at runtime. `RUSTY_JPEG_DECODE_ISA`
//! (`scalar`, `ssse3` or `avx2`; read once) caps the selection below what the
//! CPU offers, so one binary can be run on every rung.

// The x86 kernels are selected by runtime feature detection, which needs `std`;
// NEON and wasm simd128 are compile-time targets and need nothing.
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
mod avx2;
mod neon;
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
mod ssse3;
mod wasm;

/// The x86 kernel rung in use: what the CPU offers, capped by the knob.
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
mod level {
    use core::sync::atomic::{AtomicU8, Ordering};

    pub const SCALAR: u8 = 0;
    pub const SSSE3: u8 = 1;
    pub const AVX2: u8 = 2;
    const UNSET: u8 = u8::MAX;

    static LEVEL: AtomicU8 = AtomicU8::new(UNSET);

    #[inline]
    pub fn get() -> u8 {
        match LEVEL.load(Ordering::Relaxed) {
            UNSET => init(),
            l => l,
        }
    }

    fn detected() -> u8 {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("ssse3") {
            AVX2
        } else if std::is_x86_feature_detected!("ssse3") {
            SSSE3
        } else {
            SCALAR
        }
    }

    #[cold]
    fn init() -> u8 {
        let cap = match crate::knob("RUSTY_JPEG_DECODE_ISA").as_deref() {
            Some("scalar") => SCALAR,
            Some("ssse3") => SSSE3,
            _ => AVX2,
        };
        let l = detected().min(cap);
        // Only the first initialiser stores, so a forced rung (test only) is
        // never overwritten by a racing first call.
        match LEVEL.compare_exchange(UNSET, l, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => l,
            Err(current) => current,
        }
    }

    /// Force a rung for the whole-image oracle test. Process-wide, which is
    /// safe precisely because of the contract: every rung decodes to the same
    /// bytes, so a concurrent test cannot tell which one it got.
    #[cfg(test)]
    pub fn force(l: u8) {
        get();
        LEVEL.store(detected().min(l), Ordering::Relaxed);
    }
}

/// Arch-specific implementation of YCbCr conversion. Returns the number of pixels that were
/// converted.
#[allow(clippy::type_complexity)]
pub fn get_color_convert_line_ycbcr() -> Option<unsafe fn(&[u8], &[u8], &[u8], &mut [u8]) -> usize>
{
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        match level::get() {
            level::AVX2 => return Some(avx2::color_convert_line_ycbcr),
            level::SSSE3 => return Some(ssse3::color_convert_line_ycbcr),
            _ => {}
        }
    }
    // Runtime detection is not needed on aarch64.
    #[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
    {
        return Some(neon::color_convert_line_ycbcr);
    }
    #[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
    {
        return Some(wasm::color_convert_line_ycbcr);
    }
    #[allow(unreachable_code)]
    None
}

/// Arch-specific interior of a 4:2:0 fancy-upsampled row (see
/// `avx2::upsample_h2v2_pairs`). Returns the number of sample pairs written.
#[allow(clippy::type_complexity)]
pub fn get_upsample_h2v2_pairs() -> Option<unsafe fn(&[u8], &[u8], &mut [u8]) -> usize> {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if level::get() >= level::AVX2 {
            return Some(avx2::upsample_h2v2_pairs);
        }
    }
    #[allow(unreachable_code)]
    None
}

/// Arch-specific interior of a 4:2:2 fancy-upsampled row (see
/// `avx2::upsample_h2v1_windows`). Returns the number of windows written.
#[allow(clippy::type_complexity)]
pub fn get_upsample_h2v1_windows() -> Option<unsafe fn(&[u8], &mut [u8]) -> usize> {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if level::get() >= level::AVX2 {
            return Some(avx2::upsample_h2v1_windows);
        }
    }
    #[allow(unreachable_code)]
    None
}

/// Arch-specific implementation of 8x8 IDCT.
#[allow(clippy::type_complexity)]
pub fn get_dequantize_and_idct_block_8x8(
) -> Option<unsafe fn(&[i16; 64], &[u16; 64], usize, &mut [u8])> {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if level::get() >= level::SSSE3 {
            return Some(ssse3::dequantize_and_idct_block_8x8);
        }
    }
    // Runtime detection is not needed on aarch64.
    #[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
    {
        return Some(neon::dequantize_and_idct_block_8x8);
    }
    #[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
    {
        return Some(wasm::dequantize_and_idct_block_8x8);
    }
    #[allow(unreachable_code)]
    None
}

/// Arch-specific IDCT that does **two** 8x8 blocks per call.
///
/// Only offered when the pair actually pays: the caller must have two blocks
/// that both need a full transform (see `append_row_immediate`).
#[allow(clippy::type_complexity)]
pub fn get_dequantize_and_idct_block_8x8_pair(
) -> Option<unsafe fn(&[i16; 64], &[i16; 64], &[u16; 64], usize, &mut [u8], usize)> {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if level::get() >= level::AVX2 {
            return Some(avx2::dequantize_and_idct_block_8x8_pair);
        }
    }
    #[allow(unreachable_code)]
    None
}

/// The scalar IDCT, for a kernel handing over a block whose values do not fit
/// its 16-bit lanes (see `idct::simd_k`). Out of line and cold: real images
/// never get here.
#[cold]
#[inline(never)]
#[allow(dead_code)]
fn idct_fallback(
    coefficients: &[i16; 64],
    quantization_table: &[u16; 64],
    output_linestride: usize,
    output: &mut [u8],
) {
    crate::decode::idct::dequantize_and_idct_block_8x8_scalar(
        coefficients,
        quantization_table,
        output_linestride,
        output,
    );
}

#[cfg(test)]
type IdctFn = unsafe fn(&[i16; 64], &[u16; 64], usize, &mut [u8]);
#[cfg(test)]
type ColorFn = unsafe fn(&[u8], &[u8], &[u8], &mut [u8]) -> usize;

/// Every single-block IDCT kernel this CPU can execute, by name.
#[cfg(test)]
#[allow(clippy::vec_init_then_push)]
pub(crate) fn all_idct_kernels() -> alloc::vec::Vec<(&'static str, IdctFn)> {
    #[allow(unused_mut)]
    let mut v: alloc::vec::Vec<(&'static str, IdctFn)> = alloc::vec::Vec::new();
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    if std::is_x86_feature_detected!("ssse3") {
        v.push(("ssse3", ssse3::dequantize_and_idct_block_8x8));
    }
    #[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
    v.push(("neon", neon::dequantize_and_idct_block_8x8));
    #[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
    v.push(("wasm simd128", wasm::dequantize_and_idct_block_8x8));
    v
}

/// Every YCbCr->RGB kernel this CPU can execute, by name.
#[cfg(test)]
#[allow(clippy::vec_init_then_push)]
pub(crate) fn all_color_convert_kernels() -> alloc::vec::Vec<(&'static str, ColorFn)> {
    #[allow(unused_mut)]
    let mut v: alloc::vec::Vec<(&'static str, ColorFn)> = alloc::vec::Vec::new();
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if std::is_x86_feature_detected!("ssse3") {
            v.push(("ssse3", ssse3::color_convert_line_ycbcr));
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("ssse3") {
            v.push(("avx2", avx2::color_convert_line_ycbcr));
        }
    }
    #[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
    v.push(("neon", neon::color_convert_line_ycbcr));
    #[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
    v.push(("wasm simd128", wasm::color_convert_line_ycbcr));
    v
}

#[cfg(test)]
mod idct_tests {
    use crate::decode::idct::dequantize_and_idct_block_8x8_scalar as scalar;
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    /// xorshift64: coverage of sign, magnitude and saturation, not statistics.
    pub(super) struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Log-uniform magnitude below `2^bits`, random sign.
        pub fn mag(&mut self, bits: u32) -> i32 {
            let b = (self.next() % (u64::from(bits) + 1)) as u32;
            let v = (self.next() & ((1u64 << b) - 1)) as i32;
            if self.next() & 1 == 0 {
                v
            } else {
                -v
            }
        }
    }

    /// One block of a test family. The families straddle the kernels'
    /// fast-path boundary (dequantized values and first-pass outputs that just
    /// fit, or just miss, `i16`) as well as covering the realistic and the
    /// absurd.
    pub(super) fn block(rng: &mut Rng, family: u32) -> ([i16; 64], [u16; 64]) {
        let mut c = [0i16; 64];
        let mut q = [0u16; 64];
        match family {
            // Realistic: 8-bit tables, magnitudes falling off with frequency,
            // most high-frequency coefficients zero.
            0 => {
                for i in 0..64 {
                    q[i] = (rng.next() % 99 + 1) as u16;
                    let uv = (i % 8 + i / 8) as u32;
                    if rng.next() % 8 >= u64::from(uv / 2) {
                        c[i] = rng.mag(10u32.saturating_sub(uv / 2)) as i16;
                    }
                }
                c[0] = rng.mag(11) as i16;
            }
            // Uniform i16 with 8-bit tables: mostly over the boundary.
            1 => {
                for i in 0..64 {
                    c[i] = rng.next() as i16;
                    q[i] = (rng.next() % 255 + 1) as u16;
                }
            }
            // Log-uniform everything, 16-bit tables: every regime.
            2 => {
                for i in 0..64 {
                    c[i] = rng.mag(15).clamp(-32768, 32767) as i16;
                    q[i] = rng.mag(16).unsigned_abs().min(65535) as u16;
                }
            }
            // Near the first-pass boundary: q = 1, coefficients scaled so the
            // first-pass outputs land around +-32768.
            3 => {
                let scale = 1 + (rng.next() % 4096) as i32;
                for i in 0..64 {
                    q[i] = 1;
                    c[i] = (rng.mag(13) * scale / 4096).clamp(-8300, 8300) as i16;
                }
            }
            // Near the dequantization boundary: products around +-32768.
            4 => {
                for i in 0..64 {
                    let qv = 1 + (rng.next() % 300) as i32;
                    q[i] = qv as u16;
                    let target = 32768 + rng.mag(6);
                    let sign = if rng.next() & 1 == 0 { 1 } else { -1 };
                    c[i] = (sign * target / qv).clamp(-32768, 32767) as i16;
                    if !rng.next().is_multiple_of(3) {
                        c[i] /= 64;
                    }
                }
            }
            // Table entries >= 0x8000 over coefficients of -1, 0 and 1: the
            // product's low half alone looks like a small in-range value
            // (1 * 0xFFFF reads as -1), so only the table check catches it.
            5 => {
                for i in 0..64 {
                    q[i] = (rng.next() % 40 + 1) as u16;
                    c[i] = rng.mag(5) as i16;
                }
                for _ in 0..1 + rng.next() % 3 {
                    let i = (rng.next() % 64) as usize;
                    q[i] = 0x8000 | rng.next() as u16;
                    c[i] = (rng.next() % 3) as i16 - 1;
                }
            }
            // Worst cases for the first pass: columns whose coefficients carry
            // the sign pattern that maximises one output, at magnitudes around
            // `FAST_MAX_IN` (the bound under which no check is made), so the
            // outputs land just inside or just outside i16.
            6 => {
                use crate::decode::idct::simd_k::{coef, FAST_MAX_IN};
                q = [1; 64];
                for col in 0..8 {
                    if rng.next() & 1 == 0 {
                        let n = (rng.next() % 8) as usize;
                        let m = FAST_MAX_IN - 40 + (rng.next() % 440) as i32;
                        let sign = if rng.next() & 1 == 0 { 1 } else { -1 };
                        for k in 0..8 {
                            c[col + 8 * k] = (sign * m * coef(n, k).signum()) as i16;
                        }
                    } else {
                        for k in 0..8 {
                            c[col + 8 * k] = rng.mag(10) as i16;
                        }
                    }
                }
            }
            // Sparse: one column DC-only (the scalar's column shortcut), the
            // rest random; rows 4-7 zero half the time.
            _ => {
                for i in 0..64 {
                    q[i] = (rng.next() % 64 + 1) as u16;
                    c[i] = rng.mag(9) as i16;
                }
                let col = (rng.next() % 8) as usize;
                for r in 1..8 {
                    c[col + 8 * r] = 0;
                }
                if rng.next() & 1 == 0 {
                    c[32..].fill(0);
                }
            }
        }
        (c, q)
    }

    pub(super) fn extremes() -> Vec<([i16; 64], [u16; 64])> {
        let mut v = Vec::new();
        for &cv in &[i16::MAX, i16::MIN, 1, -1, 0x4000, -0x4000, 2047, -2048] {
            for &qv in &[1u16, 2, 16, 255, 256, 0x7FFF, 0x8000, u16::MAX] {
                v.push(([cv; 64], [qv; 64]));
                let mut alt = [cv; 64];
                for (i, a) in alt.iter_mut().enumerate() {
                    if (i ^ (i >> 3)) & 1 == 1 {
                        *a = a.wrapping_neg();
                    }
                }
                v.push((alt, [qv; 64]));
                // DC-only: the dispatcher never sends one, but a kernel must
                // still be right on it.
                let mut dc = [0i16; 64];
                dc[0] = cv;
                v.push((dc, [qv; 64]));
            }
        }
        v
    }

    fn run_single(name: &str, kernel: super::IdctFn, c: &[i16; 64], q: &[u16; 64], what: &str) {
        // Stride wider than the block, with guard bytes: a kernel must write
        // exactly 8 bytes per row and nothing else.
        let stride = 19;
        let len = stride * 7 + 8 + 5;
        let mut want = vec![0xA5u8; len];
        let mut got = vec![0xA5u8; len];
        scalar(c, q, stride, &mut want);
        unsafe { kernel(c, q, stride, &mut got) };
        assert_eq!(
            got, want,
            "{name} diverged from scalar on {what}: {c:?} {q:?}"
        );
    }

    /// Every single-block IDCT kernel against the scalar oracle.
    #[test]
    fn idct_kernels_match_scalar() {
        let rounds = if cfg!(debug_assertions) {
            30_000
        } else {
            300_000
        };
        for (name, kernel) in super::all_idct_kernels() {
            for (c, q) in extremes() {
                run_single(name, kernel, &c, &q, "an extreme block");
            }
            let mut rng = Rng(0x1234_5678_9abc_def0);
            for round in 0..rounds {
                let family = round % 8;
                let (c, q) = block(&mut rng, family);
                run_single(name, kernel, &c, &q, &format!("family {family}"));
            }
        }
    }

    /// The AVX2 two-block kernel against the scalar oracle, block for block.
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_pair_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("ssse3") {
            return;
        }
        let check = |ca: &[i16; 64], cb: &[i16; 64], q: &[u16; 64], what: &str| {
            // One plane, two horizontally adjacent blocks, exactly as the
            // decoder lays them out, plus guard bytes.
            let stride = 24usize;
            let offset_b = 8usize;
            let len = stride * 8 + 32;
            let mut want = vec![0x3Cu8; len];
            let mut got = vec![0x3Cu8; len];
            scalar(ca, q, stride, &mut want);
            scalar(cb, q, stride, &mut want[offset_b..]);
            unsafe {
                super::avx2::dequantize_and_idct_block_8x8_pair(
                    ca, cb, q, stride, &mut got, offset_b,
                );
            }
            assert_eq!(got, want, "pair diverged from scalar on {what}");
        };
        let mut rng = Rng(0x0fed_cba9_8765_4321);
        for (c, q) in &extremes() {
            let (r, _) = block(&mut rng, 0);
            check(c, &r, q, "extreme A");
            check(&r, c, q, "extreme B");
        }
        let rounds = if cfg!(debug_assertions) {
            15_000
        } else {
            150_000
        };
        for round in 0..rounds {
            // Mix families across the two lanes: one block on the fast path
            // and its neighbour off it must both still come out exact.
            let (ca, q) = block(&mut rng, round % 8);
            let (cb, _) = block(&mut rng, (round / 8) % 8);
            check(&ca, &cb, &q, &format!("round {round}"));
        }
    }
}

/// Whole images: the default (vector) decode must equal the forced-scalar
/// decode byte for byte, on every x86 rung this CPU has.
#[cfg(all(
    test,
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod whole_image_tests {
    use crate::decode::Decoder;
    use crate::encode::{ColorType, Encoder, SamplingFactor};
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    fn photo(w: u16, h: u16, seed: u32) -> Vec<u8> {
        // Gradients, texture and hard edges: every frequency band.
        let mut px = Vec::with_capacity(w as usize * h as usize * 3);
        let mut s = seed | 1;
        for y in 0..u32::from(h) {
            for x in 0..u32::from(w) {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let n = (s >> 24) as i32 - 128;
                let edge = if (x / 37 + y / 23) % 2 == 0 { 60 } else { -60 };
                let r = (x * 255 / u32::from(w)) as i32 + n / 3 + edge;
                let g = (y * 255 / u32::from(h)) as i32 - n / 4;
                let b = ((x + y) * 128 / (u32::from(w) + u32::from(h))) as i32 + n / 2 - edge;
                px.extend([r, g, b].map(|v| v.clamp(0, 255) as u8));
            }
        }
        px
    }

    fn corpus() -> Vec<(String, Vec<u8>)> {
        let mut v = Vec::new();
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let mut stack = vec![std::path::PathBuf::from(dir)];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "jpg" || x == "jpeg") {
                    v.push((p.display().to_string(), std::fs::read(&p).unwrap()));
                }
            }
        }
        let sampling = [
            ("4:2:0", SamplingFactor::R_4_2_0),
            ("4:2:2", SamplingFactor::R_4_2_2),
            ("4:4:4", SamplingFactor::R_4_4_4),
        ];
        for (i, &(w, h)) in [(256u16, 192u16), (203, 117), (17, 9)].iter().enumerate() {
            let rgb = photo(w, h, 0x9e37_79b9 ^ i as u32);
            let grey: Vec<u8> = rgb.chunks_exact(3).map(|p| p[1]).collect();
            for &q in &[30u8, 75, 95, 100] {
                for (label, s) in sampling {
                    for progressive in [false, true] {
                        let mut out = Vec::new();
                        let mut enc = Encoder::new(&mut out, q);
                        enc.set_sampling_factor(s);
                        enc.set_progressive(progressive);
                        enc.encode(&rgb, w, h, ColorType::Rgb).unwrap();
                        v.push((
                            format!("synthetic {w}x{h} q{q} {label} p{progressive}"),
                            out,
                        ));
                    }
                }
                let mut out = Vec::new();
                Encoder::new(&mut out, q)
                    .encode(&grey, w, h, ColorType::Luma)
                    .unwrap();
                v.push((format!("synthetic {w}x{h} q{q} grey"), out));
            }
        }
        v
    }

    type Decoded = Option<(Vec<u8>, Vec<Vec<u8>>, Vec<u8>)>;

    fn decode_all(corpus: &[(String, Vec<u8>)]) -> Vec<Decoded> {
        corpus
            .iter()
            .map(|(_, bytes)| {
                let packed = Decoder::new(&bytes[..]).decode().ok()?;
                let planar = Decoder::new(&bytes[..]).decode_planar().ok()?;
                // Single-threaded too: it takes the fused per-block path.
                let mut d = Decoder::new(&bytes[..]);
                d.set_single_threaded(true);
                let single = d.decode().ok()?;
                Some((packed, planar.into_planes(), single))
            })
            .collect()
    }

    #[test]
    fn simd_decode_matches_scalar_decode() {
        let corpus = corpus();
        assert!(corpus.len() > 60, "corpus too small: {}", corpus.len());
        super::level::force(super::level::SCALAR);
        let want = decode_all(&corpus);
        assert!(want.iter().filter(|d| d.is_some()).count() > 60);
        for (rung, name) in [(super::level::SSSE3, "ssse3"), (super::level::AVX2, "avx2")] {
            super::level::force(rung);
            let got = decode_all(&corpus);
            for (i, (label, _)) in corpus.iter().enumerate() {
                assert!(
                    got[i] == want[i],
                    "{name} decode differs from scalar on {label}"
                );
            }
        }
        super::level::force(super::level::AVX2);
    }
}
