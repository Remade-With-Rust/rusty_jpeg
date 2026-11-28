//! 256-bit x86 kernels: the two-block IDCT, YCbCr->RGB and the fancy
//! upsampler interiors. Every one is a bit-exact twin of its scalar code.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::decode::decoder::ycc_k;
use crate::decode::idct::simd_k;

/// A `pmaddwd` operand: the pair `k = [even, odd]` in every 32-bit lane.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pair(k: [i16; 2]) -> __m256i {
    _mm256_set1_epi32((i32::from(k[1]) << 16) | i32::from(k[0] as u16))
}

/// Half of one 1-D pass, as `ssse3::pass_half`, with block A in the low
/// 128-bit lane and block B in the high one. Every operation is per lane.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pass_half(
    s04: __m256i,
    s26: __m256i,
    s13: __m256i,
    s57: __m256i,
    bias: __m256i,
) -> [__m256i; 8] {
    let t2 = _mm256_madd_epi16(s26, pair(simd_k::EVEN_T2));
    let t3 = _mm256_madd_epi16(s26, pair(simd_k::EVEN_T3));
    let t0 = _mm256_add_epi32(
        _mm256_madd_epi16(s04, pair([simd_k::FSH, simd_k::FSH])),
        bias,
    );
    let t1 = _mm256_add_epi32(
        _mm256_madd_epi16(s04, pair([simd_k::FSH, -simd_k::FSH])),
        bias,
    );
    let x = [
        _mm256_add_epi32(t0, t3),
        _mm256_add_epi32(t1, t2),
        _mm256_sub_epi32(t1, t2),
        _mm256_sub_epi32(t0, t3),
    ];
    let mut t = [_mm256_setzero_si256(); 4];
    for (n, c) in simd_k::ODD.iter().enumerate() {
        t[n] = _mm256_add_epi32(
            _mm256_madd_epi16(s13, pair([c[0], c[1]])),
            _mm256_madd_epi16(s57, pair([c[2], c[3]])),
        );
    }
    let mut out = [_mm256_setzero_si256(); 8];
    for n in 0..4 {
        out[n] = _mm256_add_epi32(x[n], t[3 - n]);
        out[7 - n] = _mm256_sub_epi32(x[n], t[3 - n]);
    }
    out
}

/// One 1-D pass. `unpack*` interleave within each 128-bit lane, so lane 0
/// carries block A's lanes 0-3 / 4-7 and lane 1 block B's.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pass(s: &[__m256i; 8], bias: i32) -> ([__m256i; 8], [__m256i; 8]) {
    let bias = _mm256_set1_epi32(bias);
    let lo = pass_half(
        _mm256_unpacklo_epi16(s[0], s[4]),
        _mm256_unpacklo_epi16(s[2], s[6]),
        _mm256_unpacklo_epi16(s[1], s[3]),
        _mm256_unpacklo_epi16(s[5], s[7]),
        bias,
    );
    let hi = pass_half(
        _mm256_unpackhi_epi16(s[0], s[4]),
        _mm256_unpackhi_epi16(s[2], s[6]),
        _mm256_unpackhi_epi16(s[1], s[3]),
        _mm256_unpackhi_epi16(s[5], s[7]),
        bias,
    );
    (lo, hi)
}

/// Transpose both blocks. The unpack instructions never cross the 128-bit lane
/// boundary, so this is two independent 8x8 transposes for the price of one.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn transpose8(data: &mut [__m256i; 8]) {
    let d01l = _mm256_unpacklo_epi16(data[0], data[1]);
    let d23l = _mm256_unpacklo_epi16(data[2], data[3]);
    let d45l = _mm256_unpacklo_epi16(data[4], data[5]);
    let d67l = _mm256_unpacklo_epi16(data[6], data[7]);
    let d01h = _mm256_unpackhi_epi16(data[0], data[1]);
    let d23h = _mm256_unpackhi_epi16(data[2], data[3]);
    let d45h = _mm256_unpackhi_epi16(data[4], data[5]);
    let d67h = _mm256_unpackhi_epi16(data[6], data[7]);

    let d0123ll = _mm256_unpacklo_epi32(d01l, d23l);
    let d0123lh = _mm256_unpackhi_epi32(d01l, d23l);
    let d4567ll = _mm256_unpacklo_epi32(d45l, d67l);
    let d4567lh = _mm256_unpackhi_epi32(d45l, d67l);
    let d0123hl = _mm256_unpacklo_epi32(d01h, d23h);
    let d0123hh = _mm256_unpackhi_epi32(d01h, d23h);
    let d4567hl = _mm256_unpacklo_epi32(d45h, d67h);
    let d4567hh = _mm256_unpackhi_epi32(d45h, d67h);

    data[0] = _mm256_unpacklo_epi64(d0123ll, d4567ll);
    data[1] = _mm256_unpackhi_epi64(d0123ll, d4567ll);
    data[2] = _mm256_unpacklo_epi64(d0123lh, d4567lh);
    data[3] = _mm256_unpackhi_epi64(d0123lh, d4567lh);
    data[4] = _mm256_unpacklo_epi64(d0123hl, d4567hl);
    data[5] = _mm256_unpackhi_epi64(d0123hl, d4567hl);
    data[6] = _mm256_unpacklo_epi64(d0123hh, d4567hh);
    data[7] = _mm256_unpackhi_epi64(d0123hh, d4567hh);
}

/// Dequantize and inverse-transform two 8x8 blocks in one instruction stream:
/// the scalar kernel, bit for bit, on each.
///
/// This is `ssse3::dequantize_and_idct_block_8x8` widened to 256 bits, block A
/// in the low 128-bit lane and block B in the high one; every operation is
/// lane-independent (`madd`, 32-bit arithmetic, `unpack*`, `packs`/`packus`
/// all work per lane), so each block sees exactly the 128-bit sequence. If
/// either block leaves the 16-bit fast path, both go to the scalar kernel.
///
/// Both blocks belong to the SAME component, so they share a quantization
/// table and a line stride, and both write into one plane. Taking a single
/// output slice plus block B's relative origin avoids handing out two
/// overlapping `&mut` slices -- the blocks interleave row by row without ever
/// touching the same byte, which is true but not something the borrow checker
/// can be told.
///
/// # Safety
///
/// The CPU must support AVX2. `output` must hold
/// `output_offset_b + output_linestride * 7 + 8` bytes (asserted), and the
/// two blocks' rows must not overlap (`output_offset_b >= 8`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn dequantize_and_idct_block_8x8_pair(
    coefficients_a: &[i16; 64],
    coefficients_b: &[i16; 64],
    quantization_table: &[u16; 64],
    output_linestride: usize,
    output: &mut [u8],
    output_offset_b: usize,
) {
    // Last write is at offset_b + linestride * 7 + 7.
    assert!(
        output.len()
            > output_offset_b
                .checked_add(output_linestride.checked_mul(7).unwrap())
                .unwrap()
                .checked_add(7)
                .unwrap()
    );

    let zero = _mm256_setzero_si256();
    let mut s = [zero; 8];
    let mut bad = zero;
    let mut qor = zero;
    for (i, row) in s.iter_mut().enumerate() {
        // SAFETY: rows 0..8 of 8 elements lie inside the 64-element arrays.
        // Lane 0 <- block A row i, lane 1 <- block B row i.
        let c = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(
                coefficients_a.as_ptr().add(i * 8) as *const __m128i
            )),
            _mm_loadu_si128(coefficients_b.as_ptr().add(i * 8) as *const __m128i),
        );
        // Same table in both lanes.
        let q = _mm256_broadcastsi128_si256(_mm_loadu_si128(
            quantization_table.as_ptr().add(i * 8) as *const __m128i,
        ));
        let lo = _mm256_mullo_epi16(c, q);
        let hi = _mm256_mulhi_epi16(c, q);
        bad = _mm256_or_si256(bad, _mm256_xor_si256(hi, _mm256_srai_epi16::<15>(lo)));
        qor = _mm256_or_si256(qor, q);
        *row = lo;
    }
    bad = _mm256_or_si256(bad, _mm256_and_si256(qor, _mm256_set1_epi16(i16::MIN)));

    let (lo, hi) = pass(&s, simd_k::PASS1_BIAS);
    let k8000 = _mm256_set1_epi32(0x8000);
    let mut range = zero;
    let mut t = [zero; 8];
    for n in 0..8 {
        let a = _mm256_srai_epi32::<{ simd_k::PASS1_SHIFT }>(lo[n]);
        let b = _mm256_srai_epi32::<{ simd_k::PASS1_SHIFT }>(hi[n]);
        range = _mm256_or_si256(
            range,
            _mm256_or_si256(_mm256_add_epi32(a, k8000), _mm256_add_epi32(b, k8000)),
        );
        // `packs` is per lane: [A lanes 0-7 | B lanes 0-7].
        t[n] = _mm256_packs_epi32(a, b);
    }
    bad = _mm256_or_si256(bad, _mm256_srli_epi32::<16>(range));
    if _mm256_movemask_epi8(_mm256_cmpeq_epi8(bad, zero)) != -1 {
        super::idct_fallback(
            coefficients_a,
            quantization_table,
            output_linestride,
            output,
        );
        super::idct_fallback(
            coefficients_b,
            quantization_table,
            output_linestride,
            &mut output[output_offset_b..],
        );
        return;
    }

    transpose8(&mut t);
    let (lo, hi) = pass(&t, simd_k::PASS2_BIAS);
    let mut o = [zero; 8];
    for n in 0..8 {
        o[n] = _mm256_packs_epi32(
            _mm256_srai_epi32::<{ simd_k::PASS2_SHIFT }>(lo[n]),
            _mm256_srai_epi32::<{ simd_k::PASS2_SHIFT }>(hi[n]),
        );
    }
    transpose8(&mut o);

    for (i, row) in o.iter().enumerate() {
        // packus is per lane: lane 0's low 8 bytes are block A's row, lane 1's
        // low 8 bytes are block B's.
        let packed = _mm256_packus_epi16(*row, *row);
        // SAFETY: the assert above bounds both 8-byte stores.
        let base = output.as_mut_ptr().add(output_linestride * i);
        _mm_storel_epi64(base as *mut __m128i, _mm256_castsi256_si128(packed));
        _mm_storel_epi64(
            base.add(output_offset_b) as *mut __m128i,
            _mm256_extracti128_si256::<1>(packed),
        );
    }
}

/// `pshufb` masks that interleave 16 R, 16 G and 16 B bytes into 48 bytes of
/// RGB: `RGB_MASKS[part][channel]` selects, for output bytes
/// `16*part..16*part+16`, the source byte of `channel` (or zero, `0x80`).
const RGB_MASKS: [[[i8; 16]; 3]; 3] = [
    [rgb_mask(0, 0), rgb_mask(0, 1), rgb_mask(0, 2)],
    [rgb_mask(1, 0), rgb_mask(1, 1), rgb_mask(1, 2)],
    [rgb_mask(2, 0), rgb_mask(2, 1), rgb_mask(2, 2)],
];

const fn rgb_mask(part: usize, channel: usize) -> [i8; 16] {
    let mut mask = [-128i8; 16];
    let mut k = 0;
    while k < 16 {
        let byte = part * 16 + k;
        if byte % 3 == channel {
            mask[k] = (byte / 3) as i8;
        }
        k += 1;
    }
    mask
}

/// The three channel offsets `(HALF + K * c) >> 20` for 16 pixels, in 16-bit
/// lanes in pixel order (see `ssse3::color_convert_line_ycbcr`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn ycc_offsets(cbv: __m256i, crv: __m256i) -> [__m256i; 3] {
    let half = _mm256_set1_epi32(ycc_k::HALF);
    // Per 128-bit lane: lo = pixels 0-3 | 8-11, hi = 4-7 | 12-15; `packs`
    // of (lo, hi) puts them back in order.
    let cb_lo = _mm256_unpacklo_epi16(_mm256_slli_epi16::<8>(cbv), cbv);
    let cb_hi = _mm256_unpackhi_epi16(_mm256_slli_epi16::<8>(cbv), cbv);
    let cr_lo = _mm256_unpacklo_epi16(_mm256_slli_epi16::<8>(crv), crv);
    let cr_hi = _mm256_unpackhi_epi16(_mm256_slli_epi16::<8>(crv), crv);
    let (k_r, k_gb, k_gr, k_b) = (
        pair(ycc_k::split(ycc_k::CR_R)),
        pair(ycc_k::split(ycc_k::CB_G)),
        pair(ycc_k::split(ycc_k::CR_G)),
        pair(ycc_k::split(ycc_k::CB_B)),
    );
    let offset = |lo: __m256i, hi: __m256i| {
        _mm256_packs_epi32(
            _mm256_srai_epi32::<{ ycc_k::SHIFT }>(_mm256_add_epi32(lo, half)),
            _mm256_srai_epi32::<{ ycc_k::SHIFT }>(_mm256_add_epi32(hi, half)),
        )
    };
    [
        offset(_mm256_madd_epi16(cr_lo, k_r), _mm256_madd_epi16(cr_hi, k_r)),
        offset(
            _mm256_add_epi32(
                _mm256_madd_epi16(cb_lo, k_gb),
                _mm256_madd_epi16(cr_lo, k_gr),
            ),
            _mm256_add_epi32(
                _mm256_madd_epi16(cb_hi, k_gb),
                _mm256_madd_epi16(cr_hi, k_gr),
            ),
        ),
        offset(_mm256_madd_epi16(cb_lo, k_b), _mm256_madd_epi16(cb_hi, k_b)),
    ]
}

/// YCbCr -> RGB, 16 pixels per step: `ycbcr_to_rgb`, bit for bit, by the
/// arithmetic of `ssse3::color_convert_line_ycbcr` widened to 256 bits. An
/// 8-pixel remainder goes to the SSSE3 kernel; returns the pixels converted.
///
/// # Safety
///
/// The CPU must support AVX2 and SSSE3. Lengths are asserted.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn color_convert_line_ycbcr(y: &[u8], cb: &[u8], cr: &[u8], output: &mut [u8]) -> usize {
    assert!(output.len().is_multiple_of(3));
    let num = output.len() / 3;
    assert!(num <= y.len());
    assert!(num <= cb.len());
    assert!(num <= cr.len());

    let c128 = _mm256_set1_epi16(128);
    let mask = |part: usize, channel: usize| {
        _mm_loadu_si128(RGB_MASKS[part][channel].as_ptr() as *const __m128i)
    };
    let masks = [
        [mask(0, 0), mask(0, 1), mask(0, 2)],
        [mask(1, 0), mask(1, 1), mask(1, 2)],
        [mask(2, 0), mask(2, 1), mask(2, 2)],
    ];

    let mut i = 0;
    while i + 16 <= num {
        // SAFETY: i + 16 <= num, which the asserts bound by every slice.
        let widen =
            |p: &[u8]| _mm256_cvtepu8_epi16(_mm_loadu_si128(p.as_ptr().add(i) as *const __m128i));
        let yv = widen(y);
        let [dr, dg, db] = ycc_offsets(
            _mm256_sub_epi16(widen(cb), c128),
            _mm256_sub_epi16(widen(cr), c128),
        );
        let r = _mm256_add_epi16(yv, dr);
        let g = _mm256_add_epi16(yv, dg);
        let b = _mm256_add_epi16(yv, db);

        // packus works per 128-bit lane: [r0-7 g0-7 | r8-15 g8-15]. Reorder the
        // quadwords to [r0-7 r8-15 | g0-7 g8-15].
        let rg = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(r, g));
        let bb = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(b, b));
        let channels = [
            _mm256_castsi256_si128(rg),
            _mm256_extracti128_si256::<1>(rg),
            _mm256_castsi256_si128(bb),
        ];

        // SAFETY: bytes 3 * i .. 3 * i + 48 lie inside `output` (3 * num).
        let out = output.as_mut_ptr().add(3 * i);
        for (part, part_masks) in masks.iter().enumerate() {
            let v = _mm_or_si128(
                _mm_shuffle_epi8(channels[0], part_masks[0]),
                _mm_or_si128(
                    _mm_shuffle_epi8(channels[1], part_masks[1]),
                    _mm_shuffle_epi8(channels[2], part_masks[2]),
                ),
            );
            _mm_storeu_si128(out.add(16 * part) as *mut __m128i, v);
        }
        i += 16;
    }

    if num - i >= 8 {
        i += super::ssse3::color_convert_line_ycbcr(
            &y[i..],
            &cb[i..],
            &cr[i..],
            &mut output[3 * i..],
        );
    }
    i
}

/// The interior of a 4:2:0 "fancy" upsampled row, 16 sample pairs per step.
///
/// With column sums `t[i] = 3*near[i] + far[i]`, pair `k` writes
/// `out[2k] = (3*t[k] + t[k+1] + 8) >> 4` and
/// `out[2k+1] = (3*t[k+1] + t[k] + 8) >> 4` -- where `out` starts at output
/// byte 1 of the row; bytes 0 and `2w-1` are the caller's. Integer maths in
/// 16-bit lanes (at most 4088, no overflow), so the result is bit-identical
/// to the scalar loop by construction; `h2v2_matches_reference` in the
/// upsampler checks it. The two results of a pair are packed as
/// `lo | hi << 8`, which in little-endian memory IS the interleaved byte
/// order, so a pair costs no shuffle.
///
/// Returns how many pairs it wrote (a multiple of 16); the caller finishes the
/// rest.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn upsample_h2v2_pairs(near: &[u8], far: &[u8], out: &mut [u8]) -> usize {
    let width = near.len().min(far.len());
    // Pairs k..k+16 read samples k..k+17.
    let pairs = width.saturating_sub(1).min(out.len() / 2);
    let eight = _mm256_set1_epi16(8);
    let three = |a: __m256i| _mm256_add_epi16(a, _mm256_slli_epi16(a, 1));
    let column = |n: *const u8, f: *const u8| {
        let n = _mm256_cvtepu8_epi16(_mm_loadu_si128(n as *const __m128i));
        let f = _mm256_cvtepu8_epi16(_mm_loadu_si128(f as *const __m128i));
        _mm256_add_epi16(three(n), f)
    };
    let step = |k: usize, out: &mut [u8]| {
        let (np, fp) = (near.as_ptr().add(k), far.as_ptr().add(k));
        let t0 = column(np, fp);
        let t1 = column(np.add(1), fp.add(1));
        let lo = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(three(t0), t1), eight), 4);
        let hi = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(three(t1), t0), eight), 4);
        let packed = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 8));
        _mm256_storeu_si256(out.as_mut_ptr().add(2 * k) as *mut __m256i, packed);
    };
    let mut k = 0;
    // Two independent 16-pair groups per trip: the loop's own bookkeeping
    // (index, bound test, branch, addressing) was a sixth of a 16-pair step.
    while k + 32 <= pairs {
        step(k, out);
        step(k + 16, out);
        k += 32;
    }
    while k + 16 <= pairs {
        step(k, out);
        k += 16;
    }
    k
}

/// The interior of a 4:2:2 "fancy" upsampled row, 16 samples per step.
///
/// Window `j` is input samples `j, j+1, j+2` and writes output bytes `2j` and
/// `2j+1` of `out` -- which starts at output byte 2 of the row; the first
/// and last two bytes are the caller's:
/// `out[2j] = (3*in[j+1] + in[j] + 2) >> 2` and
/// `out[2j+1] = (3*in[j+1] + in[j+2] + 2) >> 2`. Integer maths in 16-bit
/// lanes (at most 1022, no overflow), so the result is bit-identical to the
/// scalar loop by construction; `h2v1_matches_reference` in the upsampler
/// checks it. The two results of a window are packed as `lo | hi << 8`, which
/// in little-endian memory is the interleaved byte order.
///
/// Returns how many windows it wrote (a multiple of 16); the caller finishes
/// the rest.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn upsample_h2v1_windows(input: &[u8], out: &mut [u8]) -> usize {
    // Windows j..j+16 read samples j..j+18.
    let windows = input.len().saturating_sub(2).min(out.len() / 2);
    let two = _mm256_set1_epi16(2);
    let widen = |p: *const u8| _mm256_cvtepu8_epi16(_mm_loadu_si128(p as *const __m128i));
    let mut j = 0;
    while j + 16 <= windows {
        let p = input.as_ptr().add(j);
        let (left, centre, right) = (widen(p), widen(p.add(1)), widen(p.add(2)));
        let sample = _mm256_add_epi16(_mm256_add_epi16(centre, _mm256_slli_epi16(centre, 1)), two);
        let lo = _mm256_srli_epi16(_mm256_add_epi16(sample, left), 2);
        let hi = _mm256_srli_epi16(_mm256_add_epi16(sample, right), 2);
        let packed = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 8));
        _mm256_storeu_si256(out.as_mut_ptr().add(2 * j) as *mut __m256i, packed);
        j += 16;
    }
    j
}
