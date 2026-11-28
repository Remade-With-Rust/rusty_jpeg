//! 128-bit x86 kernels: the single-block IDCT and YCbCr->RGB.
//!
//! Both are bit-exact twins of the scalar code (`idct.rs`,
//! `decoder.rs::ycbcr_to_rgb`). The arithmetic is SSE2 (`pmaddwd`, 32-bit adds
//! and shifts, saturating packs); SSSE3 is needed only for the `pshufb` RGB
//! interleave, and is what the dispatcher detects.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::decode::decoder::ycc_k;
use crate::decode::idct::simd_k;

/// A `pmaddwd` operand: the pair `k = [even, odd]` in every 32-bit lane, so
/// `madd(interleave(a, b), pair(k)) = a * k[0] + b * k[1]` per lane, exactly
/// (modulo 2^32, which is the scalar's own arithmetic).
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn pair(k: [i16; 2]) -> __m128i {
    _mm_set1_epi32((i32::from(k[1]) << 16) | i32::from(k[0] as u16))
}

/// Half of one 1-D pass: four lanes (columns, or rows on the second pass) of
/// the scalar `kernel`, from inputs interleaved as `(s0,s4)`, `(s2,s6)`,
/// `(s1,s3)` and `(s5,s7)`. Returns output `n` as `L_n(s) + bias` in `i32`,
/// exactly the scalar's value before its shift (see `idct::simd_k`).
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn pass_half(
    s04: __m128i,
    s26: __m128i,
    s13: __m128i,
    s57: __m128i,
    bias: __m128i,
) -> [__m128i; 8] {
    let t2 = _mm_madd_epi16(s26, pair(simd_k::EVEN_T2));
    let t3 = _mm_madd_epi16(s26, pair(simd_k::EVEN_T3));
    let t0 = _mm_add_epi32(_mm_madd_epi16(s04, pair([simd_k::FSH, simd_k::FSH])), bias);
    let t1 = _mm_add_epi32(_mm_madd_epi16(s04, pair([simd_k::FSH, -simd_k::FSH])), bias);
    let x = [
        _mm_add_epi32(t0, t3),
        _mm_add_epi32(t1, t2),
        _mm_sub_epi32(t1, t2),
        _mm_sub_epi32(t0, t3),
    ];
    let mut t = [_mm_setzero_si128(); 4];
    for (n, c) in simd_k::ODD.iter().enumerate() {
        t[n] = _mm_add_epi32(
            _mm_madd_epi16(s13, pair([c[0], c[1]])),
            _mm_madd_epi16(s57, pair([c[2], c[3]])),
        );
    }
    // The scalar's output order: out[n] = xs[n] + ts[3-n], out[7-n] = xs[n] - ts[3-n].
    let mut out = [_mm_setzero_si128(); 8];
    for n in 0..4 {
        out[n] = _mm_add_epi32(x[n], t[3 - n]);
        out[7 - n] = _mm_sub_epi32(x[n], t[3 - n]);
    }
    out
}

/// One 1-D pass over eight 16-bit lanes: `s[k]` holds input `k` for each lane.
/// Returns `(lanes 0-3, lanes 4-7)` of each output as `i32`.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn pass(s: &[__m128i; 8], bias: i32) -> ([__m128i; 8], [__m128i; 8]) {
    let bias = _mm_set1_epi32(bias);
    let lo = pass_half(
        _mm_unpacklo_epi16(s[0], s[4]),
        _mm_unpacklo_epi16(s[2], s[6]),
        _mm_unpacklo_epi16(s[1], s[3]),
        _mm_unpacklo_epi16(s[5], s[7]),
        bias,
    );
    let hi = pass_half(
        _mm_unpackhi_epi16(s[0], s[4]),
        _mm_unpackhi_epi16(s[2], s[6]),
        _mm_unpackhi_epi16(s[1], s[3]),
        _mm_unpackhi_epi16(s[5], s[7]),
        bias,
    );
    (lo, hi)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn transpose8(data: &mut [__m128i; 8]) {
    // Transpose a 8x8 matrix with a sequence of interleaving operations.
    // Naming: dABl contains elements from the *l*ower halves of vectors A and B, interleaved, i.e.
    // A0 B0 A1 B1 ...
    // dABCDll contains elements from the lower quarter (ll) of vectors A, B, C, D, interleaved -
    // A0 B0 C0 D0 A1 B1 C1 D1 ...
    let d01l = _mm_unpacklo_epi16(data[0], data[1]);
    let d23l = _mm_unpacklo_epi16(data[2], data[3]);
    let d45l = _mm_unpacklo_epi16(data[4], data[5]);
    let d67l = _mm_unpacklo_epi16(data[6], data[7]);
    let d01h = _mm_unpackhi_epi16(data[0], data[1]);
    let d23h = _mm_unpackhi_epi16(data[2], data[3]);
    let d45h = _mm_unpackhi_epi16(data[4], data[5]);
    let d67h = _mm_unpackhi_epi16(data[6], data[7]);
    // Operating on 32-bits will interleave *consecutive pairs* of 16-bit integers.
    let d0123ll = _mm_unpacklo_epi32(d01l, d23l);
    let d0123lh = _mm_unpackhi_epi32(d01l, d23l);
    let d4567ll = _mm_unpacklo_epi32(d45l, d67l);
    let d4567lh = _mm_unpackhi_epi32(d45l, d67l);
    let d0123hl = _mm_unpacklo_epi32(d01h, d23h);
    let d0123hh = _mm_unpackhi_epi32(d01h, d23h);
    let d4567hl = _mm_unpacklo_epi32(d45h, d67h);
    let d4567hh = _mm_unpackhi_epi32(d45h, d67h);
    // Operating on 64-bits will interleave *consecutive quadruples* of 16-bit integers.
    data[0] = _mm_unpacklo_epi64(d0123ll, d4567ll);
    data[1] = _mm_unpackhi_epi64(d0123ll, d4567ll);
    data[2] = _mm_unpacklo_epi64(d0123lh, d4567lh);
    data[3] = _mm_unpackhi_epi64(d0123lh, d4567lh);
    data[4] = _mm_unpacklo_epi64(d0123hl, d4567hl);
    data[5] = _mm_unpackhi_epi64(d0123hl, d4567hl);
    data[6] = _mm_unpacklo_epi64(d0123hh, d4567hh);
    data[7] = _mm_unpackhi_epi64(d0123hh, d4567hh);
}

/// Dequantize and inverse-transform one 8x8 block: the scalar
/// `dequantize_and_idct_block_8x8_inner`, bit for bit.
///
/// The block takes the vector path when every dequantized coefficient and
/// every first-pass output fits `i16` (always, for real images); otherwise it
/// is handed to the scalar kernel whole. See `idct::simd_k` for why that makes
/// the result exact.
///
/// # Safety
///
/// The CPU must support SSSE3. `output` must hold `output_linestride * 7 + 8`
/// bytes (asserted).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
pub unsafe fn dequantize_and_idct_block_8x8(
    coefficients: &[i16; 64],
    quantization_table: &[u16; 64],
    output_linestride: usize,
    output: &mut [u8],
) {
    // The loop below will write to positions [output_linestride * i, output_linestride * i + 8)
    // for 0<=i<8. Thus, the last accessed position is at an offset of output_linestride * 7 + 7,
    // and if that position is in-bounds, so are all other accesses.
    assert!(
        output.len()
            > output_linestride
                .checked_mul(7)
                .unwrap()
                .checked_add(7)
                .unwrap()
    );

    let zero = _mm_setzero_si128();

    // Dequantize. `mullo` is the product's low half; the product fits `i16`
    // exactly when the signed high half is the low half's sign extension.
    // That high half is only right for `q < 0x8000`, so a table with any
    // larger entry is treated as not fitting.
    let mut s = [zero; 8];
    let mut bad = zero;
    let mut qor = zero;
    for (i, row) in s.iter_mut().enumerate() {
        // SAFETY: rows 0..8 of 8 elements lie inside the 64-element arrays.
        let c = _mm_loadu_si128(coefficients.as_ptr().add(i * 8) as *const __m128i);
        let q = _mm_loadu_si128(quantization_table.as_ptr().add(i * 8) as *const __m128i);
        let lo = _mm_mullo_epi16(c, q);
        let hi = _mm_mulhi_epi16(c, q);
        bad = _mm_or_si128(bad, _mm_xor_si128(hi, _mm_srai_epi16(lo, 15)));
        qor = _mm_or_si128(qor, q);
        *row = lo;
    }
    bad = _mm_or_si128(bad, _mm_and_si128(qor, _mm_set1_epi16(i16::MIN)));

    // Columns. The outputs must fit i16 for the second pass: `v + 0x8000`
    // has a zero high half exactly when `v` is in range, so OR-ing them all
    // and testing the high halves checks every lane at once.
    let (lo, hi) = pass(&s, simd_k::PASS1_BIAS);
    let k8000 = _mm_set1_epi32(0x8000);
    let mut range = zero;
    let mut t = [zero; 8];
    for n in 0..8 {
        let a = _mm_srai_epi32(lo[n], simd_k::PASS1_SHIFT);
        let b = _mm_srai_epi32(hi[n], simd_k::PASS1_SHIFT);
        range = _mm_or_si128(
            range,
            _mm_or_si128(_mm_add_epi32(a, k8000), _mm_add_epi32(b, k8000)),
        );
        t[n] = _mm_packs_epi32(a, b);
    }
    bad = _mm_or_si128(bad, _mm_srli_epi32(range, 16));
    if _mm_movemask_epi8(_mm_cmpeq_epi8(bad, zero)) != 0xFFFF {
        return super::idct_fallback(coefficients, quantization_table, output_linestride, output);
    }

    // Rows. `packs` to i16 then `packus` to u8 saturate in turn, which is the
    // scalar's `clamp(0, 255)` of the shifted value.
    transpose8(&mut t);
    let (lo, hi) = pass(&t, simd_k::PASS2_BIAS);
    let mut o = [zero; 8];
    for n in 0..8 {
        o[n] = _mm_packs_epi32(
            _mm_srai_epi32(lo[n], simd_k::PASS2_SHIFT),
            _mm_srai_epi32(hi[n], simd_k::PASS2_SHIFT),
        );
    }
    transpose8(&mut o);

    for (i, row) in o.iter().enumerate() {
        // SAFETY: the assert above bounds `output_linestride * i + 8`.
        _mm_storel_epi64(
            output.as_mut_ptr().add(output_linestride * i) as *mut __m128i,
            _mm_packus_epi16(*row, *row),
        );
    }
}

/// YCbCr -> RGB for 8 pixels per step: `ycbcr_to_rgb`, bit for bit.
///
/// Each channel is `y + ((HALF + K * c) >> 20)` (see `decoder::ycc_k`): the
/// offsets in 32-bit lanes from `pmaddwd` of `(c << 8, c)` pairs against
/// `(K >> 8, K & 255)`, then added to `y` in 16-bit lanes and narrowed with
/// unsigned saturation, which is the scalar's clamp. Returns the number of
/// pixels converted (a multiple of 8); the caller does the rest.
///
/// # Safety
///
/// The CPU must support SSSE3. Lengths are asserted.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
pub unsafe fn color_convert_line_ycbcr(y: &[u8], cb: &[u8], cr: &[u8], output: &mut [u8]) -> usize {
    assert!(output.len().is_multiple_of(3));
    let num = output.len() / 3;
    assert!(num <= y.len());
    assert!(num <= cb.len());
    assert!(num <= cr.len());
    let num_vecs = num / 8;

    let zero = _mm_setzero_si128();
    let c128 = _mm_set1_epi16(128);
    let half = _mm_set1_epi32(ycc_k::HALF);
    let k_r = pair(ycc_k::split(ycc_k::CR_R));
    let k_gb = pair(ycc_k::split(ycc_k::CB_G));
    let k_gr = pair(ycc_k::split(ycc_k::CR_G));
    let k_b = pair(ycc_k::split(ycc_k::CB_B));

    // Control vectors for _mm_shuffle_epi8. -0x7F is selected so that the resulting position
    // after _mm_shuffle_epi8 will be filled with 0, so that the r, g, and b vectors can then
    // be OR-ed together.
    let shufr = _mm_setr_epi8(
        0, -0x7F, -0x7F, 1, -0x7F, -0x7F, 2, -0x7F, -0x7F, 3, -0x7F, -0x7F, 4, -0x7F, -0x7F, 5,
    );
    let shufg = _mm_setr_epi8(
        -0x7F, 0, -0x7F, -0x7F, 1, -0x7F, -0x7F, 2, -0x7F, -0x7F, 3, -0x7F, -0x7F, 4, -0x7F, -0x7F,
    );
    let shufb = _mm_alignr_epi8(shufg, shufg, 15);
    // For the next part of the rgb vectors, we need to select R values from 6 up, G and B from
    // 5 up. The highest bit of -0x7F + 6 is still set, so the corresponding location will
    // still be 0.
    let shufr1 = _mm_add_epi8(shufb, _mm_set1_epi8(6));
    let shufg1 = _mm_add_epi8(shufr, _mm_set1_epi8(5));
    let shufb1 = _mm_add_epi8(shufg, _mm_set1_epi8(5));

    for i in 0..num_vecs {
        // SAFETY: 8 * i + 8 <= num, which the asserts bound by every slice.
        let load =
            |p: &[u8]| _mm_unpacklo_epi8(_mm_loadl_epi64(p.as_ptr().add(i * 8) as *const _), zero);
        let yv = load(y);
        let cbv = _mm_sub_epi16(load(cb), c128);
        let crv = _mm_sub_epi16(load(cr), c128);

        // (c << 8, c) pairs; `c << 8` stays within i16 for c in -128..=127.
        let cb_lo = _mm_unpacklo_epi16(_mm_slli_epi16(cbv, 8), cbv);
        let cb_hi = _mm_unpackhi_epi16(_mm_slli_epi16(cbv, 8), cbv);
        let cr_lo = _mm_unpacklo_epi16(_mm_slli_epi16(crv, 8), crv);
        let cr_hi = _mm_unpackhi_epi16(_mm_slli_epi16(crv, 8), crv);

        let offset = |lo: __m128i, hi: __m128i| {
            _mm_packs_epi32(
                _mm_srai_epi32(_mm_add_epi32(lo, half), ycc_k::SHIFT),
                _mm_srai_epi32(_mm_add_epi32(hi, half), ycc_k::SHIFT),
            )
        };
        let dr = offset(_mm_madd_epi16(cr_lo, k_r), _mm_madd_epi16(cr_hi, k_r));
        let dg = offset(
            _mm_add_epi32(_mm_madd_epi16(cb_lo, k_gb), _mm_madd_epi16(cr_lo, k_gr)),
            _mm_add_epi32(_mm_madd_epi16(cb_hi, k_gb), _mm_madd_epi16(cr_hi, k_gr)),
        );
        let db = offset(_mm_madd_epi16(cb_lo, k_b), _mm_madd_epi16(cb_hi, k_b));

        let r = _mm_packus_epi16(_mm_add_epi16(yv, dr), zero);
        let g = _mm_packus_epi16(_mm_add_epi16(yv, dg), zero);
        let b = _mm_packus_epi16(_mm_add_epi16(yv, db), zero);

        // Shuffle rrrrrrrrggggggggbbbbbbbb to rgbrgbrgb...
        let rgb_low = _mm_or_si128(
            _mm_shuffle_epi8(r, shufr),
            _mm_or_si128(_mm_shuffle_epi8(g, shufg), _mm_shuffle_epi8(b, shufb)),
        );
        let rgb_hi = _mm_or_si128(
            _mm_shuffle_epi8(r, shufr1),
            _mm_or_si128(_mm_shuffle_epi8(g, shufg1), _mm_shuffle_epi8(b, shufb1)),
        );

        // SAFETY: bytes 24 * i .. 24 * i + 24 lie inside `output` (3 * num).
        let out = output.as_mut_ptr().add(24 * i);
        _mm_storeu_si128(out as *mut __m128i, rgb_low);
        _mm_storel_epi64(out.add(16) as *mut __m128i, rgb_hi);
    }

    num_vecs * 8
}
