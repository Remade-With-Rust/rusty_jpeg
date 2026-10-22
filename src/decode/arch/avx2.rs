//! AVX2 IDCT that transforms **two 8x8 blocks at once**.
//!
//! This is the SSSE3 kernel widened to 256 bits, with block A in the low
//! 128-bit lane and block B in the high one. That is safe to do mechanically
//! because every operation involved is lane-independent:
//!
//! - the arithmetic is elementwise (`mulhrs`/`adds`/`subs`/`slli`/`srai`);
//! - `_mm256_unpack*` interleave WITHIN each 128-bit lane, so the 8x8 transpose
//!   below transposes each block separately without any cross-lane traffic;
//! - `_mm256_packus_epi16` likewise packs per lane.
//!
//! So each lane executes exactly the instruction sequence SSSE3 executed, on
//! exactly the same inputs. The output is **byte-identical** to calling the
//! SSSE3 kernel twice -- which is what `avx2_pair_matches_ssse3_twice` asserts,
//! and why this needed no tolerance and no requantisation of the constants.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// One IDCT pass over 8 rows, each holding two blocks' worth of coefficients.
///
/// Line-for-line the SSSE3 `idct8`; only the register width differs. The
/// fixed-point constants are unchanged, so the rounding is unchanged.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn idct8(data: &mut [__m256i; 8]) {
    let p2 = data[2];
    let p3 = data[6];
    let p1 = _mm256_mulhrs_epi16(_mm256_adds_epi16(p2, p3), _mm256_set1_epi16(17734)); // 0.5411961
    let t2 = _mm256_subs_epi16(
        _mm256_subs_epi16(p1, p3),
        _mm256_mulhrs_epi16(p3, _mm256_set1_epi16(27779)), // 0.847759065
    );
    let t3 = _mm256_adds_epi16(p1, _mm256_mulhrs_epi16(p2, _mm256_set1_epi16(25079))); // 0.765366865

    let p2 = data[0];
    let p3 = data[4];
    let t0 = _mm256_adds_epi16(p2, p3);
    let t1 = _mm256_subs_epi16(p2, p3);

    let x0 = _mm256_adds_epi16(t0, t3);
    let x3 = _mm256_subs_epi16(t0, t3);
    let x1 = _mm256_adds_epi16(t1, t2);
    let x2 = _mm256_subs_epi16(t1, t2);

    let t0 = data[7];
    let t1 = data[5];
    let t2 = data[3];
    let t3 = data[1];

    let p3 = _mm256_adds_epi16(t0, t2);
    let p4 = _mm256_adds_epi16(t1, t3);
    let p1 = _mm256_adds_epi16(t0, t3);
    let p2 = _mm256_adds_epi16(t1, t2);
    let p5 = _mm256_adds_epi16(p3, p4);
    let p5 = _mm256_adds_epi16(p5, _mm256_mulhrs_epi16(p5, _mm256_set1_epi16(5763))); // 0.175875602

    let t0 = _mm256_mulhrs_epi16(t0, _mm256_set1_epi16(9786)); // 0.298631336
    let t1 = _mm256_adds_epi16(
        _mm256_adds_epi16(t1, t1),
        _mm256_mulhrs_epi16(t1, _mm256_set1_epi16(1741)), // 0.053119869
    );
    let t2 = _mm256_adds_epi16(
        _mm256_adds_epi16(t2, _mm256_adds_epi16(t2, t2)),
        _mm256_mulhrs_epi16(t2, _mm256_set1_epi16(2383)), // 0.072711026
    );
    let t3 = _mm256_adds_epi16(t3, _mm256_mulhrs_epi16(t3, _mm256_set1_epi16(16427))); // 0.501321110

    let p1 = _mm256_subs_epi16(p5, _mm256_mulhrs_epi16(p1, _mm256_set1_epi16(29490))); // 0.899976223
    let p2 = _mm256_subs_epi16(
        _mm256_subs_epi16(_mm256_subs_epi16(p5, p2), p2),
        _mm256_mulhrs_epi16(p2, _mm256_set1_epi16(18446)), // 0.562915447
    );

    let p3 = _mm256_subs_epi16(
        _mm256_mulhrs_epi16(p3, _mm256_set1_epi16(-31509)), // -0.961570560
        p3,
    );
    let p4 = _mm256_mulhrs_epi16(p4, _mm256_set1_epi16(-12785)); // -0.390180644

    let t3 = _mm256_adds_epi16(_mm256_adds_epi16(p1, p4), t3);
    let t2 = _mm256_adds_epi16(_mm256_adds_epi16(p2, p3), t2);
    let t1 = _mm256_adds_epi16(_mm256_adds_epi16(p2, p4), t1);
    let t0 = _mm256_adds_epi16(_mm256_adds_epi16(p1, p3), t0);

    data[0] = _mm256_adds_epi16(x0, t3);
    data[7] = _mm256_subs_epi16(x0, t3);
    data[1] = _mm256_adds_epi16(x1, t2);
    data[6] = _mm256_subs_epi16(x1, t2);
    data[2] = _mm256_adds_epi16(x2, t1);
    data[5] = _mm256_subs_epi16(x2, t1);
    data[3] = _mm256_adds_epi16(x3, t0);
    data[4] = _mm256_subs_epi16(x3, t0);
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

/// Dequantize and inverse-transform two 8x8 blocks in one instruction stream.
///
/// Both blocks belong to the SAME component, so they share a quantization
/// table and a line stride, and both write into one plane. Taking a single
/// output slice plus block B's relative origin avoids handing out two
/// overlapping `&mut` slices -- the blocks interleave row by row without ever
/// touching the same byte, which is true but not something the borrow checker
/// can be told.
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

    const SHIFT: i32 = 3;

    let mut data = [_mm256_setzero_si256(); 8];
    for (i, item) in data.iter_mut().enumerate() {
        // Lane 0 <- block A row i, lane 1 <- block B row i.
        let coef = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(
                coefficients_a.as_ptr().wrapping_add(i * 8) as *const _
            )),
            _mm_loadu_si128(coefficients_b.as_ptr().wrapping_add(i * 8) as *const _),
        );
        // Same table in both lanes.
        let quant = _mm256_broadcastsi128_si256(_mm_loadu_si128(
            quantization_table.as_ptr().wrapping_add(i * 8) as *const _,
        ));
        *item = _mm256_slli_epi16::<SHIFT>(_mm256_mullo_epi16(coef, quant));
    }

    // A half-height column pass for the 82.7% of blocks whose rows 4-7 are zero
    // was built, gated bit-identical, and MEASURED WORSE (2/11, z -2.11, twice):
    // the per-pair zero test plus an unpredictable branch cost more than the ~15
    // instructions it saved, and the pair kernel needs BOTH blocks sparse where
    // 82.7% is a per-block figure. Removed rather than left behind a toggle,
    // because the branch was the cost. See WHYS.md D4g.
    idct8(&mut data);
    transpose8(&mut data);
    idct8(&mut data);
    transpose8(&mut data);

    for (i, item) in data.iter().enumerate() {
        const OFFSET: i16 = 128 << (SHIFT + 3);
        const ROUNDING_BIAS: i16 = (1 << (SHIFT + 3)) >> 1;

        let with_offset = _mm256_adds_epi16(*item, _mm256_set1_epi16(OFFSET + ROUNDING_BIAS));
        // packus is per-lane: lane 0's low 8 bytes are block A's row, lane 1's
        // low 8 bytes are block B's.
        let packed = _mm256_packus_epi16(
            _mm256_srai_epi16::<{ SHIFT + 3 }>(with_offset),
            _mm256_setzero_si256(),
        );

        let base = output.as_mut_ptr().wrapping_add(output_linestride * i);

        let mut buf = [0u8; 16];
        _mm_storeu_si128(buf.as_mut_ptr() as *mut _, _mm256_castsi256_si128(packed));
        core::ptr::copy_nonoverlapping::<u8>(buf.as_ptr(), base, 8);

        _mm_storeu_si128(
            buf.as_mut_ptr() as *mut _,
            _mm256_extracti128_si256::<1>(packed),
        );
        core::ptr::copy_nonoverlapping::<u8>(buf.as_ptr(), base.wrapping_add(output_offset_b), 8);
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

/// YCbCr -> RGB, 16 pixels per step: the SSSE3 kernel widened to 256 bits.
///
/// Converts **exactly the pixels the SSSE3 kernel converts** (all but the last
/// 8-15, which stay with the scalar tail) and returns that count, because the
/// scalar formula rounds differently: covering a different set of pixels would
/// change the image. The arithmetic is the SSSE3 sequence per 16-bit lane
/// (`slli`/`adds`/`subs`/`mulhrs`/`srai`/`packus`, same constants), so each
/// pixel is bit-identical to what SSSE3 produced; only the RGB interleave
/// differs, and that is pure byte movement. `color_convert_matches_ssse3`
/// asserts both properties.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn color_convert_line_ycbcr(y: &[u8], cb: &[u8], cr: &[u8], output: &mut [u8]) -> usize {
    assert!(output.len() % 3 == 0);
    let num = output.len() / 3;
    assert!(num <= y.len());
    assert!(num <= cb.len());
    assert!(num <= cr.len());
    // The SSSE3 kernel's coverage: it reads 16 bytes to use 8, so it stops one
    // vector short of the end.
    let covered = (num / 8).saturating_sub(1) * 8;

    const SHIFT: i32 = 6;
    let c128 = _mm256_set1_epi16(128 << SHIFT);
    let half = _mm256_set1_epi16((1 << SHIFT) >> 1);
    let k_cr_r = _mm256_set1_epi16(13173);
    let k_cb_g = _mm256_set1_epi16(11276);
    let k_cr_g = _mm256_set1_epi16(23401);
    let k_cb_b = _mm256_set1_epi16(25297);
    let mask = |part: usize, channel: usize| {
        _mm_loadu_si128(RGB_MASKS[part][channel].as_ptr() as *const __m128i)
    };
    let masks = [
        [mask(0, 0), mask(0, 1), mask(0, 2)],
        [mask(1, 0), mask(1, 1), mask(1, 2)],
        [mask(2, 0), mask(2, 1), mask(2, 2)],
    ];

    let mut i = 0;
    while i + 16 <= covered {
        let widen = |p: &[u8]| {
            _mm256_slli_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128(p.as_ptr().add(i) as *const _)),
                SHIFT,
            )
        };
        let yv = _mm256_adds_epi16(widen(y), half);
        let cbv = _mm256_subs_epi16(widen(cb), c128);
        let crv = _mm256_subs_epi16(widen(cr), c128);

        let cr_140200 = _mm256_adds_epi16(_mm256_mulhrs_epi16(crv, k_cr_r), crv);
        let cb_034414 = _mm256_mulhrs_epi16(cbv, k_cb_g);
        let cr_071414 = _mm256_mulhrs_epi16(crv, k_cr_g);
        let cb_177200 = _mm256_adds_epi16(_mm256_mulhrs_epi16(cbv, k_cb_b), cbv);

        let r = _mm256_srai_epi16(_mm256_adds_epi16(yv, cr_140200), SHIFT);
        let g = _mm256_srai_epi16(
            _mm256_subs_epi16(yv, _mm256_adds_epi16(cb_034414, cr_071414)),
            SHIFT,
        );
        let b = _mm256_srai_epi16(_mm256_adds_epi16(yv, cb_177200), SHIFT);

        // packus works per 128-bit lane: [r0-7 g0-7 | r8-15 g8-15]. Reorder the
        // quadwords to [r0-7 r8-15 | g0-7 g8-15].
        let rg = _mm256_permute4x64_epi64(_mm256_packus_epi16(r, g), 0b11_01_10_00);
        let bb = _mm256_permute4x64_epi64(_mm256_packus_epi16(b, b), 0b11_01_10_00);
        let channels = [
            _mm256_castsi256_si128(rg),
            _mm256_extracti128_si256(rg, 1),
            _mm256_castsi256_si128(bb),
        ];

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

    // An odd 8-pixel step remains when `covered` is not a multiple of 16. Give
    // the tail to the SSSE3 kernel: its own coverage of the tail is exactly
    // `covered - i` (both stop one vector short of the same end).
    if i < covered {
        let done = super::ssse3::color_convert_line_ycbcr(
            &y[i..],
            &cb[i..],
            &cr[i..],
            &mut output[3 * i..],
        );
        debug_assert_eq!(i + done, covered);
    }
    covered
}

#[cfg(test)]
mod color_tests {
    #[test]
    fn color_convert_matches_ssse3() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("ssse3") {
            return;
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut byte = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        };
        for num in (0..80).chain([255, 256, 257, 959, 960, 961, 1920, 4097]) {
            let y: Vec<u8> = (0..num).map(|_| byte()).collect();
            let mut cb: Vec<u8> = (0..num).map(|_| byte()).collect();
            let cr: Vec<u8> = (0..num).map(|_| byte()).collect();
            // Saturation corners.
            if num > 4 {
                cb[0] = 0;
                cb[1] = 255;
            }
            let mut want = vec![0xAAu8; 3 * num];
            let mut got = vec![0x55u8; 3 * num];
            let (nw, ng) = unsafe {
                (
                    super::super::ssse3::color_convert_line_ycbcr(&y, &cb, &cr, &mut want),
                    super::color_convert_line_ycbcr(&y, &cb, &cr, &mut got),
                )
            };
            assert_eq!(ng, nw, "coverage differs at num {num}");
            assert_eq!(got[..3 * nw], want[..3 * nw], "pixels differ at num {num}");
        }
    }
}
