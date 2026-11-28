//! wasm simd128 kernels: the single-block IDCT and YCbCr->RGB, both bit-exact
//! twins of the scalar code. `i32x4.dot_i16x8_s` is `pmaddwd`, so the IDCT is
//! the x86 kernel's arithmetic (see `idct::simd_k`); the colour kernel uses
//! the native 32-bit multiply.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::decode::{decoder::ycc_k, idct::simd_k};

/// A dot-product operand: the pair `k = [even, odd]` in every 32-bit lane.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
#[target_feature(enable = "simd128")]
fn pair(k: [i16; 2]) -> v128 {
    i32x4_splat((i32::from(k[1]) << 16) | i32::from(k[0] as u16))
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
#[target_feature(enable = "simd128")]
fn interleave_lo(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
#[target_feature(enable = "simd128")]
fn interleave_hi(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b)
}

/// Half of one 1-D pass: four lanes of the scalar `kernel`, each output as
/// `L_n(s) + bias` in `i32` -- the scalar's value before its shift.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
#[target_feature(enable = "simd128")]
fn pass_half(s04: v128, s26: v128, s13: v128, s57: v128, bias: v128) -> [v128; 8] {
    let t2 = i32x4_dot_i16x8(s26, pair(simd_k::EVEN_T2));
    let t3 = i32x4_dot_i16x8(s26, pair(simd_k::EVEN_T3));
    let t0 = i32x4_add(i32x4_dot_i16x8(s04, pair([simd_k::FSH, simd_k::FSH])), bias);
    let t1 = i32x4_add(
        i32x4_dot_i16x8(s04, pair([simd_k::FSH, -simd_k::FSH])),
        bias,
    );
    let x = [
        i32x4_add(t0, t3),
        i32x4_add(t1, t2),
        i32x4_sub(t1, t2),
        i32x4_sub(t0, t3),
    ];
    let mut t = [i32x4_splat(0); 4];
    for (n, c) in simd_k::ODD.iter().enumerate() {
        t[n] = i32x4_add(
            i32x4_dot_i16x8(s13, pair([c[0], c[1]])),
            i32x4_dot_i16x8(s57, pair([c[2], c[3]])),
        );
    }
    let mut out = [i32x4_splat(0); 8];
    for n in 0..4 {
        out[n] = i32x4_add(x[n], t[3 - n]);
        out[7 - n] = i32x4_sub(x[n], t[3 - n]);
    }
    out
}

/// One 1-D pass over eight 16-bit lanes; returns lanes 0-3 and 4-7 as `i32`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
#[target_feature(enable = "simd128")]
fn pass(s: &[v128; 8], bias: i32) -> ([v128; 8], [v128; 8]) {
    let bias = i32x4_splat(bias);
    let lo = pass_half(
        interleave_lo(s[0], s[4]),
        interleave_lo(s[2], s[6]),
        interleave_lo(s[1], s[3]),
        interleave_lo(s[5], s[7]),
        bias,
    );
    let hi = pass_half(
        interleave_hi(s[0], s[4]),
        interleave_hi(s[2], s[6]),
        interleave_hi(s[1], s[3]),
        interleave_hi(s[5], s[7]),
        bias,
    );
    (lo, hi)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
fn transpose8(data: &mut [v128; 8]) {
    // Transpose a 8x8 matrix with a sequence of interleaving operations.
    // Naming: dABl contains elements from the *l*ower halves of vectors A and B, interleaved, i.e.
    // A0 B0 A1 B1 ...
    // dABCDll contains elements from the lower quarter (ll) of vectors A, B, C, D, interleaved -
    // A0 B0 C0 D0 A1 B1 C1 D1 ...
    let d01l = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(data[0], data[1]);
    let d23l = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(data[2], data[3]);
    let d45l = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(data[4], data[5]);
    let d67l = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(data[6], data[7]);
    let d01h = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(data[0], data[1]);
    let d23h = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(data[2], data[3]);
    let d45h = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(data[4], data[5]);
    let d67h = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(data[6], data[7]);

    // Operating on 32-bits will interleave *consecutive pairs* of 16-bit integers.
    let d0123ll = i32x4_shuffle::<0, 4, 1, 5>(d01l, d23l);
    let d0123lh = i32x4_shuffle::<2, 6, 3, 7>(d01l, d23l);
    let d4567ll = i32x4_shuffle::<0, 4, 1, 5>(d45l, d67l);
    let d4567lh = i32x4_shuffle::<2, 6, 3, 7>(d45l, d67l);
    let d0123hl = i32x4_shuffle::<0, 4, 1, 5>(d01h, d23h);
    let d0123hh = i32x4_shuffle::<2, 6, 3, 7>(d01h, d23h);
    let d4567hl = i32x4_shuffle::<0, 4, 1, 5>(d45h, d67h);
    let d4567hh = i32x4_shuffle::<2, 6, 3, 7>(d45h, d67h);

    // Operating on 64-bits will interleave *consecutive quadruples* of 16-bit integers.
    data[0] = i64x2_shuffle::<0, 2>(d0123ll, d4567ll);
    data[1] = i64x2_shuffle::<1, 3>(d0123ll, d4567ll);
    data[2] = i64x2_shuffle::<0, 2>(d0123lh, d4567lh);
    data[3] = i64x2_shuffle::<1, 3>(d0123lh, d4567lh);
    data[4] = i64x2_shuffle::<0, 2>(d0123hl, d4567hl);
    data[5] = i64x2_shuffle::<1, 3>(d0123hl, d4567hl);
    data[6] = i64x2_shuffle::<0, 2>(d0123hh, d4567hh);
    data[7] = i64x2_shuffle::<1, 3>(d0123hh, d4567hh);
}

/// Dequantize and inverse-transform one 8x8 block: the scalar kernel, bit for
/// bit. A block whose dequantized coefficients or first-pass outputs leave
/// `i16` (never, for real images) is handed to the scalar kernel whole.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub fn dequantize_and_idct_block_8x8(
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

    // Dequantize in 32-bit lanes (exact for q < 0x8000; a larger table entry
    // sends the block to the scalar kernel), recording whether every product
    // fits i16: `v + 0x8000` has a zero high half exactly when it does.
    let k8000 = i32x4_splat(0x8000);
    let mut range = i32x4_splat(0);
    let mut qor = i32x4_splat(0);
    let mut s = [i32x4_splat(0); 8];
    for (i, row) in s.iter_mut().enumerate() {
        // SAFETY: rows 0..8 of 8 elements lie inside the 64-element arrays.
        let (c, q) = unsafe {
            (
                v128_load(coefficients.as_ptr().add(i * 8) as *const v128),
                v128_load(quantization_table.as_ptr().add(i * 8) as *const v128),
            )
        };
        qor = v128_or(qor, q);
        let plo = i32x4_extmul_low_i16x8(c, q);
        let phi = i32x4_extmul_high_i16x8(c, q);
        range = v128_or(range, v128_or(i32x4_add(plo, k8000), i32x4_add(phi, k8000)));
        *row = i16x8_narrow_i32x4(plo, phi);
    }

    let (lo, hi) = pass(&s, simd_k::PASS1_BIAS);
    let mut t = [i32x4_splat(0); 8];
    for n in 0..8 {
        let a = i32x4_shr(lo[n], simd_k::PASS1_SHIFT as u32);
        let b = i32x4_shr(hi[n], simd_k::PASS1_SHIFT as u32);
        range = v128_or(range, v128_or(i32x4_add(a, k8000), i32x4_add(b, k8000)));
        t[n] = i16x8_narrow_i32x4(a, b);
    }
    let bad = v128_or(u32x4_shr(range, 16), v128_and(qor, i16x8_splat(i16::MIN)));
    if v128_any_true(bad) {
        return super::idct_fallback(coefficients, quantization_table, output_linestride, output);
    }

    // Rows. Saturating narrows to i16 then u8 are the scalar's clamp.
    transpose8(&mut t);
    let (lo, hi) = pass(&t, simd_k::PASS2_BIAS);
    let mut o = [i32x4_splat(0); 8];
    for n in 0..8 {
        o[n] = i16x8_narrow_i32x4(
            i32x4_shr(lo[n], simd_k::PASS2_SHIFT as u32),
            i32x4_shr(hi[n], simd_k::PASS2_SHIFT as u32),
        );
    }
    transpose8(&mut o);

    for (i, row) in o.iter().enumerate() {
        // SAFETY: the assert above bounds `output_linestride * i + 8`.
        unsafe {
            v128_store64_lane::<0>(
                u8x16_narrow_i16x8(*row, *row),
                output.as_mut_ptr().add(output_linestride * i) as *mut u64,
            );
        }
    }
}

/// YCbCr -> RGB, 8 pixels per step: `ycbcr_to_rgb`, bit for bit. Each channel
/// is `y + ((HALF + K * c) >> 20)` (see `decoder::ycc_k`) in 32-bit lanes,
/// added to `y` in 16-bit lanes and narrowed with unsigned saturation (the
/// scalar's clamp). Returns the number of pixels converted.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub fn color_convert_line_ycbcr(
    y_slice: &[u8],
    cb_slice: &[u8],
    cr_slice: &[u8],
    output: &mut [u8],
) -> usize {
    assert!(output.len().is_multiple_of(3));
    let num = output.len() / 3;
    assert!(num <= y_slice.len());
    assert!(num <= cb_slice.len());
    assert!(num <= cr_slice.len());

    let num_vecs = num / 8;
    let half = i32x4_splat(ycc_k::HALF);
    let c128 = i16x8_splat(128);

    for i in 0..num_vecs {
        let y: v128;
        let cb: v128;
        let cr: v128;
        // SAFETY: 8 * i + 8 <= num, which the asserts bound by every slice.
        unsafe {
            y = v128_load64_zero(y_slice.as_ptr().add(i * 8) as *const _);
            cb = v128_load64_zero(cb_slice.as_ptr().add(i * 8) as *const _);
            cr = v128_load64_zero(cr_slice.as_ptr().add(i * 8) as *const _);
        }

        let yv = i16x8_extend_low_u8x16(y);
        let cbv = i16x8_sub(i16x8_extend_low_u8x16(cb), c128);
        let crv = i16x8_sub(i16x8_extend_low_u8x16(cr), c128);
        let (cb_lo, cb_hi) = (i32x4_extend_low_i16x8(cbv), i32x4_extend_high_i16x8(cbv));
        let (cr_lo, cr_hi) = (i32x4_extend_low_i16x8(crv), i32x4_extend_high_i16x8(crv));
        let k = |v: i32| i32x4_splat(v);
        let offset = |lo: v128, hi: v128| {
            i16x8_narrow_i32x4(
                i32x4_shr(i32x4_add(lo, half), ycc_k::SHIFT as u32),
                i32x4_shr(i32x4_add(hi, half), ycc_k::SHIFT as u32),
            )
        };
        let dr = offset(
            i32x4_mul(cr_lo, k(ycc_k::CR_R)),
            i32x4_mul(cr_hi, k(ycc_k::CR_R)),
        );
        let dg = offset(
            i32x4_add(
                i32x4_mul(cb_lo, k(ycc_k::CB_G)),
                i32x4_mul(cr_lo, k(ycc_k::CR_G)),
            ),
            i32x4_add(
                i32x4_mul(cb_hi, k(ycc_k::CB_G)),
                i32x4_mul(cr_hi, k(ycc_k::CR_G)),
            ),
        );
        let db = offset(
            i32x4_mul(cb_lo, k(ycc_k::CB_B)),
            i32x4_mul(cb_hi, k(ycc_k::CB_B)),
        );

        let zero = u8x16_splat(0);
        let r = u8x16_narrow_i16x8(i16x8_add(yv, dr), zero);
        let g = u8x16_narrow_i16x8(i16x8_add(yv, dg), zero);
        let b = u8x16_narrow_i16x8(i16x8_add(yv, db), zero);

        // Shuffle rrrrrrrrggggggggbbbbbbbb to rgbrgbrgb...

        let rg_lanes =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r, g);

        let rgb_low = i8x16_shuffle::<
            0,
            1,
            16, // r0, g0, b0
            2,
            3,
            17, // r1, g1, b1
            4,
            5,
            18, // r2, g2, b2
            6,
            7,
            19, // r3, g3, b3
            8,
            9,
            20, // r4, g4, b4
            10,
        >(rg_lanes, b); // r5

        let rgb_hi = i8x16_shuffle::<
            11,
            21,
            12, // g5, b5, r6
            13,
            22,
            14, // g6, b6, r7
            15,
            23,
            0, // g7, b7, --
            0,
            0,
            0, // --, --, --
            0,
            0,
            0, // --, --, --
            0,
        >(rg_lanes, b); // --

        // SAFETY: bytes 24 * i .. 24 * i + 24 lie inside `output` (3 * num).
        unsafe {
            v128_store(output.as_mut_ptr().add(24 * i) as *mut _, rgb_low);
            v128_store64_lane::<0>(rgb_hi, output.as_mut_ptr().add(24 * i + 16) as *mut _);
        }
    }

    num_vecs * 8
}
