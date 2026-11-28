//! NEON kernels (behind `nightly_aarch64_neon`): the single-block IDCT and
//! YCbCr->RGB, both bit-exact twins of the scalar code. The IDCT is the x86
//! kernel's arithmetic with `vmull`/`vmlal` in place of `pmaddwd` (see
//! `idct::simd_k`); the colour kernel uses NEON's native 32-bit multiply.

#[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
use core::arch::aarch64::*;

#[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
use crate::decode::{decoder::ycc_k, idct::simd_k};

/// Half of one 1-D pass: four lanes of the scalar `kernel`, each output as
/// `L_n(s) + bias` in `i32` -- the scalar's value before its shift.
/// `vmull`/`vmlal`/`vmlsl` multiply 16-bit lanes into 32-bit ones and add
/// with wrapping, which is the scalar's own modulo-2^32 arithmetic.
#[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pass_half(s: [int16x4_t; 8], bias: int32x4_t) -> [int32x4_t; 8] {
    let t2 = vmlal_n_s16(
        vmull_n_s16(s[2], simd_k::EVEN_T2[0]),
        s[6],
        simd_k::EVEN_T2[1],
    );
    let t3 = vmlal_n_s16(
        vmull_n_s16(s[2], simd_k::EVEN_T3[0]),
        s[6],
        simd_k::EVEN_T3[1],
    );
    let t0 = vmlal_n_s16(vmlal_n_s16(bias, s[0], simd_k::FSH), s[4], simd_k::FSH);
    let t1 = vmlsl_n_s16(vmlal_n_s16(bias, s[0], simd_k::FSH), s[4], simd_k::FSH);
    let x = [
        vaddq_s32(t0, t3),
        vaddq_s32(t1, t2),
        vsubq_s32(t1, t2),
        vsubq_s32(t0, t3),
    ];
    let mut t = [vdupq_n_s32(0); 4];
    for (n, c) in simd_k::ODD.iter().enumerate() {
        t[n] = vmlal_n_s16(
            vmlal_n_s16(vmlal_n_s16(vmull_n_s16(s[1], c[0]), s[3], c[1]), s[5], c[2]),
            s[7],
            c[3],
        );
    }
    let mut out = [vdupq_n_s32(0); 8];
    for n in 0..4 {
        out[n] = vaddq_s32(x[n], t[3 - n]);
        out[7 - n] = vsubq_s32(x[n], t[3 - n]);
    }
    out
}

/// One 1-D pass over eight 16-bit lanes; returns lanes 0-3 and 4-7 as `i32`.
#[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pass(s: &[int16x8_t; 8], bias: i32) -> ([int32x4_t; 8], [int32x4_t; 8]) {
    let bias = vdupq_n_s32(bias);
    let mut lo = [vdup_n_s16(0); 8];
    let mut hi = [vdup_n_s16(0); 8];
    for k in 0..8 {
        lo[k] = vget_low_s16(s[k]);
        hi[k] = vget_high_s16(s[k]);
    }
    (pass_half(lo, bias), pass_half(hi, bias))
}

#[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn transpose8(data: &mut [int16x8_t; 8]) {
    // Use NEON's 2x2 matrix transposes (vtrn) to do the transposition in each 4x4 block, then
    // combine the 4x4 blocks.
    let a01 = vtrnq_s16(data[0], data[1]);
    let a23 = vtrnq_s16(data[2], data[3]);

    let four0 = vtrnq_s32(vreinterpretq_s32_s16(a01.0), vreinterpretq_s32_s16(a23.0));
    let four1 = vtrnq_s32(vreinterpretq_s32_s16(a01.1), vreinterpretq_s32_s16(a23.1));

    let a45 = vtrnq_s16(data[4], data[5]);
    let a67 = vtrnq_s16(data[6], data[7]);

    let four2 = vtrnq_s32(vreinterpretq_s32_s16(a45.0), vreinterpretq_s32_s16(a67.0));
    let four3 = vtrnq_s32(vreinterpretq_s32_s16(a45.1), vreinterpretq_s32_s16(a67.1));

    data[0] = vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(four0.0), vget_low_s32(four2.0)));
    data[1] = vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(four1.0), vget_low_s32(four3.0)));
    data[2] = vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(four0.1), vget_low_s32(four2.1)));
    data[3] = vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(four1.1), vget_low_s32(four3.1)));
    data[4] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(four0.0), vget_high_s32(four2.0)));
    data[5] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(four1.0), vget_high_s32(four3.0)));
    data[6] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(four0.1), vget_high_s32(four2.1)));
    data[7] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(four1.1), vget_high_s32(four3.1)));
}

/// Dequantize and inverse-transform one 8x8 block: the scalar kernel, bit for
/// bit. A block whose dequantized coefficients or first-pass outputs leave
/// `i16` (never, for real images) is handed to the scalar kernel whole.
///
/// # Safety
///
/// `output` must hold `output_linestride * 7 + 8` bytes (asserted).
#[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
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

    // Dequantize in 32-bit lanes (exact for q < 0x8000; a larger table entry
    // sends the block to the scalar kernel), recording whether every product
    // fits i16: `v + 0x8000` has a zero high half exactly when it does.
    let k8000 = vdupq_n_s32(0x8000);
    let mut range = vdupq_n_u32(0);
    let mut qor = vdupq_n_u16(0);
    let mut s = [vdupq_n_s16(0); 8];
    for (i, row) in s.iter_mut().enumerate() {
        // SAFETY: rows 0..8 of 8 elements lie inside the 64-element arrays.
        let c = vld1q_s16(coefficients.as_ptr().add(i * 8));
        let q = vld1q_u16(quantization_table.as_ptr().add(i * 8));
        qor = vorrq_u16(qor, q);
        let qs = vreinterpretq_s16_u16(q);
        let plo = vmull_s16(vget_low_s16(c), vget_low_s16(qs));
        let phi = vmull_high_s16(c, qs);
        range = vorrq_u32(range, vreinterpretq_u32_s32(vaddq_s32(plo, k8000)));
        range = vorrq_u32(range, vreinterpretq_u32_s32(vaddq_s32(phi, k8000)));
        *row = vcombine_s16(vmovn_s32(plo), vmovn_s32(phi));
    }

    let (lo, hi) = pass(&s, simd_k::PASS1_BIAS);
    let mut t = [vdupq_n_s16(0); 8];
    for n in 0..8 {
        let a = vshrq_n_s32::<{ simd_k::PASS1_SHIFT }>(lo[n]);
        let b = vshrq_n_s32::<{ simd_k::PASS1_SHIFT }>(hi[n]);
        range = vorrq_u32(range, vreinterpretq_u32_s32(vaddq_s32(a, k8000)));
        range = vorrq_u32(range, vreinterpretq_u32_s32(vaddq_s32(b, k8000)));
        t[n] = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
    }
    if vmaxvq_u32(vshrq_n_u32::<16>(range)) != 0
        || vmaxvq_u16(vandq_u16(qor, vdupq_n_u16(0x8000))) != 0
    {
        return super::idct_fallback(coefficients, quantization_table, output_linestride, output);
    }

    // Rows. Saturating narrows to i16 then u8 are the scalar's clamp.
    transpose8(&mut t);
    let (lo, hi) = pass(&t, simd_k::PASS2_BIAS);
    let mut o = [vdupq_n_s16(0); 8];
    for n in 0..8 {
        o[n] = vcombine_s16(
            vqmovn_s32(vshrq_n_s32::<{ simd_k::PASS2_SHIFT }>(lo[n])),
            vqmovn_s32(vshrq_n_s32::<{ simd_k::PASS2_SHIFT }>(hi[n])),
        );
    }
    transpose8(&mut o);

    for (i, row) in o.iter().enumerate() {
        // SAFETY: the assert above bounds `output_linestride * i + 8`.
        vst1_u8(
            output.as_mut_ptr().add(output_linestride * i),
            vqmovun_s16(*row),
        );
    }
}

/// YCbCr -> RGB, 8 pixels per step: `ycbcr_to_rgb`, bit for bit. Each channel
/// is `y + ((HALF + K * c) >> 20)` (see `decoder::ycc_k`) in 32-bit lanes,
/// added to `y` in 16-bit lanes and narrowed with unsigned saturation (the
/// scalar's clamp). Returns the number of pixels converted.
///
/// # Safety
///
/// Lengths are asserted.
#[cfg(all(feature = "nightly_aarch64_neon", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
pub unsafe fn color_convert_line_ycbcr(y: &[u8], cb: &[u8], cr: &[u8], output: &mut [u8]) -> usize {
    assert!(output.len().is_multiple_of(3));
    let num = output.len() / 3;
    assert!(num <= y.len());
    assert!(num <= cb.len());
    assert!(num <= cr.len());
    let num_vecs = num / 8;

    let half = vdupq_n_s32(ycc_k::HALF);
    let c128 = vdupq_n_s16(128);
    for i in 0..num_vecs {
        // SAFETY: 8 * i + 8 <= num, which the asserts bound by every slice.
        let yv = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y.as_ptr().add(i * 8))));
        let cbv = vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(cb.as_ptr().add(i * 8)))),
            c128,
        );
        let crv = vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(cr.as_ptr().add(i * 8)))),
            c128,
        );
        let (cb_lo, cb_hi) = (vmovl_s16(vget_low_s16(cbv)), vmovl_high_s16(cbv));
        let (cr_lo, cr_hi) = (vmovl_s16(vget_low_s16(crv)), vmovl_high_s16(crv));

        let offset = |lo: int32x4_t, hi: int32x4_t| {
            vcombine_s16(
                vmovn_s32(vshrq_n_s32::<{ ycc_k::SHIFT }>(lo)),
                vmovn_s32(vshrq_n_s32::<{ ycc_k::SHIFT }>(hi)),
            )
        };
        let dr = offset(
            vmlaq_n_s32(half, cr_lo, ycc_k::CR_R),
            vmlaq_n_s32(half, cr_hi, ycc_k::CR_R),
        );
        let dg = offset(
            vmlaq_n_s32(vmlaq_n_s32(half, cb_lo, ycc_k::CB_G), cr_lo, ycc_k::CR_G),
            vmlaq_n_s32(vmlaq_n_s32(half, cb_hi, ycc_k::CB_G), cr_hi, ycc_k::CR_G),
        );
        let db = offset(
            vmlaq_n_s32(half, cb_lo, ycc_k::CB_B),
            vmlaq_n_s32(half, cb_hi, ycc_k::CB_B),
        );

        let r = vqmovun_s16(vaddq_s16(yv, dr));
        let g = vqmovun_s16(vaddq_s16(yv, dg));
        let b = vqmovun_s16(vaddq_s16(yv, db));

        // SAFETY: bytes 24 * i .. 24 * i + 24 lie inside `output` (3 * num).
        vst3_u8(output.as_mut_ptr().add(24 * i), uint8x8x3_t(r, g, b));
    }

    num_vecs * 8
}
