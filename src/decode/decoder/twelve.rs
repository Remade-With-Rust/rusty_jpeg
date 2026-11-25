//! Rendering of 12-bit lossy frames: dequantization and inverse DCT, chroma
//! upsampling and colour conversion on 16-bit samples.
//!
//! Everything here is libjpeg-turbo's 12-bit arithmetic -- the accurate integer
//! IDCT (jidctint.c) and its reduced sizes (jidctred.c) with the 12-bit
//! `PASS1_BITS`, the "fancy" upsampling of jdsample.c with its alternating
//! rounding biases, the fixed-point colour conversion of jdcolor.c -- so the
//! output is `djpeg`'s, sample for sample. (libjpeg-turbo has no SIMD for it
//! either.) The 8-bit path keeps the kernels it always had.

use crate::decode::decoder::{choose_color_convert_func, ColorTransform};
use crate::decode::error::{Error, Result};
use crate::decode::parser::{Component, Dimensions};
use alloc::borrow::ToOwned;
use alloc::vec;
use alloc::vec::Vec;

/// The largest sample, and the level shift (A.3.1).
const MAX: i32 = 4095;
const CENTRE: i32 = 2048;

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 1;

// `x * 2^13`, rounded: the constants of jidctint.c and jidctred.c.
const FIX_0_211164243: i64 = 1730;
const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_509795579: i64 = 4176;
const FIX_0_541196100: i64 = 4433;
const FIX_0_601344887: i64 = 4926;
const FIX_0_720959822: i64 = 5906;
const FIX_0_765366865: i64 = 6270;
const FIX_0_850430095: i64 = 6967;
const FIX_0_899976223: i64 = 7373;
const FIX_1_061594337: i64 = 8697;
const FIX_1_175875602: i64 = 9633;
const FIX_1_272758580: i64 = 10426;
const FIX_1_451774981: i64 = 11893;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_172734803: i64 = 17799;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;
const FIX_3_624509785: i64 = 29692;

/// libjpeg's `DESCALE`. Nothing overflows: a dequantized coefficient fits 32
/// bits, and a pass gains fewer than 20.
#[inline(always)]
fn descale(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// libjpeg's `range_limit[x & RANGE_MASK]`: level shift and clamp, with the
/// wrap that a value beyond the table (corrupt data only) takes.
#[inline(always)]
fn range_limit(x: i64) -> u16 {
    let wrapped = ((x as i32) << 18) >> 18;
    (wrapped + CENTRE).clamp(0, MAX) as u16
}

/// One pass of the 8-point transform, not yet descaled.
#[inline(always)]
fn pass8(i: [i64; 8]) -> [i64; 8] {
    let z1 = (i[2] + i[6]) * FIX_0_541196100;
    let tmp2 = z1 - i[6] * FIX_1_847759065;
    let tmp3 = z1 + i[2] * FIX_0_765366865;
    let tmp0 = (i[0] + i[4]) << CONST_BITS;
    let tmp1 = (i[0] - i[4]) << CONST_BITS;
    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let z5 = (i[7] + i[3] + i[5] + i[1]) * FIX_1_175875602;
    let z1 = -(i[7] + i[1]) * FIX_0_899976223;
    let z2 = -(i[5] + i[3]) * FIX_2_562915447;
    let z3 = z5 - (i[7] + i[3]) * FIX_1_961570560;
    let z4 = z5 - (i[5] + i[1]) * FIX_0_390180644;
    let tmp0 = i[7] * FIX_0_298631336 + z1 + z3;
    let tmp1 = i[5] * FIX_2_053119869 + z2 + z4;
    let tmp2 = i[3] * FIX_3_072711026 + z2 + z3;
    let tmp3 = i[1] * FIX_1_501321110 + z1 + z4;

    [
        tmp10 + tmp3,
        tmp11 + tmp2,
        tmp12 + tmp1,
        tmp13 + tmp0,
        tmp13 - tmp0,
        tmp12 - tmp1,
        tmp11 - tmp2,
        tmp10 - tmp3,
    ]
}

/// One pass of the 4-point output transform (`_jpeg_idct_4x4`). Input 4 is not
/// used.
#[inline(always)]
fn pass4(i: [i64; 8]) -> [i64; 4] {
    let even = i[0] << (CONST_BITS + 1);
    let tmp2 = i[2] * FIX_1_847759065 - i[6] * FIX_0_765366865;
    let tmp10 = even + tmp2;
    let tmp12 = even - tmp2;
    let tmp0 = i[5] * FIX_1_451774981 + i[1] * FIX_1_061594337
        - i[7] * FIX_0_211164243
        - i[3] * FIX_2_172734803;
    let tmp2 = i[3] * FIX_0_899976223 + i[1] * FIX_2_562915447
        - i[7] * FIX_0_509795579
        - i[5] * FIX_0_601344887;
    [tmp10 + tmp2, tmp12 + tmp0, tmp12 - tmp0, tmp10 - tmp2]
}

/// One pass of the 2-point output transform (`_jpeg_idct_2x2`). Inputs 2, 4
/// and 6 are not used.
#[inline(always)]
fn pass2(i: [i64; 8]) -> [i64; 2] {
    let tmp10 = i[0] << (CONST_BITS + 2);
    let tmp0 = i[5] * FIX_0_850430095 + i[1] * FIX_3_624509785
        - i[7] * FIX_0_720959822
        - i[3] * FIX_1_272758580;
    [tmp10 + tmp0, tmp10 - tmp0]
}

/// Dequantize and transform one block to `N x N` samples. `extra` is the
/// additional scaling inside the reduced sizes' passes.
#[inline(always)]
fn idct<const N: usize>(
    coefficients: &[i16; 64],
    table: &[u16; 64],
    extra: u32,
    pass: impl Fn([i64; 8]) -> [i64; N],
    stride: usize,
    output: &mut [u16],
) {
    let mut workspace = [[0i64; 8]; N];
    for x in 0..8 {
        let column: [i64; 8] = core::array::from_fn(|y| {
            i64::from(coefficients[y * 8 + x]) * i64::from(table[y * 8 + x])
        });
        // libjpeg's workspace is `int`, hence the truncations.
        if column[1..].iter().fold(0, |any, &c| any | c) == 0 {
            // What the pass would give, without it.
            for row in workspace.iter_mut() {
                row[x] = i64::from((column[0] << PASS1_BITS) as i32);
            }
            continue;
        }
        for (row, value) in workspace.iter_mut().zip(pass(column)) {
            row[x] = i64::from(descale(value, CONST_BITS - PASS1_BITS + extra) as i32);
        }
    }
    for (row, line) in workspace.iter().zip(output.chunks_mut(stride)) {
        for (sample, value) in line[..N].iter_mut().zip(pass(*row)) {
            *sample = range_limit(descale(value, CONST_BITS + PASS1_BITS + 3 + extra));
        }
    }
}

fn idct_block(
    scale: usize,
    coefficients: &[i16; 64],
    table: &[u16; 64],
    dc_only: bool,
    stride: usize,
    output: &mut [u16],
) {
    if dc_only || scale == 1 {
        // Both passes reduce to their shortcuts, one value for the block.
        // (`_jpeg_idct_1x1` descales by 3 only; the bit of `PASS1_BITS` that
        // the larger sizes carry is even, so the rounding agrees.)
        let dc = i64::from(coefficients[0]) * i64::from(table[0]);
        let value = range_limit(descale(dc, 3));
        for line in output.chunks_mut(stride).take(scale) {
            line[..scale].fill(value);
        }
        return;
    }
    match scale {
        8 => idct::<8>(coefficients, table, 0, pass8, stride, output),
        4 => idct::<4>(coefficients, table, 1, pass4, stride, output),
        _ => idct::<2>(coefficients, table, 2, pass2, stride, output),
    }
}

/// Render whole block rows of a component: `coefficients` (and `nonzero`, the
/// blocks' AC masks, unless they no longer describe them) to their samples.
pub(super) fn render_rows(
    component: &Component,
    coefficients: &[i16],
    nonzero: Option<&[u64]>,
    table: &[u16; 64],
    output: &mut [u16],
) {
    let scale = component.dct_scale;
    let blocks_wide = usize::from(component.block_size.width);
    let stride = blocks_wide * scale;
    for (i, block) in coefficients.chunks_exact(64).enumerate() {
        let block: &[i16; 64] = block.try_into().unwrap();
        let dc_only = match nonzero {
            Some(masks) => masks[i] == 0,
            None => block[1..].iter().all(|&c| c == 0),
        };
        let at = (i / blocks_wide) * stride * scale + (i % blocks_wide) * scale;
        idct_block(scale, block, table, dc_only, stride, &mut output[at..]);
    }
}

/// Upsample, colour-convert and interleave the planes.
pub(super) fn compute_image(
    components: &[Component],
    mut planes: Vec<Vec<u16>>,
    output_size: Dimensions,
    color_transform: ColorTransform,
) -> Result<Vec<u16>> {
    if planes.is_empty() || planes.iter().any(Vec::is_empty) {
        return Err(Error::Format("not all components have data".to_owned()));
    }
    let width = usize::from(output_size.width);
    let height = usize::from(output_size.height);

    if let [component] = components {
        let stride = usize::from(component.block_size.width) * component.dct_scale;
        let mut plane = planes.remove(0);
        if stride != width {
            for y in 1..height {
                plane.copy_within(y * stride..y * stride + width, y * width);
            }
        }
        plane.truncate(width * height);
        return Ok(plane);
    }

    // The same refusals as at 8 bits.
    choose_color_convert_func(components.len(), color_transform)?;
    let convert: fn(&mut [u16]) = match color_transform {
        ColorTransform::YCbCr => |line| line.chunks_exact_mut(3).for_each(ycc_to_rgb),
        // As at 8 bits, the output is CMYK with 0 for no ink where the file
        // (Adobe's convention) has 0 for full ink: the complement of what
        // libjpeg returns, and for the CMY of YCCK that is the RGB.
        ColorTransform::YCCK => |line| {
            line.chunks_exact_mut(4).for_each(|pixel| {
                ycc_to_rgb(&mut pixel[..3]);
                pixel[3] = complement(pixel[3]);
            })
        },
        ColorTransform::CMYK => |line| line.iter_mut().for_each(|s| *s = complement(*s)),
        _ => |_| {},
    };

    let h_max = max_factor(components, |c| c.horizontal_sampling_factor);
    let v_max = max_factor(components, |c| c.vertical_sampling_factor);
    let n = components.len();
    let mut image = vec![0u16; width * height * n];
    // An expanded row can run past the output by less than one group (of 4).
    let mut line = vec![0u16; width + 4];
    for (y, row) in image.chunks_exact_mut(width * n).enumerate() {
        for (i, (component, plane)) in components.iter().zip(&planes).enumerate() {
            let h = h_max / usize::from(component.horizontal_sampling_factor);
            let v = v_max / usize::from(component.vertical_sampling_factor);
            let samples = upsample_row(component, plane, (h, v), y, width, &mut line);
            for (pixel, &sample) in row.chunks_exact_mut(n).zip(samples) {
                pixel[i] = sample;
            }
        }
        convert(row);
    }
    Ok(image)
}

fn max_factor(components: &[Component], factor: fn(&Component) -> u8) -> usize {
    usize::from(components.iter().map(factor).max().unwrap_or(1))
}

/// Row `y` of the component at full size: a row of its plane, or `line`,
/// filled. The expansion factors are whole (`Upsampler::new` has refused the
/// frame otherwise).
fn upsample_row<'a>(
    component: &Component,
    plane: &'a [u16],
    (h, v): (usize, usize),
    y: usize,
    width: usize,
    line: &'a mut [u16],
) -> &'a [u16] {
    let stride = usize::from(component.block_size.width) * component.dct_scale;
    let w = usize::from(component.size.width);
    let rows = usize::from(component.size.height);
    let row = |r: usize| &plane[r * stride..r * stride + w];
    let near = (y / v).min(rows - 1);
    // The second-nearest row of a 2:1 vertical expansion, inside the image.
    let far = if y % 2 == 1 {
        (near + 1).min(rows - 1)
    } else {
        near.saturating_sub(1)
    };
    // libjpeg has no fancy upsampling at 1/8 scale.
    let fancy = component.dct_scale > 1;
    let out = &mut line[..width.div_ceil(h) * h];
    let last = w - 1;
    match (h, v) {
        (1, 1) => return &row(near)[..width.min(w)],
        (2, 1) if fancy && w > 2 => {
            let sample = row(near).iter().map(|&s| u32::from(s));
            expand(sample, out, |c, left, right| {
                (((3 * c + left + 1) >> 2), ((3 * c + right + 2) >> 2))
            });
        }
        (1, 2) if fancy => {
            let bias = 1 + (y % 2) as u32;
            for ((o, &n), &f) in out.iter_mut().zip(row(near)).zip(row(far)) {
                *o = ((3 * u32::from(n) + u32::from(f) + bias) >> 2) as u16;
            }
        }
        (2, 2) if fancy && w > 2 => {
            let columns = row(near).iter().zip(row(far));
            let sums = columns.map(|(&n, &f)| 3 * u32::from(n) + u32::from(f));
            expand(sums, out, |c, left, right| {
                (((3 * c + left + 8) >> 4), ((3 * c + right + 7) >> 4))
            });
        }
        _ => {
            let s = row(near);
            for (x, group) in out.chunks_exact_mut(h).enumerate() {
                group.fill(s[x.min(last)]);
            }
        }
    }
    &out[..width]
}

/// Two outputs for each input, from it and its neighbours. At the ends the
/// missing neighbour is the input itself, which weights it fully.
#[inline(always)]
fn expand(
    input: impl Iterator<Item = u32>,
    output: &mut [u16],
    filter: impl Fn(u32, u32, u32) -> (u32, u32),
) {
    let mut input = input.fuse();
    let Some(mut current) = input.next() else {
        return;
    };
    let mut left = current;
    for pair in output.chunks_exact_mut(2) {
        let right = input.next().unwrap_or(current);
        let (a, b) = filter(current, left, right);
        pair[0] = a as u16;
        pair[1] = b as u16;
        left = current;
        current = right;
    }
}

fn complement(sample: u16) -> u16 {
    (MAX - i32::from(sample)) as u16
}

/// jdcolor.c's fixed-point conversion, at 12 bits.
fn ycc_to_rgb(pixel: &mut [u16]) {
    const SCALE: u32 = 16;
    const HALF: i32 = 1 << (SCALE - 1);
    // `x * 2^16`, rounded.
    const FIX_1_40200: i32 = 91881;
    const FIX_1_77200: i32 = 116130;
    const FIX_0_71414: i32 = 46802;
    const FIX_0_34414: i32 = 22554;
    let y = i32::from(pixel[0]);
    let cb = i32::from(pixel[1]) - CENTRE;
    let cr = i32::from(pixel[2]) - CENTRE;
    let r = y + ((FIX_1_40200 * cr + HALF) >> SCALE);
    let g = y + ((HALF - FIX_0_34414 * cb - FIX_0_71414 * cr) >> SCALE);
    let b = y + ((FIX_1_77200 * cb + HALF) >> SCALE);
    pixel[0] = r.clamp(0, MAX) as u16;
    pixel[1] = g.clamp(0, MAX) as u16;
    pixel[2] = b.clamp(0, MAX) as u16;
}
