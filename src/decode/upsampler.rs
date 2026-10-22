use crate::decode::error::{Error, Result, UnsupportedFeature};
use crate::decode::parser::Component;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

/// A colour converter: one output row from one row per component.
pub type ColorConvertFn = fn(&[&[u8]], &mut [u8]);

pub struct Upsampler {
    components: Vec<UpsamplerComponent>,
    line_buffer_size: usize,
    /// The vectorized H2V2 interior, resolved once per image.
    h2v2_pairs: Option<H2v2Pairs>,
}

/// See `arch::get_upsample_h2v2_pairs`.
type H2v2Pairs = unsafe fn(&[u8], &[u8], &mut [u8]) -> usize;

struct UpsamplerComponent {
    kind: Kind,
    width: usize,
    height: usize,
    row_stride: usize,
}

/// Static dispatch over the upsamplers. This used to be a `Box<dyn Upsample>`
/// called once per component per output row (3,240 indirect calls per 1080p
/// frame), which also kept the row loops from being inlined into the caller.
#[derive(Clone, Copy)]
enum Kind {
    H1V1,
    H2V1,
    H1V2,
    H2V2,
    /// Nearest neighbour, for any other integer ratio.
    Generic {
        horizontal_scaling_factor: u8,
        vertical_scaling_factor: u8,
    },
}

/// Per-call scratch: one line buffer per component plus the H2V2 column sums.
/// Allocated once per [`Upsampler::upsample_and_interleave_rows`] call, not
/// once per row -- the per-row `vec![vec![..]]` was 4 allocations and a zeroing
/// per output row, ~0.15 ms of a 1080p decode.
struct Scratch {
    lines: Vec<Vec<u8>>,
    sums: Vec<u16>,
}

impl Upsampler {
    pub fn new(
        components: &[Component],
        output_width: u16,
        output_height: u16,
    ) -> Result<Upsampler> {
        let h_max = components
            .iter()
            .map(|c| c.horizontal_sampling_factor)
            .max()
            .unwrap();
        let v_max = components
            .iter()
            .map(|c| c.vertical_sampling_factor)
            .max()
            .unwrap();
        let mut upsampler_components = Vec::with_capacity(components.len());

        for component in components {
            let kind = choose_upsampler(
                (
                    component.horizontal_sampling_factor,
                    component.vertical_sampling_factor,
                ),
                (h_max, v_max),
                output_width,
                output_height,
            )?;
            upsampler_components.push(UpsamplerComponent {
                kind,
                width: component.size.width as usize,
                height: component.size.height as usize,
                row_stride: component.block_size.width as usize * component.dct_scale,
            });
        }

        let buffer_size =
            components.iter().map(|c| c.size.width).max().unwrap() as usize * h_max as usize;

        #[cfg(not(feature = "platform_independent"))]
        let h2v2_pairs = crate::decode::arch::get_upsample_h2v2_pairs();
        #[cfg(feature = "platform_independent")]
        let h2v2_pairs = None;

        Ok(Upsampler {
            components: upsampler_components,
            line_buffer_size: buffer_size,
            h2v2_pairs,
        })
    }

    /// Upsample and colour-convert the output rows `rows` into `output`, which
    /// holds exactly those rows back to back.
    ///
    /// `direct_full_res`: a full-resolution (H1V1) component is handed to the
    /// colour converter as a slice of its plane instead of being copied into a
    /// line buffer first. Only valid for converters that read exactly
    /// `output_width` samples per component (the zip-based ones); the others
    /// keep the line-buffer contract they always had.
    pub fn upsample_and_interleave_rows(
        &self,
        component_data: &[Vec<u8>],
        rows: Range<usize>,
        output_width: usize,
        output: &mut [u8],
        color_convert: ColorConvertFn,
        direct_full_res: bool,
    ) {
        let component_count = component_data.len();
        debug_assert_eq!(component_count, self.components.len());
        if rows.is_empty() {
            return;
        }
        let line_size = output.len() / rows.len();

        let mut scratch = Scratch {
            lines: vec![vec![0u8; self.line_buffer_size]; component_count],
            sums: Vec::new(),
        };

        for (row, line) in rows.zip(output.chunks_exact_mut(line_size)) {
            for (i, (component, line_buffer)) in
                self.components.iter().zip(&mut scratch.lines).enumerate()
            {
                let input = &component_data[i];
                if direct_full_res && matches!(component.kind, Kind::H1V1) {
                    continue;
                }
                upsample_row(
                    component,
                    input,
                    row,
                    output_width,
                    line_buffer,
                    &mut scratch.sums,
                    self.h2v2_pairs,
                );
            }

            // The colour converters take 3 or 4 components (anything else is
            // rejected when they are chosen), so a fixed array suffices.
            let mut views: [&[u8]; 4] = [&[]; 4];
            for (i, component) in self.components.iter().enumerate() {
                views[i] = if direct_full_res && matches!(component.kind, Kind::H1V1) {
                    let start = row * component.row_stride;
                    &component_data[i][start..start + output_width]
                } else {
                    &scratch.lines[i]
                };
            }
            color_convert(&views[..component_count], line);
        }
    }
}

fn choose_upsampler(
    sampling_factors: (u8, u8),
    max_sampling_factors: (u8, u8),
    output_width: u16,
    output_height: u16,
) -> Result<Kind> {
    let h1 = sampling_factors.0 == max_sampling_factors.0 || output_width == 1;
    let v1 = sampling_factors.1 == max_sampling_factors.1 || output_height == 1;
    let h2 = sampling_factors.0 * 2 == max_sampling_factors.0;
    let v2 = sampling_factors.1 * 2 == max_sampling_factors.1;

    if h1 && v1 {
        Ok(Kind::H1V1)
    } else if h2 && v1 {
        Ok(Kind::H2V1)
    } else if h1 && v2 {
        Ok(Kind::H1V2)
    } else if h2 && v2 {
        Ok(Kind::H2V2)
    } else if max_sampling_factors.0 % sampling_factors.0 != 0
        || max_sampling_factors.1 % sampling_factors.1 != 0
    {
        Err(Error::Unsupported(
            UnsupportedFeature::NonIntegerSubsamplingRatio,
        ))
    } else {
        Ok(Kind::Generic {
            horizontal_scaling_factor: max_sampling_factors.0 / sampling_factors.0,
            vertical_scaling_factor: max_sampling_factors.1 / sampling_factors.1,
        })
    }
}

#[inline]
fn upsample_row(
    c: &UpsamplerComponent,
    input: &[u8],
    row: usize,
    output_width: usize,
    output: &mut [u8],
    sums: &mut Vec<u16>,
    h2v2_pairs: Option<H2v2Pairs>,
) {
    match c.kind {
        Kind::H1V1 => {
            let input = &input[row * c.row_stride..];
            output[..output_width].copy_from_slice(&input[..output_width]);
        }
        Kind::H2V1 => h2v1(&input[row * c.row_stride..], c.width, output),
        Kind::H1V2 => h1v2(input, c.height, c.row_stride, row, output_width, output),
        Kind::H2V2 => h2v2(
            input,
            c.width,
            c.height,
            c.row_stride,
            row,
            output,
            sums,
            h2v2_pairs,
        ),
        Kind::Generic {
            horizontal_scaling_factor,
            vertical_scaling_factor,
        } => {
            let mut index = 0;
            let start = (row / vertical_scaling_factor as usize) * c.row_stride;
            let input = &input[start..(start + c.width)];
            for val in input {
                for _ in 0..horizontal_scaling_factor {
                    output[index] = *val;
                    index += 1;
                }
            }
        }
    }
}

/// The far row of a vertically "fancy" (triangle-filter) upsample.
///
/// Exactly the row the original float expression
/// `(row / 2.0 + frac * 3.0 - 0.25).min(h - 1) as usize` selected (with
/// `frac` 0.5 for odd rows): the row below for an odd output row, the row
/// above for an even one, clamped to the plane. Integer, so no float
/// round-trip per row; the equivalence is pinned by `far_row_matches_float`.
#[inline]
fn far_row(row: usize, input_height: usize) -> usize {
    let near = row / 2;
    if row & 1 == 1 {
        (near + 1).min(input_height - 1)
    } else {
        near.saturating_sub(1).min(input_height - 1)
    }
}

fn h2v1(input: &[u8], input_width: usize, output: &mut [u8]) {
    if input_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }
    let input = &input[..input_width];
    let output = &mut output[..input_width * 2];

    output[0] = input[0];
    output[1] = ((input[0] as u32 * 3 + input[1] as u32 + 2) >> 2) as u8;

    // Interior: sample i yields outputs 2i and 2i+1 from (i-1, i, i+1). Written
    // over `windows(3)` / `chunks_exact_mut(2)` so the bounds are proven once
    // and the loop vectorizes; same integer arithmetic as before.
    for (o, w) in output[2..input_width * 2 - 2]
        .chunks_exact_mut(2)
        .zip(input.windows(3))
    {
        let sample = 3 * w[1] as u16 + 2;
        o[0] = ((sample + w[0] as u16) >> 2) as u8;
        o[1] = ((sample + w[2] as u16) >> 2) as u8;
    }

    output[(input_width - 1) * 2] =
        ((input[input_width - 1] as u32 * 3 + input[input_width - 2] as u32 + 2) >> 2) as u8;
    output[(input_width - 1) * 2 + 1] = input[input_width - 1];
}

fn h1v2(
    input: &[u8],
    input_height: usize,
    row_stride: usize,
    row: usize,
    output_width: usize,
    output: &mut [u8],
) {
    let near = row / 2;
    let far = far_row(row, input_height);
    let input_near = &input[near * row_stride..][..output_width];
    let input_far = &input[far * row_stride..][..output_width];

    for ((o, &n), &f) in output[..output_width]
        .iter_mut()
        .zip(input_near)
        .zip(input_far)
    {
        *o = ((3 * n as u16 + f as u16 + 2) >> 2) as u8;
    }
}

/// 4:2:0 chroma: triangle filter in both directions, libjpeg's "fancy"
/// upsampling, bit for bit what this function always produced.
///
/// The interior goes to the AVX2 kernel when there is one; the rest is two
/// passes instead of the old fused bounds-checked loop: first the vertical
/// sums `t = 3*near + far` (one contiguous zip), then the horizontal pass over
/// adjacent sum pairs. The old single loop carried `t` through a scalar
/// recurrence and wrote two interleaved outputs per step through checked
/// indices, and did not vectorize.
#[allow(clippy::too_many_arguments)]
fn h2v2(
    input: &[u8],
    input_width: usize,
    input_height: usize,
    row_stride: usize,
    row: usize,
    output: &mut [u8],
    sums: &mut Vec<u16>,
    pairs_kernel: Option<H2v2Pairs>,
) {
    let near = row / 2;
    let far = far_row(row, input_height);
    let input_near = &input[near * row_stride..][..input_width];
    let input_far = &input[far * row_stride..][..input_width];

    if input_width == 1 {
        let value = ((3 * input_near[0] as u32 + input_far[0] as u32 + 2) >> 2) as u8;
        output[0] = value;
        output[1] = value;
        return;
    }
    let output = &mut output[..input_width * 2];
    let column = |i: usize| 3 * input_near[i] as u16 + input_far[i] as u16;
    output[0] = ((column(0) + 2) >> 2) as u8;
    output[input_width * 2 - 1] = ((column(input_width - 1) + 2) >> 2) as u8;

    // Pair k (samples k and k+1) writes interior bytes 2k and 2k+1.
    let interior = &mut output[1..input_width * 2 - 1];
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(not(feature = "platform_independent"))]
    if let Some(kernel) = pairs_kernel {
        // SAFETY: the kernel was chosen by runtime feature detection, and it
        // reads only within `input_near`/`input_far` and writes only within
        // `interior`.
        #[allow(unsafe_code)]
        {
            done = unsafe { kernel(input_near, input_far, interior) };
        }
    }
    #[cfg(feature = "platform_independent")]
    let _ = pairs_kernel;

    sums.clear();
    sums.extend(
        input_near[done..]
            .iter()
            .zip(&input_far[done..])
            .map(|(&n, &f)| 3 * n as u16 + f as u16),
    );
    for (o, p) in interior[2 * done..]
        .chunks_exact_mut(2)
        .zip(sums.windows(2))
    {
        let (t0, t1) = (p[0], p[1]);
        o[0] = ((3 * t0 + t1 + 8) >> 4) as u8;
        o[1] = ((3 * t1 + t0 + 8) >> 4) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The integer far-row rule must pick exactly the row the float
    /// expression it replaced picked, for every row of every plane height.
    #[test]
    fn far_row_matches_float() {
        for input_height in 1..=64usize {
            for row in 0..2 * input_height {
                let row_near = row as f32 / 2.0;
                let row_frac = (row & 1) as f32 * 0.5;
                let row_far = (row_near + row_frac * 3.0 - 0.25).min((input_height - 1) as f32);
                assert_eq!(
                    far_row(row, input_height),
                    row_far as usize,
                    "row {row} height {input_height}"
                );
                assert_eq!(row / 2, row_near as usize);
            }
        }
    }

    // The pre-rewrite kernels, verbatim: the oracle the fast ones must match.
    fn h2v2_reference(
        input: &[u8],
        input_width: usize,
        input_height: usize,
        row_stride: usize,
        row: usize,
        output: &mut [u8],
    ) {
        let row_near = row as f32 / 2.0;
        let row_frac = (row & 1) as f32 * 0.5;
        let row_far = (row_near + row_frac * 3.0 - 0.25).min((input_height - 1) as f32);
        let input_near = &input[row_near as usize * row_stride..];
        let input_far = &input[row_far as usize * row_stride..];
        if input_width == 1 {
            let value = ((3 * input_near[0] as u32 + input_far[0] as u32 + 2) >> 2) as u8;
            output[0] = value;
            output[1] = value;
            return;
        }
        let mut t1 = 3 * input_near[0] as u32 + input_far[0] as u32;
        output[0] = ((t1 + 2) >> 2) as u8;
        for i in 1..input_width {
            let t0 = t1;
            t1 = 3 * input_near[i] as u32 + input_far[i] as u32;
            output[i * 2 - 1] = ((3 * t0 + t1 + 8) >> 4) as u8;
            output[i * 2] = ((3 * t1 + t0 + 8) >> 4) as u8;
        }
        output[input_width * 2 - 1] = ((t1 + 2) >> 2) as u8;
    }

    fn h2v1_reference(input: &[u8], input_width: usize, output: &mut [u8]) {
        if input_width == 1 {
            output[0] = input[0];
            output[1] = input[0];
            return;
        }
        output[0] = input[0];
        output[1] = ((input[0] as u32 * 3 + input[1] as u32 + 2) >> 2) as u8;
        for i in 1..input_width - 1 {
            let sample = 3 * input[i] as u32 + 2;
            output[i * 2] = ((sample + input[i - 1] as u32) >> 2) as u8;
            output[i * 2 + 1] = ((sample + input[i + 1] as u32) >> 2) as u8;
        }
        output[(input_width - 1) * 2] =
            ((input[input_width - 1] as u32 * 3 + input[input_width - 2] as u32 + 2) >> 2) as u8;
        output[(input_width - 1) * 2 + 1] = input[input_width - 1];
    }

    fn noise(len: usize, mut state: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn h2v2_matches_reference() {
        let mut sums = Vec::new();
        for (w, h) in [
            (1, 1),
            (1, 5),
            (2, 1),
            (2, 2),
            (3, 7),
            (16, 3),
            (17, 9),
            (18, 2),
            (33, 3),
            (64, 33),
            (960, 2),
            (961, 4),
        ] {
            let stride = w + 7;
            let input = noise(stride * h, (w * 131 + h) as u64 | 1);
            for row in 0..2 * h {
                let mut want = vec![0u8; 2 * w];
                let mut got = vec![0u8; 2 * w];
                h2v2_reference(&input, w, h, stride, row, &mut want);
                h2v2(&input, w, h, stride, row, &mut got, &mut sums, None);
                assert_eq!(got, want, "scalar: w {w} h {h} row {row}");
                #[cfg(not(feature = "platform_independent"))]
                if let Some(kernel) = crate::decode::arch::get_upsample_h2v2_pairs() {
                    got.fill(0);
                    h2v2(&input, w, h, stride, row, &mut got, &mut sums, Some(kernel));
                    assert_eq!(got, want, "simd: w {w} h {h} row {row}");
                }
            }
        }
    }

    #[test]
    fn h2v1_matches_reference() {
        for w in [1usize, 2, 3, 4, 17, 960, 961] {
            let input = noise(w + 5, w as u64 * 7 + 1);
            let mut want = vec![0u8; 2 * w];
            let mut got = vec![0u8; 2 * w];
            h2v1_reference(&input, w, &mut want);
            h2v1(&input, w, &mut got);
            assert_eq!(got, want, "w {w}");
        }
    }
}
