//! The chip plan's YUV-input gate: decode-encode-decode PSNR through the packed
//! YUYV path is the RGB path's, on the test corpus. Both paths hand the encoder
//! the same picture at the same 4:2:2 sampling; the YUYV path only skips the
//! RGB conversion a camera never needed.
use rusty_jpeg::decode::Decoder;
use rusty_jpeg::encode::{ColorType, Encoder, SamplingFactor, YuyvImage};

fn decode_rgb(bytes: &[u8]) -> (u16, u16, Vec<u8>) {
    let mut d = Decoder::new(bytes);
    let px = d.decode().unwrap();
    let info = d.info().unwrap();
    assert_eq!(
        px.len(),
        usize::from(info.width) * usize::from(info.height) * 3,
        "three-channel output"
    );
    (info.width, info.height, px)
}

/// JFIF (BT.601 full-range) RGB -> YCbCr: the conversion the RGB path applies.
fn ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (r, g, b) = (f32::from(r), f32::from(g), f32::from(b));
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    let q = |v: f32| v.round().clamp(0.0, 255.0) as u8;
    (q(y), q(cb), q(cr))
}

/// RGB as packed YUYV 4:2:2, chroma averaged over each horizontal pair the way
/// a camera's ISP delivers it.
fn rgb_to_yuyv(rgb: &[u8], w: u16, h: u16) -> Vec<u8> {
    let (wu, hu) = (usize::from(w), usize::from(h));
    let cw = wu.div_ceil(2);
    let mut out = Vec::with_capacity(cw * 4 * hu);
    for r in 0..hu {
        for c in 0..cw {
            let px = |x: usize| {
                let i = (r * wu + x.min(wu - 1)) * 3;
                ycbcr(rgb[i], rgb[i + 1], rgb[i + 2])
            };
            let (y0, cb0, cr0) = px(2 * c);
            let (y1, cb1, cr1) = px(2 * c + 1);
            let avg = |a: u8, b: u8| ((u16::from(a) + u16::from(b) + 1) / 2) as u8;
            out.extend_from_slice(&[y0, avg(cb0, cb1), y1, avg(cr0, cr1)]);
        }
    }
    out
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// Decode a corpus image, encode it through the RGB path and through packed
/// YUYV at the same quality and sampling, decode both: the two PSNRs against
/// the source must agree. Sizes are printed for the record.
/// A 320x240 picture with colour gradients, sharp chroma edges and fine luma
/// texture — the corpus fixture is 32x32, too small to exercise 4:2:2 chroma.
fn synthetic(w: u16, h: u16) -> Vec<u8> {
    let (wu, hu) = (usize::from(w), usize::from(h));
    let mut rgb = Vec::with_capacity(wu * hu * 3);
    for y in 0..hu {
        for x in 0..wu {
            let stripe = (x / 40 + y / 30) % 2 == 0;
            let r = if stripe {
                (x * 255 / wu) as u8
            } else {
                255 - (y * 255 / hu) as u8
            };
            let g = ((x * 7 + y * 3) % 256) as u8 ^ if stripe { 0x20 } else { 0 };
            let b = if (x / 8 + y / 8) % 3 == 0 {
                200
            } else {
                (y * 255 / hu) as u8
            };
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

#[test]
fn yuyv_input_keeps_the_rgb_paths_psnr() {
    let fixture = decode_rgb(include_bytes!("fixtures/progressive_libjpeg.jpg"));
    let corpus = [fixture, (320, 240, synthetic(320, 240))];
    for (i, (w, h, rgb)) in corpus.iter().enumerate() {
        let (w, h, rgb) = (*w, *h, rgb.as_slice());
        for q in [75u8, 90] {
            let mut via_rgb = Vec::new();
            let mut enc = Encoder::new(&mut via_rgb, q);
            enc.set_sampling_factor(SamplingFactor::F_2_1);
            enc.encode(rgb, w, h, ColorType::Rgb).unwrap();
            let (_, _, back_rgb) = decode_rgb(&via_rgb);

            let yuyv = rgb_to_yuyv(rgb, w, h);
            let mut via_yuyv = Vec::new();
            let img = YuyvImage::new(&yuyv, YuyvImage::row_bytes(w), w, h).unwrap();
            let mut enc = Encoder::new(&mut via_yuyv, q);
            enc.set_sampling_factor(img.sampling_factor());
            enc.encode_image(img).unwrap();
            let (_, _, back_yuyv) = decode_rgb(&via_yuyv);

            let (p_rgb, p_yuyv) = (psnr(rgb, &back_rgb), psnr(rgb, &back_yuyv));
            println!(
                "corpus[{i}] {w}x{h} q{q}: rgb path {p_rgb:.2} dB ({} B), yuyv path {p_yuyv:.2} dB ({} B)",
                via_rgb.len(),
                via_yuyv.len()
            );
            assert!(
                p_rgb > 20.0 && p_yuyv > 20.0,
                "both paths reconstruct the picture: {p_rgb:.2} / {p_yuyv:.2} dB"
            );
            assert!(
                (p_rgb - p_yuyv).abs() <= 0.5,
                "PSNR moved {:.2} dB between the RGB and the YUYV path",
                p_rgb - p_yuyv
            );
        }
    }
}
