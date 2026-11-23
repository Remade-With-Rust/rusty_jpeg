//! Property tests over the public encode -> decode round trip.
//!
//! The `roundtrip` fuzz target varies pixels, size, quality, three samplings
//! and the optimize flag. These vary every encoder knob instead, and check
//! oracles the fuzz target does not: both decode paths accept the file, agree
//! on geometry, and a near-lossless encode reproduces its input.
#![cfg(feature = "std")]

use proptest::prelude::*;
use rusty_jpeg::decode::Decoder;
use rusty_jpeg::encode::{ColorType, Encoder, QuantizationTableType, SamplingFactor};
use std::io::Cursor;

/// Every sampling factor the encoder exposes.
const SAMPLINGS: [SamplingFactor; 16] = {
    use SamplingFactor::*;
    [
        F_1_1, F_2_1, F_1_2, F_2_2, F_4_1, F_4_2, F_1_4, F_2_4, R_4_4_4, R_4_4_0, R_4_4_1, R_4_2_2,
        R_4_2_0, R_4_2_1, R_4_1_1, R_4_1_0,
    ]
};

/// Every packed colour type `Encoder::encode` accepts.
const COLORS: [ColorType; 10] = {
    use ColorType::*;
    [
        Luma, Rgb, Rgba, Bgr, Bgra, Ycbcr, Cmyk, CmykAsYcck, Ycck, Yuyv,
    ]
};

/// One encoder configuration.
#[derive(Debug, Clone)]
struct Config {
    width: u16,
    height: u16,
    color: ColorType,
    sampling: SamplingFactor,
    quality: u8,
    progressive: bool,
    restart: u16,
    optimize: bool,
    streaming_optimize: bool,
    trellis: bool,
    push_blocks: bool,
    table: Option<u16>,
}

fn config() -> impl Strategy<Value = Config> {
    (
        (
            1u16..48,
            1u16..48,
            0..COLORS.len(),
            0..SAMPLINGS.len(),
            0u8..=100,
        ),
        (
            any::<bool>(),
            0u16..5,
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
        ),
        proptest::option::of(1u16..=255),
    )
        .prop_map(
            |(
                (width, height, c, s, quality),
                (progressive, restart, optimize, so, trellis, pb),
                table,
            )| Config {
                width,
                height,
                color: COLORS[c],
                sampling: SAMPLINGS[s],
                quality,
                progressive,
                restart,
                optimize,
                streaming_optimize: so,
                trellis,
                push_blocks: pb,
                table,
            },
        )
}

/// Encodes `data` under `cfg`; `None` when the encoder rejects the config.
fn encode(cfg: &Config, data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut enc = Encoder::new(&mut out, cfg.quality);
    enc.set_sampling_factor(cfg.sampling);
    enc.set_progressive(cfg.progressive);
    enc.set_restart_interval(cfg.restart);
    enc.set_optimized_huffman_tables(cfg.optimize);
    enc.set_streaming_optimize(cfg.streaming_optimize);
    enc.set_trellis(cfg.trellis);
    enc.set_push_blocks(cfg.push_blocks);
    if let Some(v) = cfg.table {
        let t = QuantizationTableType::Custom(Box::new([v; 64]));
        enc.set_quantization_tables(t.clone(), t);
    }
    enc.encode(data, cfg.width, cfg.height, cfg.color).ok()?;
    Some(out)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Anything our encoder accepts, both decode paths read back at the
    /// declared size.
    #[test]
    fn encoded_file_decodes(cfg in config(), seed in any::<u64>()) {
        let len = usize::from(cfg.height) * match cfg.color {
            ColorType::Yuyv => usize::from(cfg.width).div_ceil(2) * 4,
            ColorType::Luma => usize::from(cfg.width),
            ColorType::Rgb | ColorType::Bgr | ColorType::Ycbcr => usize::from(cfg.width) * 3,
            _ => usize::from(cfg.width) * 4,
        };
        let mut s = seed | 1;
        let data: Vec<u8> = (0..len).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
        let Some(jpg) = encode(&cfg, &data) else { return Ok(()) };

        let mut d = Decoder::new(Cursor::new(&jpg));
        d.set_single_threaded(true);
        let pixels = d.decode();
        prop_assert!(pixels.is_ok(), "decode: {:?}", pixels.err());
        let info = d.info().unwrap();
        prop_assert_eq!((info.width, info.height), (cfg.width, cfg.height));
        prop_assert_eq!(
            pixels.unwrap().len(),
            usize::from(cfg.width) * usize::from(cfg.height) * info.pixel_format.pixel_bytes()
        );

        let mut d = Decoder::new(Cursor::new(&jpg));
        d.set_single_threaded(true);
        let planar = d.decode_planar();
        prop_assert!(planar.is_ok(), "decode_planar: {:?}", planar.err());
        let planar = planar.unwrap();
        prop_assert_eq!((planar.width, planar.height), (cfg.width, cfg.height));
    }

    /// Quality 100, Luma: every divisor is 1, so the round trip is the
    /// integer DCT pair's error alone, which stays within a few levels.
    #[test]
    fn q100_luma_is_near_lossless(
        w in 1u16..40, h in 1u16..40,
        data in proptest::collection::vec(any::<u8>(), 1600),
        progressive in any::<bool>(), optimize in any::<bool>(),
    ) {
        let cfg = Config {
            width: w, height: h, color: ColorType::Luma, sampling: SamplingFactor::F_1_1,
            quality: 100, progressive, restart: 0, optimize, streaming_optimize: false,
            trellis: false, push_blocks: false, table: None,
        };
        let px = &data[..usize::from(w) * usize::from(h)];
        let jpg = encode(&cfg, px).expect("q100 luma encodes");
        let mut d = Decoder::new(Cursor::new(&jpg));
        let out = d.decode().expect("decodes");
        let worst = px.iter().zip(&out).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        prop_assert!(worst <= 3, "max abs error {}", worst);
    }
}

/// `QuantizationTableType::Custom` takes `u16` values, but the DQT segment is
/// written with 8-bit precision: values above 255 must clamp to 255, never
/// truncate, so the file declares the table the encoder quantized with.
#[test]
fn custom_table_dqt_matches_request() {
    let px = vec![128u8; 16 * 16];
    for v in [1u16, 255, 256, 1000, 2048] {
        let cfg = Config {
            width: 16,
            height: 16,
            color: ColorType::Luma,
            sampling: SamplingFactor::F_1_1,
            quality: 75,
            progressive: false,
            restart: 0,
            optimize: false,
            streaming_optimize: false,
            trellis: false,
            push_blocks: false,
            table: Some(v),
        };
        let jpg = encode(&cfg, &px).unwrap();
        let at = jpg.windows(2).position(|w| w == [0xFF, 0xDB]).expect("DQT");
        // FFDB, Lq (2), Pq|Tq (1), then Q0.
        let pq = jpg[at + 4] >> 4;
        let q0 = if pq == 0 {
            u16::from(jpg[at + 5])
        } else {
            u16::from_be_bytes([jpg[at + 5], jpg[at + 6]])
        };
        assert_eq!(
            q0,
            v.min(255),
            "custom value {v} written to DQT as {q0} (Pq={pq})"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// The streaming optimize route is an implementation detail: it lays the
    /// scan out differently (interleaved) from the materializing route, but
    /// codes the same coefficients, so both must decode to the same pixels.
    #[test]
    fn streaming_optimize_matches_materialized(
        cfg in config(), data in proptest::collection::vec(any::<u8>(), 48 * 48 * 4),
    ) {
        let cfg = Config { optimize: true, progressive: false, color: ColorType::Rgb, ..cfg };
        let px = &data[..usize::from(cfg.width) * usize::from(cfg.height) * 3];
        let a = encode(&Config { streaming_optimize: true, ..cfg.clone() }, px);
        let b = encode(&Config { streaming_optimize: false, ..cfg.clone() }, px);
        let decode = |j: Option<Vec<u8>>| j.map(|j| Decoder::new(Cursor::new(j)).decode().ok());
        prop_assert!(decode(a) == decode(b), "streaming and materialized decode differently");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// The worker-thread decode path must produce the single-threaded
    /// path's bytes exactly.
    #[test]
    fn threaded_decode_matches_single(cfg in config(), seed in any::<u64>()) {
        let cfg = Config { color: ColorType::Rgb, streaming_optimize: false, ..cfg };
        let mut s = seed | 1;
        let len = usize::from(cfg.width) * usize::from(cfg.height) * 3;
        let px: Vec<u8> = (0..len).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
        let Some(jpg) = encode(&cfg, &px) else { return Ok(()) };
        let run = |single| {
            let mut d = Decoder::new(Cursor::new(&jpg));
            d.set_single_threaded(single);
            d.decode().ok()
        };
        prop_assert!(run(true) == run(false), "threaded decode differs");
    }
}
