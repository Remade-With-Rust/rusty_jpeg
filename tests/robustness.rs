//! The decoder must not PANIC on malformed input.
//!
//! A decoder is a parser for hostile bytes. `Err` is the correct outcome for a
//! corrupt file; a panic is not — in a library it takes down the caller's
//! process, and this crate is published for anyone to point at untrusted data.
//!
//! `fuzz/` carries proper cargo-fuzz targets for deep exploration. This is the
//! part that runs on stable, in ordinary CI, on every change: a deterministic
//! mutation schedule over self-generated seeds. Deterministic matters — a
//! failure here reproduces exactly and converts straight into a regression test,
//! rather than being a note that a fuzzer once saw something.
//!
//! Measured baseline when this was written: 80,000 mutated inputs across four
//! seeds produced 38,192 successful decodes, 41,808 clean rejections and
//! **zero panics**.

mod stream_builder;

use rusty_jpeg::decode::Decoder;
use rusty_jpeg::encode::{ColorType, Encoder, SamplingFactor};

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn seed_jpeg(w: usize, h: usize, sampling: SamplingFactor, optimize: bool) -> Vec<u8> {
    seed_jpeg_rst(w, h, sampling, optimize, 0)
}

/// `restart` > 0 sets a restart interval of that many MCUs, so mutations reach
/// the restart-marker resynchronisation (`process_restart`).
fn seed_jpeg_rst(
    w: usize,
    h: usize,
    sampling: SamplingFactor,
    optimize: bool,
    restart: u16,
) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let o = (y * w + x) * 3;
            rgb[o] = ((x * 7 + y * 3) % 256) as u8;
            rgb[o + 1] = ((x * 3 + y * 11) % 256) as u8;
            rgb[o + 2] = ((x * 13 + y * 5) % 256) as u8;
        }
    }
    let mut out = Vec::new();
    let mut enc = Encoder::new(&mut out, 85);
    enc.set_sampling_factor(sampling);
    enc.set_optimized_huffman_tables(optimize);
    if restart > 0 {
        enc.set_restart_interval(restart);
    }
    enc.encode(&rgb, w as u16, h as u16, ColorType::Rgb)
        .expect("seed encode");
    out
}

/// Mutations aimed where a JPEG parser is structurally weakest: marker bytes,
/// segment lengths, and the end of the entropy stream.
fn mutate(src: &[u8], kind: u64, rng: &mut u64) -> Vec<u8> {
    let mut v = src.to_vec();
    if v.len() < 8 {
        return v;
    }
    match kind % 6 {
        0 => {
            let i = (xorshift(rng) as usize) % v.len();
            v[i] ^= 1 << (xorshift(rng) % 8);
        }
        1 => {
            let n = (xorshift(rng) as usize) % v.len();
            v.truncate(n.max(2));
        }
        2 => {
            for i in 2..v.len().saturating_sub(4) {
                if v[i] == 0xFF && !matches!(v[i + 1], 0x00 | 0xD8 | 0xD9) {
                    v[i + 2] = xorshift(rng) as u8;
                    v[i + 3] = xorshift(rng) as u8;
                    break;
                }
            }
        }
        3 => {
            let i = (xorshift(rng) as usize) % (v.len() - 2) + 1;
            v[i] = 0xFF;
            v[i + 1] = (xorshift(rng) % 0xFD) as u8 + 1;
        }
        4 => {
            let i = (xorshift(rng) as usize) % v.len();
            let n = ((xorshift(rng) as usize) % 64).min(v.len() - i);
            for b in &mut v[i..i + n] {
                *b = xorshift(rng) as u8;
            }
        }
        _ => {
            let i = (xorshift(rng) as usize) % v.len();
            let n = ((xorshift(rng) as usize) % 32).min(v.len() - i);
            v[i..i + n].fill(0);
        }
    }
    v
}

/// A real libjpeg progressive file, shipped as a fixture because THIS CRATE
/// CANNOT PRODUCE ONE with the layout that matters: libjpeg emits DHT per scan,
/// so its leading DC-only scan names an AC table defined only later, whereas our
/// encoder writes all four tables up front.
///
/// Its absence from this corpus is exactly why 80,000 mutations over
/// baseline-only seeds found zero panics while a DC-scan `unwrap` was live in
/// two published releases. A fuzz corpus is only as good as the shapes in it.
const LIBJPEG_PROGRESSIVE: &[u8] = include_bytes!("fixtures/progressive_libjpeg.jpg");

/// A lossless (SOF3) image, 24x16 grey with a restart interval of two rows,
/// built byte by byte: the encoder here writes no lossless JPEG, so without
/// this seed no mutation reaches the lossless scan decoder, its restart
/// handling or its fill of rows whose data ran out.
fn seed_lossless() -> Vec<u8> {
    use stream_builder::*;
    let (w, h) = (24usize, 16usize);
    let mut v = frame(
        0xC3,
        w as u16,
        h as u16,
        1,
        (2 * w) as u16,
        &dqt8(&qtable()),
    );
    v.extend(scan_header(&[1], 4, 0, 0));
    for y0 in (0..h).step_by(2) {
        let mut bw = BitWriter::new();
        for i in 0..2 * w {
            // Any differences decode; these keep the samples moving.
            put_diff(&mut bw, ((i * 7 + y0 * 3) % 23) as i32 - 11);
        }
        v.extend(bw.finish());
        if y0 + 2 < h {
            v.extend(rst(y0 / 2));
        }
    }
    v.extend_from_slice(&[0xFF, 0xD9]);
    v
}

/// A sequential image of three components in three scans, the middle one with
/// a restart interval: the multi-scan paths, where a stream cut between scans
/// leaves whole components to be filled.
fn seed_three_scans() -> Vec<u8> {
    use stream_builder::*;
    let (w, h) = (32u16, 24u16);
    let mcus = 4 * 3;
    let mut v = frame(0xC0, w, h, 3, 0, &dqt8(&qtable()));
    for id in 1..=3u8 {
        let dri = if id == 2 { 2 } else { 0 };
        segment(&mut v, 0xDD, &(dri as u16).to_be_bytes());
        v.extend(scan_header(&[id], 0, 63, 0));
        let segs = segments(mcus, 1, dri);
        for (i, seg) in segs.iter().enumerate() {
            v.extend_from_slice(seg);
            if i + 1 < segs.len() {
                v.extend(rst(i));
            }
        }
    }
    v.extend_from_slice(&[0xFF, 0xD9]);
    v
}

#[test]
fn malformed_input_never_panics() {
    let seeds = [
        seed_jpeg(48, 48, SamplingFactor::R_4_2_0, true),
        seed_jpeg(48, 48, SamplingFactor::R_4_4_4, false),
        // Ragged dimensions exercise the partial-MCU edge paths.
        seed_jpeg(37, 23, SamplingFactor::R_4_2_2, true),
        // Progressive, with a scan layout our own encoder never emits.
        LIBJPEG_PROGRESSIVE.to_vec(),
        // Restart markers every 2 MCUs: mutated RSTs, dropped segments and
        // bogus markers drive the libjpeg-style resynchronisation.
        seed_jpeg_rst(48, 40, SamplingFactor::R_4_2_0, false, 2),
        // Lossless, and three scans: built by hand (see above).
        seed_lossless(),
        seed_three_scans(),
    ];
    // The hand-built seeds must be valid streams, or their mutations test
    // only the first error they hit.
    for seed in &seeds[5..] {
        let mut d = Decoder::new(&seed[..]);
        d.decode().expect("hand-built seed decodes");
        assert!(d.warnings().is_empty(), "{:?}", d.warnings());
    }

    // Small images and a modest count so this stays a CI test, not a fuzz run.
    // `fuzz/` is where depth comes from.
    const ITERS: usize = 1500;
    let (mut ok, mut err) = (0u32, 0u32);

    for (s, src) in seeds.iter().enumerate() {
        for i in 0..ITERS {
            let mut rng =
                0x9E3779B97F4A7C15u64 ^ ((s * ITERS + i) as u64).wrapping_mul(0x0F1B_2C3D);
            xorshift(&mut rng);
            let data = mutate(src, i as u64, &mut rng);

            // Both output paths: they assemble results differently and have
            // diverged before.
            let planar = i % 2 == 0;
            // Half the cases (two in every four, both output paths) in strict
            // mode: the refusals are code too.
            let strict = i % 4 >= 2;
            let res = std::panic::catch_unwind(|| {
                let mut d = Decoder::new(&data[..]);
                d.set_single_threaded(true);
                d.set_strict(strict);
                d.set_max_decoding_buffer_size(32 * 1024 * 1024);
                if planar {
                    d.decode_planar().map(|p| p.components.len())
                } else {
                    d.decode().map(|p| p.len())
                }
            });
            match res {
                Ok(Ok(_)) => ok += 1,
                Ok(Err(_)) => err += 1,
                Err(_) => panic!(
                    "PANIC on malformed input: seed={s} case={i} path={} bytes={}",
                    if planar { "planar" } else { "packed" },
                    data.len()
                ),
            }
        }
    }

    // Guard the guard: if every mutation were rejected at the header, this test
    // would pass while exercising almost nothing.
    assert!(
        ok > 100,
        "only {ok} of {} mutations decoded ({err} rejected) - the corpus is not \
         reaching the decoder body, so this test is not testing much",
        seeds.len() * ITERS
    );
}

/// Anything our own encoder emits, our own decoder must read back. This is the
/// shape that would have caught the DHT-ordering defect, where the file declared
/// one set of Huffman tables and coded the scan with another.
#[test]
fn our_encoder_output_always_decodes() {
    for &optimize in &[false, true] {
        for &sampling in &[
            SamplingFactor::R_4_4_4,
            SamplingFactor::R_4_2_2,
            SamplingFactor::R_4_2_0,
        ] {
            for &(w, h) in &[(1usize, 1usize), (8, 8), (17, 9), (64, 64), (127, 65)] {
                let jpg = seed_jpeg(w, h, sampling, optimize);
                let out = Decoder::new(&jpg[..]).decode().unwrap_or_else(|e| {
                    panic!("{w}x{h} optimize={optimize} sampling={sampling:?}: {e}")
                });
                assert_eq!(
                    out.len(),
                    w * h * 3,
                    "{w}x{h} optimize={optimize} sampling={sampling:?}: wrong size"
                );
            }
        }
    }
}
