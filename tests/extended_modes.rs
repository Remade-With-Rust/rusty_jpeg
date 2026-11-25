//! 12-bit DCT frames (SOF1, SOF2), arithmetic coding (SOF9, SOF10, DAC) and
//! two-component lossless frames (SOF3, as in DNG).
//!
//! The streams in `fixtures/modes` were written by libjpeg-turbo 3.1.0 from
//! synthetic pictures, and the expected hashes are of what it decodes them to:
//! both with `tools/generate_modes_libjpeg.c`.

use rusty_jpeg::decode::{CodingProcess, Decoder, Error, PixelFormat, UnsupportedFeature, Warning};

macro_rules! fixture {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!("fixtures/modes/", $name, ".jpg")) as &[u8],
        )
    };
}

const FIXTURES: [(&str, &[u8]); 12] = [
    fixture!("lossless12_pair_scans"),
    fixture!("lossless14_pair"),
    fixture!("prog12_420_arithmetic"),
    fixture!("prog12_422_huffman"),
    fixture!("prog8_422_arithmetic"),
    fixture!("prog8_422_huffman"),
    fixture!("seq12_420_huffman"),
    fixture!("seq12_440_separate"),
    fixture!("seq12_grey_arithmetic"),
    fixture!("seq12_ycck"),
    fixture!("seq8_420_arithmetic"),
    fixture!("seq8_420_huffman"),
];

fn stream(name: &str) -> &'static [u8] {
    FIXTURES.iter().find(|f| f.0 == name).unwrap().1
}

fn hash(samples: impl IntoIterator<Item = u16>) -> u64 {
    samples
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .fold(0xcbf29ce484222325, |h, byte| {
            (h ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

/// Decode a complete stream, refusing any repair.
fn samples(bytes: &[u8]) -> Vec<u16> {
    let mut d = Decoder::new(bytes);
    d.set_strict(true);
    d.decode_u16().unwrap()
}

#[test]
fn wide_modes_match_libjpeg_turbo_exactly() {
    for (name, want) in [
        ("seq12_420_huffman", 0x6003099946f83b7b),
        ("prog12_420_arithmetic", 0x6003099946f83b7b),
        ("prog12_422_huffman", 0xbe2837924bcde433),
        ("seq12_440_separate", 0x05a09438ce84443d),
        ("seq12_grey_arithmetic", 0xa690aee45a7369a1),
        ("lossless14_pair", 0x8203cc02383bc24e),
        ("lossless12_pair_scans", 0x9e7427b33cbbfb3a),
    ] {
        assert_eq!(hash(samples(stream(name))), want, "{name}");
    }
    // Four components come out complemented (see `twelve::compute_image`).
    let cmyk = samples(stream("seq12_ycck"));
    assert_eq!(hash(cmyk.iter().map(|s| 4095 - s)), 0x194e03e3981d20e2);
}

/// An arithmetic-coded file holds the same coefficients as its Huffman twin.
#[test]
fn arithmetic_matches_huffman() {
    for kind in ["seq8_420", "prog8_422"] {
        let huffman = samples(stream(&format!("{kind}_huffman")));
        let arithmetic = samples(stream(&format!("{kind}_arithmetic")));
        assert!(huffman == arithmetic, "{kind}");

        let planar = |coding: &str| {
            let mut d = Decoder::new(stream(&format!("{kind}_{coding}")));
            d.set_single_threaded(true);
            let image = d.decode_planar().unwrap();
            image.into_planes()
        };
        assert!(planar("huffman") == planar("arithmetic"), "{kind} planar");
    }
}

#[test]
fn headers_and_sample_layout() {
    let mut d = Decoder::new(stream("seq12_420_huffman"));
    d.read_info().unwrap();
    let h = d.frame_header().unwrap();
    assert_eq!((h.width, h.height, h.components), (35, 23, 3));
    assert_eq!((h.precision, h.bytes_per_sample()), (12, 2));
    assert_eq!(h.coding_process, CodingProcess::DctSequential);
    assert!(!h.arithmetic_coding);
    assert_eq!(d.info().unwrap().pixel_format, PixelFormat::RGB24);
    let wide = samples(stream("seq12_420_huffman"));
    assert_eq!(wide.len(), 35 * 23 * 3);
    assert!(wide.iter().all(|&s| s <= 4095));
    // `decode` gives the same samples as native-endian bytes.
    let bytes: Vec<u8> = wide.iter().flat_map(|s| s.to_ne_bytes()).collect();
    assert_eq!(d.decode().unwrap(), bytes);

    let mut d = Decoder::new(stream("seq12_grey_arithmetic"));
    d.read_info().unwrap();
    assert_eq!(d.info().unwrap().pixel_format, PixelFormat::L16);
    assert!(d.frame_header().unwrap().arithmetic_coding);

    let mut d = Decoder::new(stream("seq8_420_arithmetic"));
    let eight = d.decode().unwrap();
    assert_eq!(eight.len(), 35 * 23 * 3);
    assert_eq!(d.frame_header().unwrap().bytes_per_sample(), 1);
    let widened: Vec<u16> = eight.iter().map(|&b| b.into()).collect();
    assert_eq!(samples(stream("seq8_420_arithmetic")), widened);

    // Two components have no pixel format; the samples of a row interleave
    // into one row of the CFA tile.
    let mut d = Decoder::new(stream("lossless14_pair"));
    d.read_info().unwrap();
    assert_eq!(d.info(), None);
    let h = d.frame_header().unwrap();
    assert_eq!((h.width, h.height, h.components), (24, 16, 2));
    assert_eq!(h.precision, 14);
    assert_eq!(d.decode_u16().unwrap().len(), 48 * 16);
}

/// Reduced sizes: the DC of each block at 1/8, and the right dimensions.
#[test]
fn scaled_twelve_bit() {
    for (w, h) in [(5, 3), (9, 6), (18, 12), (35, 23)] {
        let mut d = Decoder::new(stream("seq12_420_huffman"));
        assert_eq!(d.scale(w, h).unwrap(), (w, h));
        let out = d.decode_u16().unwrap();
        assert_eq!(out.len(), usize::from(w) * usize::from(h) * 3);
        assert!(out.iter().all(|&s| s <= 4095));
    }
}

/// Where the segment with this marker starts (at the marker).
fn find(bytes: &[u8], marker: u8) -> usize {
    bytes
        .windows(2)
        .position(|w| w == [0xFF, marker])
        .expect("marker present")
}

fn with_sof(name: &str, from: u8, change: impl Fn(&mut [u8])) -> Vec<u8> {
    let mut bytes = stream(name).to_vec();
    let at = find(&bytes, from);
    change(&mut bytes[at..]);
    bytes
}

fn unsupported(bytes: &[u8]) -> UnsupportedFeature {
    match Decoder::new(bytes).decode() {
        Err(Error::Unsupported(feature)) => feature,
        other => panic!("{:?}", other.map(|v| v.len())),
    }
}

#[test]
fn refusals() {
    // Baseline is 8-bit (B.2.2).
    let bytes = with_sof("seq12_420_huffman", 0xC1, |sof| sof[1] = 0xC0);
    assert!(matches!(
        Decoder::new(&bytes[..]).decode(),
        Err(Error::Format(_))
    ));
    // Other lossy precisions do not exist.
    for precision in [0, 1, 2, 7, 9, 10, 11, 13, 16, 255] {
        let bytes = with_sof("seq12_420_huffman", 0xC1, |sof| sof[4] = precision);
        assert!(Decoder::new(&bytes[..]).decode().is_err(), "{precision}");
    }
    // Lossless arithmetic.
    let bytes = with_sof("lossless14_pair", 0xC3, |sof| sof[1] = 0xCB);
    assert_eq!(
        unsupported(&bytes),
        UnsupportedFeature::ArithmeticEntropyCoding
    );
    // Two components: lossless and full-size only.
    let bytes = with_sof("lossless14_pair", 0xC3, |sof| sof[11] = 0x21);
    assert_eq!(unsupported(&bytes), UnsupportedFeature::ComponentCount(2));
    for sof in [0xC1, 0xC2, 0xC9, 0xCA] {
        let bytes = with_sof("lossless12_pair_scans", 0xC3, |s| s[1] = sof);
        assert_eq!(unsupported(&bytes), UnsupportedFeature::ComponentCount(2));
    }
    // 12-bit planes.
    let mut d = Decoder::new(stream("seq12_420_huffman"));
    assert!(matches!(
        d.decode_planar(),
        Err(Error::Unsupported(UnsupportedFeature::SamplePrecision(12)))
    ));
}

/// `bytes` with a segment inserted after SOI.
fn insert(bytes: &[u8], marker: u8, body: &[u8]) -> Vec<u8> {
    let mut out = bytes[..2].to_vec();
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&(body.len() as u16 + 2).to_be_bytes());
    out.extend_from_slice(body);
    out.extend_from_slice(&bytes[2..]);
    out
}

#[test]
fn dac_segments() {
    let base = stream("prog8_422_arithmetic");
    let want = samples(base);
    // The defaults spelled out, and settings of tables the scans do not use.
    for body in [
        &[][..],
        &[0x00, 0x10, 0x10, 0x05],
        &[0x0F, 0xF0, 0x1F, 0x00, 0x05, 0xFF, 0x12, 0xFF],
    ] {
        assert!(samples(&insert(base, 0xCC, body)) == want, "{body:?}");
    }
    // In a Huffman file it is irrelevant.
    let huffman = stream("seq8_420_huffman");
    assert!(samples(&insert(huffman, 0xCC, &[0x00, 0x21])) == samples(huffman));
    // Odd length, L > U, a class other than 0 and 1.
    for body in [
        &[0x00][..],
        &[0x00, 0x10, 0x10],
        &[0x00, 0x01],
        &[0x20, 0x10],
    ] {
        assert!(
            matches!(
                Decoder::new(&insert(base, 0xCC, body)[..]).decode(),
                Err(Error::Format(_))
            ),
            "{body:?}"
        );
    }
    // Other parameters than the file was coded with decode to something else,
    // or at least do not fail.
    for (a, b) in [(0xF0, 0x3F), (0xFF, 0x00), (0x00, 0xFF)] {
        let _ = decode(&insert(base, 0xCC, &[0x00, a, 0x01, a, 0x10, b, 0x11, b]));
    }
}

/// A bounded decode of a hostile stream: it may fail, and may not panic.
fn decode(bytes: &[u8]) -> Result<(Vec<u16>, Vec<Warning>), Error> {
    let mut d = Decoder::new(bytes);
    d.set_max_decoding_buffer_size(1 << 20);
    d.set_single_threaded(true);
    let samples = d.decode_u16()?;
    Ok((samples, d.warnings().to_vec()))
}

#[test]
fn truncation_anywhere() {
    for (name, bytes) in FIXTURES {
        let whole = samples(bytes).len();
        let first_scan = find(bytes, 0xDA);
        let header = 2 + usize::from(bytes[first_scan + 3]);
        for cut in 0..bytes.len() {
            match decode(&bytes[..cut]) {
                // Anything with scan data decodes to a full image and says
                // what is wrong with it.
                Ok((out, warnings)) => {
                    assert_eq!(out.len(), whole, "{name} cut at {cut}");
                    assert!(warnings.contains(&Warning::Truncated), "{name} {cut}");
                    assert!(cut > first_scan + header, "{name} {cut}");
                }
                Err(Error::UnexpectedEof) => {
                    assert!(cut <= first_scan + header, "{name} cut at {cut}")
                }
                Err(error) => panic!("{name} cut at {cut}: {error}"),
            }
            let mut strict = Decoder::new(&bytes[..cut]);
            strict.set_strict(true);
            assert!(
                matches!(strict.decode(), Err(Error::UnexpectedEof)),
                "{name} {cut}"
            );
        }
        // Only the EOI gone: every sample is there.
        let (out, _) = decode(&bytes[..bytes.len() - 2]).unwrap();
        assert!(out == samples(bytes), "{name} without EOI");
    }
}

#[test]
fn bit_flips_anywhere() {
    for (name, bytes) in FIXTURES {
        let whole = samples(bytes).len();
        let first_scan = find(bytes, 0xDA);
        let mut recovered = 0;
        for at in 2..bytes.len() {
            for bit in [(at * 5) % 8, (at * 5 + 3) % 8] {
                let mut damaged = bytes.to_vec();
                damaged[at] ^= 1 << bit;
                if let Ok((out, _)) = decode(&damaged) {
                    if at > first_scan + 14 {
                        assert_eq!(out.len(), whole, "{name} at {at}");
                        recovered += 1;
                    }
                }
            }
        }
        assert!(recovered > 0, "{name}");
    }
}

/// Every frame type over every fixture's scans and tables, and over zeros.
#[test]
fn mismatched_frame_types() {
    for (_, bytes) in FIXTURES {
        let at = [0xC0, 0xC1, 0xC2, 0xC3, 0xC9, 0xCA]
            .into_iter()
            .find_map(|m| bytes.windows(4).position(|w| w[..3] == [0xFF, m, 0]))
            .unwrap();
        for sof in (0xC0..=0xCF).filter(|m| ![0xC4, 0xC8, 0xCC].contains(m)) {
            for precision in [8, 12, 16] {
                let mut changed = bytes.to_vec();
                changed[at + 1] = sof;
                changed[at + 4] = precision;
                let _ = decode(&changed);
                let scan = find(&changed, 0xDA) + 14;
                for fill in [0x00, 0xFE, 0xFF, 0xAA] {
                    let mut filled = changed.clone();
                    filled[scan..].fill(fill);
                    let _ = decode(&filled);
                }
            }
        }
    }
}

/// The size limit holds before anything is allocated from the header, and
/// counts a 12-bit sample as the two bytes it is.
#[test]
fn allocation_is_bounded() {
    for name in ["seq12_420_huffman", "seq8_420_arithmetic"] {
        for marker in [0xC1, 0xC9] {
            let mut bytes = stream(name).to_vec();
            let Some(at) = bytes
                .windows(4)
                .position(|w| w[..3] == [0xFF, marker, 0] && w[3] == 17)
            else {
                continue;
            };
            bytes[at + 5..at + 9].fill(0xFF);
            assert!(matches!(decode(&bytes), Err(Error::Format(_))), "{name}");
        }
    }
    let limited = |limit| {
        let mut d = Decoder::new(stream("seq12_grey_arithmetic"));
        d.set_max_decoding_buffer_size(limit);
        d.decode().map(|v| v.len())
    };
    assert!(limited(35 * 23 * 2 - 1).is_err());
    // The stored coefficients are bounded too: 5 x 3 blocks.
    assert_eq!(limited(35 * 23 * 2).unwrap(), 35 * 23 * 2);
}

/// Corrupt arithmetic data is reported, and refused in strict mode.
#[test]
fn bad_arithmetic_code_is_a_warning() {
    let bytes = stream("prog12_420_arithmetic");
    let mut reported = 0;
    for seed in 0..40u32 {
        // Noise (free of FF) over the tail of the file, in its last scans.
        let mut damaged = bytes.to_vec();
        let end = damaged.len() - 2;
        let mut state = seed;
        for byte in &mut damaged[end - 300..end] {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8 & 0x7F;
        }
        let Ok((out, warnings)) = decode(&damaged) else {
            continue;
        };
        assert_eq!(out.len(), 35 * 23 * 3);
        if warnings.contains(&Warning::BadArithmeticCode) {
            reported += 1;
            let mut strict = Decoder::new(&damaged[..]);
            strict.set_strict(true);
            assert!(matches!(strict.decode(), Err(Error::Format(_))));
        }
    }
    assert!(reported > 0);
}
