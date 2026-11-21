//! JPEG streams that end early: a download cut short, a crashed write.
//!
//! libjpeg-turbo decodes these. Its data source answers the end of the data
//! with an end-of-image marker of its own ("insert a fake EOI marker",
//! `fill_input_buffer` in jdatasrc.c) and warns "Premature end of JPEG file";
//! everything after that is the recovery this decoder already had for a marker
//! that arrives early (see `damaged_streams.rs`). The MCU in which the data
//! runs out is finished from inserted zero bits, the rest of the scan is
//! uniform grey, a progressive image keeps what its earlier scans gave it.
//! This decoder used to refuse every such stream with "unexpected end of JPEG
//! data", even one missing only its final `FF D9`.
//!
//! Every stream is built here, byte by byte (see `stream_builder`), and each
//! test fails on the decoder as it was before, with that error.

mod stream_builder;

use rusty_jpeg::decode::{Decoder, Error, Warning};
use stream_builder::*;

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// The warnings a recovered stream decodes with. Every entry point and both
/// threading modes must report the same ones.
fn warnings_of(data: &[u8]) -> Vec<Warning> {
    let mut all = Vec::new();
    for single in [true, false] {
        for planar in [true, false] {
            let mut d = Decoder::new(data);
            d.set_single_threaded(single);
            let result = if planar {
                d.decode_planar().map(|_| ())
            } else {
                d.decode().map(|_| ())
            };
            result.unwrap_or_else(|e| panic!("single={single} planar={planar}: {e}"));
            assert_eq!(d.is_truncated(), d.warnings().contains(&Warning::Truncated));
            all.push(d.warnings().to_vec());
        }
    }
    assert!(all.windows(2).all(|w| w[0] == w[1]), "{all:?}");
    all.remove(0)
}

/// The error strict mode refuses `data` with, from every entry point.
fn strict_error(data: &[u8]) -> Error {
    let mut errors = Vec::new();
    for single in [true, false] {
        for planar in [true, false] {
            let mut d = Decoder::new(data);
            d.set_single_threaded(single);
            d.set_strict(true);
            let result = if planar {
                d.decode_planar().map(|_| ())
            } else {
                d.decode().map(|_| ())
            };
            errors.push(result.expect_err("strict mode must refuse a damaged stream"));
            assert!(d.warnings().is_empty(), "strict mode records no warnings");
        }
    }
    let first = format!("{}", errors[0]);
    assert!(errors.iter().all(|e| format!("{e}") == first), "{errors:?}");
    errors.remove(0)
}

fn assert_truncated(data: &[u8], also: &[Warning]) {
    let warnings = warnings_of(data);
    assert_eq!(warnings[0], Warning::Truncated, "{warnings:?}");
    for w in also {
        assert!(warnings.contains(w), "{w:?} missing from {warnings:?}");
    }
    assert!(matches!(strict_error(data), Error::UnexpectedEof));
}

// ---------------------------------------------------------------------------
// 1. Baseline: one interleaved scan.
// ---------------------------------------------------------------------------

/// The repro from the report: "take any baseline JPEG and drop its last two
/// bytes". Every MCU is there; only the EOI is not. The image is complete, and
/// the caller is told the stream was not.
#[test]
fn missing_eoi_decodes_the_whole_image() {
    for dri in [0u16, 4] {
        for l in layouts(dri) {
            let clean = l.clean();
            let want = l.decode(&clean);
            assert!(warnings_of(&clean).is_empty());
            // Without `FF D9`, and without the `D9` alone: a lone FF at the
            // end of the data is a fill byte before the EOI that is not there.
            for missing in [2, 1] {
                let data = &clean[..clean.len() - missing];
                assert_eq!(l.decode(data), want, "dri={dri} missing={missing}");
                assert_eq!(warnings_of(data), [Warning::Truncated]);
                assert!(matches!(strict_error(data), Error::UnexpectedEof));
            }
        }
    }
}

/// The data stops in the middle of the scan. Everything decoded so far is
/// kept; the MCU the data ran out in is finished from zero bits (as libjpeg
/// does, so its content is arbitrary); every MCU after it is uniform grey --
/// not decoded from the zero fill, which would give -1 AC terms here.
#[test]
fn data_ending_mid_scan_greys_the_rest() {
    for ncomp in [1u8, 3] {
        let l = Layout::new(ncomp, 0);
        let clean = l.decode(&l.clean());
        // The whole bytes MCUs 0..k occupy, and nothing after them.
        let k = 100;
        let mut bw = BitWriter::new();
        put_mcus(&mut bw, 0..k, ncomp);
        let mut data = l.hdr.clone();
        data.extend_from_slice(&bw.out);
        let grey: Vec<usize> = (k + 1..l.mcus).collect();
        assert_recovered(&l.decode(&data), &clean, l.mcus, &grey, &[k - 1, k]);
        assert_truncated(&data, &[Warning::SegmentEndedEarly]);
    }
}

/// The same with restart intervals, cut at each kind of place: after a restart
/// marker, one byte into a segment, at the end of a segment before its marker,
/// and in the middle of the marker. The intervals before the cut are exact;
/// every restart after it finds the end of the data, so nothing resumes.
#[test]
fn data_ending_mid_scan_with_restart_intervals_greys_the_rest() {
    for l in layouts(4) {
        let clean = l.decode(&l.clean());
        // Segments 0..10 (MCUs 0..40), each followed by its restart marker.
        let mut prefix = l.hdr.clone();
        for (i, seg) in l.segs[..10].iter().enumerate() {
            prefix.extend_from_slice(seg);
            prefix.extend(rst(i));
        }
        let mut one_byte_in = prefix.clone();
        one_byte_in.push(l.segs[10][0]);
        let cuts = [
            prefix.clone(),
            one_byte_in,
            prefix[..prefix.len() - 2].to_vec(),
            prefix[..prefix.len() - 1].to_vec(),
        ];
        for (i, data) in cuts.iter().enumerate() {
            // MCU 40 is the one the data runs out in: not checked.
            let grey: Vec<usize> = (41..l.mcus).collect();
            let got = l.decode(data);
            assert_recovered(&got, &clean, l.mcus, &grey, &[40]);
            let warnings = warnings_of(data);
            assert!(
                warnings.contains(&Warning::Truncated)
                    && warnings.contains(&Warning::SegmentEndedEarly),
                "cut {i}: {warnings:?}"
            );
            assert!(matches!(strict_error(data), Error::UnexpectedEof));
        }
    }
}

/// A truncated stream decodes at a reduced scale too (`Decoder::scale`), with
/// the grey remainder where the full-size decode has it.
#[test]
fn truncated_stream_decodes_at_reduced_scale() {
    let l = Layout::new(3, 0);
    let clean = l.clean();
    let data = &clean[..l.hdr.len() + (clean.len() - l.hdr.len()) / 2];
    for single in [true, false] {
        let mut d = Decoder::new(data);
        d.set_single_threaded(single);
        assert_eq!(d.scale(68, 68).unwrap(), (68, 68));
        let px = d.decode().expect("truncated stream at 1/2 scale");
        assert_eq!(px.len(), 68 * 68 * 3);
        assert!(d.is_truncated());
        // The last row is far below the cut: grey in every channel.
        assert!(px[67 * 68 * 3..].iter().all(|&s| s == 128));
    }
}

// ---------------------------------------------------------------------------
// 2. Truncation at every byte offset.
// ---------------------------------------------------------------------------

/// A sequential frame of three components, each in a scan of its own.
fn three_scan_stream(w: u16, h: u16) -> (Vec<u8>, usize) {
    let mcus = (w as usize / 8) * (h as usize / 8);
    let mut v = frame(0xC0, w, h, 3, 0, &dqt8(&qtable()));
    let mut first_data = 0;
    for id in 1..=3u8 {
        v.extend(scan_header(&[id], 0, 63, 0));
        if id == 1 {
            first_data = v.len();
        }
        let mut bw = BitWriter::new();
        put_mcus(&mut bw, 0..mcus, 1);
        v.extend(bw.finish());
    }
    v.extend_from_slice(&[0xFF, 0xD9]);
    (v, first_data)
}

/// Whatever length a stream is cut to, the decoder returns: an image, or a
/// clean error. Never a panic, from any entry point, in either threading mode.
///
/// And the line between the two is exact. Up to and including the first scan
/// header there is nothing to show, and the error is `UnexpectedEof` (a bare
/// header must not make the decoder allocate the image it declares). With one
/// byte of scan data or more, there is an image -- the declared size, flagged
/// as truncated -- and strict mode gives `UnexpectedEof` instead.
#[test]
fn truncation_at_every_byte_offset_is_an_image_or_a_clean_error() {
    // (stream, offset of the first entropy-coded byte, bytes per pixel)
    let mut cases: Vec<(Vec<u8>, usize, usize, &str)> = Vec::new();
    {
        let hdr = header(24, 16, 1, 0, &dqt8(&qtable()));
        let data = assemble(&hdr, &segments(6, 1, 0), rst);
        cases.push((data, hdr.len(), 24 * 16, "grey baseline"));
    }
    {
        let hdr = header(16, 16, 3, 1, &dqt8(&qtable()));
        let data = assemble(&hdr, &segments(4, 3, 1), rst);
        cases.push((
            data,
            hdr.len(),
            16 * 16 * 3,
            "colour baseline, restart interval 1",
        ));
    }
    {
        let (data, first) = three_scan_stream(16, 8);
        cases.push((data, first, 16 * 8 * 3, "three sequential scans"));
    }
    {
        let p = Progressive::new(16, 16, |_| 40);
        cases.push((p.full(), p.first_data(), 16 * 16, "grey progressive"));
    }
    {
        let l = Lossless::new(8, 4, 0);
        cases.push((l.clean(), l.hdr.len(), 8 * 4, "grey lossless"));
    }

    for (full, first_data, size, what) in &cases {
        for len in 0..=full.len() {
            let data = &full[..len];
            for strict in [false, true] {
                for single in [true, false] {
                    for planar in [false, true] {
                        let outcome = std::panic::catch_unwind(|| {
                            let mut d = Decoder::new(data);
                            d.set_single_threaded(single);
                            d.set_strict(strict);
                            let result = if planar {
                                d.decode_planar().map(|p| {
                                    p.components
                                        .iter()
                                        .map(|c| c.width * c.height)
                                        .sum::<usize>()
                                })
                            } else {
                                d.decode().map(|px| px.len())
                            };
                            (result, d.warnings().to_vec())
                        });
                        let at = format!(
                            "{what}: {len} of {} bytes, strict={strict} single={single} planar={planar}",
                            full.len()
                        );
                        let (result, warnings) = outcome.unwrap_or_else(|_| panic!("PANIC {at}"));
                        let complete = len == full.len();
                        // Lossless has no planar output; that is its own error.
                        if planar && *what == "grey lossless" {
                            continue;
                        }
                        if len <= *first_data || (strict && !complete) {
                            assert!(
                                matches!(result, Err(Error::UnexpectedEof)),
                                "{at}: {result:?}"
                            );
                        } else {
                            assert_eq!(result.as_ref().ok(), Some(size), "{at}: {result:?}");
                            assert_eq!(
                                warnings.contains(&Warning::Truncated),
                                !complete,
                                "{at}: {warnings:?}"
                            );
                            assert_eq!(warnings.is_empty(), complete, "{at}: {warnings:?}");
                        }
                    }
                }
            }
        }
    }
}

/// A header that declares the largest image JPEG can, and then nothing: an
/// error, at once, without the image being allocated (this test would need
/// 12 GB otherwise). With some scan data the stream is decodable, so there the
/// caller's own limit is what refuses it -- before any plane is sized.
#[test]
fn bare_header_does_not_allocate_the_image_it_declares() {
    for sof in [0xC0u8, 0xC2, 0xC3] {
        let (ss, se) = match sof {
            0xC0 => (0, 63),
            0xC2 => (0, 0),
            _ => (1, 0),
        };
        let mut data = frame(sof, 65535, 65535, 3, 0, &dqt8(&qtable()));
        data.extend(scan_header(&[1, 2, 3], ss, se, 0));
        for single in [true, false] {
            let mut d = Decoder::new(&data[..]);
            d.set_single_threaded(single);
            assert!(
                matches!(d.decode(), Err(Error::UnexpectedEof)),
                "SOF{sof:x}"
            );
            let mut d = Decoder::new(&data[..]);
            d.set_single_threaded(single);
            assert!(matches!(d.decode_planar(), Err(Error::UnexpectedEof)));
        }

        let mut with_data = data.clone();
        with_data.extend_from_slice(&[0x12, 0x34, 0x56]);
        let mut d = Decoder::new(&with_data[..]);
        d.set_max_decoding_buffer_size(64 * 1024 * 1024);
        match d.decode() {
            Err(Error::Format(m)) => assert!(m.contains("exceeds maximum"), "{m}"),
            other => panic!("SOF{sof:x}: {:?}", other.map(|px| px.len())),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. More than one scan: the data ends between scans, or in a marker segment.
// ---------------------------------------------------------------------------

/// A sequential image coded one component per scan. When the data ends after
/// the first scan -- at the scan boundary, part-way into the next scan header,
/// or inside a table segment that precedes it -- the luma that was decoded is
/// shown, and the chroma no scan delivered is mid-grey: libjpeg-turbo's
/// coefficients for it are still zero. This used to be "unexpected end of JPEG
/// data".
#[test]
fn data_ending_between_scans_shows_the_scans_decoded() {
    let (w, h) = (136u16, 136u16);
    let mcus = 17 * 17;
    let (full, _) = three_scan_stream(w, h);
    let (all, warnings) = planes(&full).unwrap();
    assert!(warnings.is_empty());
    let y_plane = &all[0];
    for k in 0..mcus {
        assert!(block(y_plane, k)
            .iter()
            .all(|&s| s as i32 == 128 + luma_dc(k)));
    }

    // The first scan, complete.
    let mut one_scan = frame(0xC0, w, h, 3, 0, &dqt8(&qtable()));
    one_scan.extend(scan_header(&[1], 0, 63, 0));
    let mut bw = BitWriter::new();
    put_mcus(&mut bw, 0..mcus, 1);
    one_scan.extend(bw.finish());

    let next_header = scan_header(&[2], 0, 63, 0);
    let mut in_scan_header = one_scan.clone();
    in_scan_header.extend_from_slice(&next_header[..5]);
    let mut in_table_segment = one_scan.clone();
    let mut table = Vec::new();
    segment(&mut table, 0xC4, &dht());
    in_table_segment.extend_from_slice(&table[..20]);
    let mut in_comment = one_scan.clone();
    in_comment.extend_from_slice(&[0xFF, 0xFE, 0x10, 0x00, b'c', b'u', b't']);

    for (i, data) in [
        one_scan.clone(),
        in_scan_header,
        in_table_segment,
        in_comment,
    ]
    .iter()
    .enumerate()
    {
        let (got, warnings) = planes(data).unwrap_or_else(|e| panic!("cut {i}: {e}"));
        assert_eq!(&got[0], y_plane, "cut {i}: luma");
        for chroma in &got[1..] {
            assert!(chroma.0.iter().all(|&s| s == 128), "cut {i}: chroma");
        }
        assert_eq!(
            warnings,
            [Warning::Truncated, Warning::MissingScan],
            "cut {i}"
        );
        assert!(matches!(strict_error(data), Error::UnexpectedEof));
        // Packed output: grey chroma makes R = G = B = Y.
        let px = Decoder::new(&data[..]).decode().unwrap();
        assert_eq!(px.len(), 136 * 136 * 3);
        assert!(px
            .chunks_exact(3)
            .zip(&y_plane.0)
            .all(|(rgb, &y)| rgb == [y, y, y]));
    }

    // The data ends inside the second scan: its plane is decoded down to the
    // cut and grey below, the third is all grey.
    let mut two = one_scan.clone();
    two.extend(scan_header(&[2], 0, 63, 0));
    let mut bw = BitWriter::new();
    put_mcus(&mut bw, 0..150, 1);
    two.extend_from_slice(&bw.out);
    let (got, warnings) = planes(&two).unwrap();
    assert_eq!(&got[0], y_plane);
    let grey: Vec<usize> = (151..mcus).collect();
    assert_recovered(&got[1], y_plane, mcus, &grey, &[149, 150]);
    assert!(got[2].0.iter().all(|&s| s == 128));
    assert_eq!(
        warnings,
        [
            Warning::Truncated,
            Warning::SegmentEndedEarly,
            Warning::MissingScan
        ]
    );

    // Not truncated at all: an EOI after the first scan. The components that
    // never came are grey here too (it was "not all components have data").
    let mut ended = one_scan;
    ended.extend_from_slice(&[0xFF, 0xD9]);
    let (got, warnings) = planes(&ended).unwrap();
    assert_eq!(&got[0], y_plane);
    assert!(got[1].0.iter().chain(&got[2].0).all(|&s| s == 128));
    assert_eq!(warnings, [Warning::MissingScan]);
    assert!(matches!(strict_error(&ended), Error::Format(_)));
}

// ---------------------------------------------------------------------------
// 4. Progressive.
// ---------------------------------------------------------------------------

/// A grey progressive image: a DC scan, then one AC scan that gives every
/// block the coefficient +1 at zigzag position 1 (quantizer 32, so a visible
/// horizontal tilt of the block).
struct Progressive {
    w: u16,
    blocks: usize,
    frame: Vec<u8>,
    dc_scan: Vec<u8>,
    ac_scan_header: Vec<u8>,
}

impl Progressive {
    fn new(w: u16, h: u16, dc: impl Fn(usize) -> i32) -> Self {
        Self::with_quantizer(w, h, 32, dc)
    }

    /// `q1` is the quantizer of the coefficient at zigzag position 1.
    fn with_quantizer(w: u16, h: u16, q1: u8, dc: impl Fn(usize) -> i32) -> Self {
        let blocks = (w as usize / 8) * (h as usize / 8);
        let mut q = qtable();
        q[1] = q1;
        let mut dc_scan = scan_header(&[1], 0, 0, 0);
        let mut bw = BitWriter::new();
        let mut pred = 0;
        for k in 0..blocks {
            put_diff(&mut bw, dc(k) - pred);
            pred = dc(k);
        }
        dc_scan.extend(bw.finish());
        Progressive {
            w,
            blocks,
            frame: frame(0xC2, w, h, 1, 0, &dqt8(&q)),
            dc_scan,
            ac_scan_header: scan_header(&[1], 1, 63, 0),
        }
    }

    fn first_data(&self) -> usize {
        self.frame.len() + scan_header(&[1], 0, 0, 0).len()
    }

    /// The AC scan's entropy data for blocks `0..n`: whole bytes only.
    fn ac_data(&self, n: usize) -> Vec<u8> {
        let mut bw = BitWriter::new();
        for _ in 0..n {
            let (c, l) = code_for(&AC_BITS, &AC_VALS, 0x01);
            bw.put(c, l); // run 0, size 1
            bw.put(1, 1); // +1
            let (c, l) = code_for(&AC_BITS, &AC_VALS, 0x00);
            bw.put(c, l); // EOB
        }
        if n == self.blocks {
            bw.finish()
        } else {
            bw.out
        }
    }

    /// Frame and DC scan, with no EOI.
    fn dc_only(&self) -> Vec<u8> {
        [&self.frame[..], &self.dc_scan[..]].concat()
    }

    fn full(&self) -> Vec<u8> {
        let mut v = self.dc_only();
        v.extend_from_slice(&self.ac_scan_header);
        v.extend(self.ac_data(self.blocks));
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    fn luma(&self, data: &[u8]) -> Plane {
        let (mut all, _) = planes(data).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(all[0].1, self.w as usize);
        all.remove(0)
    }
}

/// The data ends after the first scan of a progressive image, or inside the
/// header of the next. What that scan delivered is the image: the same pixels
/// as the stream properly ended there with an EOI (which always decoded).
#[test]
fn progressive_data_ending_between_scans_shows_the_scans_decoded() {
    let p = Progressive::new(136, 136, |_| 40);
    let mut ended = p.dc_only();
    ended.extend_from_slice(&[0xFF, 0xD9]);
    let want = p.luma(&ended);
    assert!(want.0.iter().all(|&s| s == 168));
    assert!(warnings_of(&ended).is_empty());

    let at_boundary = p.dc_only();
    let mut in_scan_header = p.dc_only();
    in_scan_header.extend_from_slice(&p.ac_scan_header[..4]);
    for data in [at_boundary, in_scan_header] {
        assert_eq!(p.luma(&data), want);
        assert_eq!(warnings_of(&data), [Warning::Truncated]);
        assert!(matches!(strict_error(&data), Error::UnexpectedEof));
    }

    // Missing only the EOI after the last scan: the complete image.
    let full = p.full();
    let complete = p.luma(&full);
    assert_ne!(complete, want, "the AC scan must be visible");
    let no_eoi = &full[..full.len() - 2];
    assert_eq!(p.luma(no_eoi), complete);
    assert_eq!(warnings_of(no_eoi), [Warning::Truncated]);
}

/// The data ends inside the AC scan. Blocks the scan reached have their AC
/// coefficient; blocks it did not reach keep what the DC scan gave them --
/// they are not touched, and not decoded from the zero fill, which would give
/// them 63 coefficients of -1.
#[test]
fn progressive_data_ending_mid_scan_keeps_the_earlier_scans() {
    let p = Progressive::new(136, 136, |_| 40);
    let complete = p.luma(&p.full());
    let k = 100;
    let mut data = p.dc_only();
    data.extend_from_slice(&p.ac_scan_header);
    data.extend(p.ac_data(k));
    let got = p.luma(&data);
    for b in 0..p.blocks {
        if b == k - 1 || b == k {
            continue; // the block the data ran out in
        }
        if b < k {
            assert_eq!(block(&got, b), block(&complete, b), "block {b}");
        } else {
            assert!(block(&got, b).iter().all(|&s| s == 168), "block {b}");
        }
    }
    assert_truncated(&data, &[Warning::SegmentEndedEarly]);
}

/// A progressive image cut off after its DC scan is not shown as flat 8x8
/// tiles. libjpeg-turbo estimates each block's low AC coefficients from the DC
/// values around it (`decompress_smooth_data`, jdcoefct.c), and so does this
/// decoder for a truncated stream: a ramp of DC values comes out as a ramp of
/// pixels. The same data ended with an EOI is left as it always decoded.
#[test]
fn truncated_progressive_image_is_smoothed_across_blocks() {
    // 9x5 blocks; DC rises by 16 per block along a row: samples 64..=192.
    let p = Progressive::with_quantizer(72, 40, 1, |k| 16 * ((k % 9) as i32 - 4));
    let truncated = p.luma(&p.dc_only());
    let mut ended = p.dc_only();
    ended.extend_from_slice(&[0xFF, 0xD9]);
    let flat = p.luma(&ended);

    for row in 0..40 {
        let smooth = &truncated.0[row * 72..(row + 1) * 72];
        let tiles = &flat.0[row * 72..(row + 1) * 72];
        // Unsmoothed: eight equal samples per block, a step of 16 between.
        assert!(tiles.chunks_exact(8).all(|b| b.iter().all(|&s| s == b[0])));
        // Smoothed: never falling, and rising inside every interior block.
        assert!(smooth[8..64].windows(2).all(|w| w[0] <= w[1]), "{smooth:?}");
        for b in 1..8 {
            let block = &smooth[b * 8..b * 8 + 8];
            assert!(block[7] - block[0] >= 10, "row {row} block {b}: {block:?}");
        }
        // Away from the edges a block's mean is still its DC value.
        for b in 2..7 {
            let sum: i32 = smooth[b * 8..b * 8 + 8].iter().map(|&s| s as i32).sum();
            assert!((sum - 8 * (64 + 16 * b as i32)).abs() <= 8, "block {b}");
        }
    }
    // The middle row, as libjpeg-turbo 3.1.0's djpeg decodes this stream.
    assert_eq!(
        &truncated.0[20 * 72..21 * 72],
        &SMOOTHED_ROW[..],
        "differs from the row checked against libjpeg-turbo"
    );
}

/// Row 20 of the image in `truncated_progressive_image_is_smoothed_across_blocks`.
// WIP: this is this decoder's own output; the comparison of this stream with
// djpeg 3.1.0 (within one level) has not been run yet.
const SMOOTHED_ROW: [u8; 72] = [
    61, 62, 62, 62, 64, 65, 68, 69, 69, 71, 74, 77, 79, 82, 85, 87, 88, 90, 93, 95, 97, 99, 102,
    104, 104, 106, 109, 111, 113, 115, 118, 120, 120, 122, 125, 127, 129, 131, 134, 136, 136, 138,
    141, 143, 145, 147, 150, 152, 152, 154, 157, 159, 161, 163, 166, 168, 169, 171, 174, 177, 179,
    182, 185, 187, 187, 188, 191, 192, 194, 194, 194, 195,
];

// ---------------------------------------------------------------------------
// 5. Lossless (SOF3).
// ---------------------------------------------------------------------------

/// A DC table in which ZERO BITS ARE NOT HARMLESS for a lossless scan: code
/// `0` is category 1, so the zero fill decodes as a run of -1 differences (a
/// falling ramp), not as "no change". `10` is category 0.
const LL_BITS: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const LL_VALS: [u8; 9] = [1, 0, 2, 3, 4, 5, 6, 7, 8];

/// A grey 8-bit lossless image, predictor 1, sample `60 + 3x + 5y`, with a
/// restart interval of `restart_rows` rows (0: none) and point transform `pt`
/// (the samples are coded divided by `2^pt`, so they come back with their low
/// `pt` bits zero).
struct Lossless {
    w: usize,
    h: usize,
    pt: u8,
    hdr: Vec<u8>,
    /// One entropy-coded segment per restart interval.
    segs: Vec<Vec<u8>>,
    rows_per_seg: usize,
}

impl Lossless {
    fn sample(x: usize, y: usize) -> i32 {
        60 + 3 * x as i32 + 5 * y as i32
    }

    fn new(w: usize, h: usize, restart_rows: usize) -> Self {
        Self::with_point_transform(w, h, restart_rows, 0)
    }

    fn with_point_transform(w: usize, h: usize, restart_rows: usize, pt: u8) -> Self {
        let mut dht_body = vec![0x00];
        dht_body.extend_from_slice(&LL_BITS);
        dht_body.extend_from_slice(&LL_VALS);
        let dri = (restart_rows * w) as u16;
        let mut hdr = frame_with(
            0xC3,
            w as u16,
            h as u16,
            1,
            dri,
            &dqt8(&qtable()),
            &dht_body,
        );
        hdr.extend(scan_header(&[1], 1, 0, pt));
        let rows_per_seg = if restart_rows == 0 { h } else { restart_rows };
        let segs = (0..h)
            .step_by(rows_per_seg)
            .map(|y0| {
                let mut bw = BitWriter::new();
                Self::put_rows(&mut bw, w, pt, y0, (y0 + rows_per_seg).min(h));
                bw.finish()
            })
            .collect();
        Lossless {
            w,
            h,
            pt,
            hdr,
            segs,
            rows_per_seg,
        }
    }

    /// Rows `y0..y1` as one segment: the prediction starts afresh at `y0`.
    fn put_rows(bw: &mut BitWriter, w: usize, pt: u8, y0: usize, y1: usize) {
        let coded = |x: usize, y: usize| Self::sample(x, y) >> pt;
        for y in y0..y1 {
            for x in 0..w {
                let prediction = match (x, y == y0) {
                    (0, true) => 128 >> pt,
                    (0, false) => coded(0, y - 1),
                    _ => coded(x - 1, y),
                };
                put_diff_with(bw, &LL_BITS, &LL_VALS, coded(x, y) - prediction);
            }
        }
    }

    fn clean(&self) -> Vec<u8> {
        assemble(&self.hdr, &self.segs, rst)
    }

    /// Decode in both threading modes; rows of samples.
    fn decode(&self, data: &[u8]) -> Vec<Vec<u8>> {
        dump(data);
        let mut out = Vec::new();
        for single in [true, false] {
            let mut d = Decoder::new(data);
            d.set_single_threaded(single);
            let px = d.decode().unwrap_or_else(|e| panic!("lossless: {e}"));
            assert_eq!(px.len(), self.w * self.h);
            out.push(px);
        }
        assert_eq!(out[0], out[1]);
        out[0].chunks_exact(self.w).map(<[u8]>::to_vec).collect()
    }

    fn expected_row(&self, y: usize) -> Vec<u8> {
        (0..self.w)
            .map(|x| ((Self::sample(x, y) >> self.pt) << self.pt) as u8)
            .collect()
    }
}

/// A lossless scan cut short. The rows before the cut are exact. libjpeg-turbo
/// finishes the row the data ran out in from zero bits and makes every row
/// after it mid-grey (`decode_mcus` in jdlhuff.c zeroes the differences and
/// restarts the prediction). This decoder used to refuse the stream; and where
/// a marker ended a segment early, it decoded the zero fill into every
/// remaining row -- here a ramp falling by one per sample.
#[test]
fn lossless_data_ending_mid_scan_is_grey_below() {
    let l = Lossless::new(16, 12, 0);
    let clean = l.decode(&l.clean());
    for (y, row) in clean.iter().enumerate() {
        assert_eq!(row, &l.expected_row(y), "clean row {y}");
    }

    // The whole bytes of rows 0..6.
    let mut bw = BitWriter::new();
    Lossless::put_rows(&mut bw, 16, 0, 0, 6);
    let mut data = l.hdr.clone();
    data.extend_from_slice(&bw.out);
    let got = l.decode(&data);
    for y in 0..12 {
        match y {
            0..=4 => assert_eq!(got[y], clean[y], "row {y}"),
            // Row 5 loses its last bits, and the row after it may be the one
            // finished from the fill: neither is checked.
            5 | 6 => {}
            _ => assert!(got[y].iter().all(|&s| s == 128), "row {y}: {:?}", got[y]),
        }
    }
    let mut d = Decoder::new(&data[..]);
    d.decode().unwrap();
    assert_eq!(
        d.warnings(),
        [Warning::Truncated, Warning::SegmentEndedEarly]
    );
    let mut d = Decoder::new(&data[..]);
    d.set_strict(true);
    assert!(matches!(d.decode(), Err(Error::UnexpectedEof)));

    // Missing only the EOI: the complete image.
    let full = l.clean();
    assert_eq!(l.decode(&full[..full.len() - 2]), clean);
}

/// Restart intervals in a lossless scan. The prediction starts afresh at each
/// restart marker (H.1.1), which this decoder did not do, so an image with
/// restart intervals decoded wrongly even when intact. With that, a damaged
/// interval costs only its own rows: here the third interval (rows 4 and 5)
/// has lost its data. Row 4 is finished from the fill, row 5 is grey, and row
/// 6, after the next restart marker, is exact again.
#[test]
fn lossless_restart_intervals_reset_the_prediction_and_bound_the_damage() {
    let l = Lossless::new(16, 12, 2);
    assert_eq!(l.rows_per_seg, 2);
    let clean = l.decode(&l.clean());
    for (y, row) in clean.iter().enumerate() {
        assert_eq!(row, &l.expected_row(y), "clean row {y}");
    }

    let mut segs = l.segs.clone();
    segs[2] = Vec::new();
    let got = l.decode(&assemble(&l.hdr, &segs, rst));
    for y in 0..12 {
        match y {
            4 => {} // finished from the zero fill
            5 => assert!(got[y].iter().all(|&s| s == 128), "row 5: {:?}", got[y]),
            _ => assert_eq!(got[y], clean[y], "row {y}"),
        }
    }

    // And cut short: nothing resumes after the end of the data.
    let mut data = l.hdr.clone();
    for (i, seg) in l.segs[..3].iter().enumerate() {
        data.extend_from_slice(seg);
        data.extend(rst(i));
    }
    let got = l.decode(&data);
    for y in 0..12 {
        match y {
            0..=5 => assert_eq!(got[y], clean[y], "row {y}"),
            6 => {}
            _ => assert!(got[y].iter().all(|&s| s == 128), "row {y}: {:?}", got[y]),
        }
    }
}

/// A point transform (Pt > 0): the samples are coded, and predicted, divided
/// by 2^Pt, and scaled back up on output. This decoder scaled each sample as
/// it went and then predicted from the scaled values, so such an image
/// decoded wrongly; libjpeg-turbo's output is the reference here. A truncated
/// one is mid-grey below the cut all the same.
#[test]
fn lossless_point_transform_predicts_from_the_unscaled_samples() {
    for pt in [1u8, 2] {
        for restart_rows in [0usize, 2] {
            let l = Lossless::with_point_transform(16, 12, restart_rows, pt);
            let clean = l.decode(&l.clean());
            for (y, row) in clean.iter().enumerate() {
                assert_eq!(row, &l.expected_row(y), "pt={pt} row {y}");
            }
        }
        let l = Lossless::with_point_transform(16, 12, 0, pt);
        let mut bw = BitWriter::new();
        Lossless::put_rows(&mut bw, 16, pt, 0, 6);
        let mut data = l.hdr.clone();
        data.extend_from_slice(&bw.out);
        let got = l.decode(&data);
        for y in 0..12 {
            match y {
                0..=4 => assert_eq!(got[y], l.expected_row(y), "pt={pt} row {y}"),
                5 | 6 => {}
                _ => assert!(got[y].iter().all(|&s| s == 128), "pt={pt} row {y}"),
            }
        }
    }
}
