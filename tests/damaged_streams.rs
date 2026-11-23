//! JPEG streams that libjpeg-turbo decodes and this decoder used to refuse.
//!
//! They come from a damaged-TIFF corpus (JPEG-in-TIFF strips): a quantization
//! table with a zero entry, restart markers out of sequence, and corrupt
//! entropy data that happens to contain `FF xx`, which reads as a marker. In
//! each case libjpeg-turbo warns and decodes what it can; refusing the whole
//! image is the stricter behaviour, not the compatible one.
//!
//! Every stream is built here, byte by byte, so each test states exactly which
//! defect it carries. The expected pixels follow libjpeg-turbo's recovery rules
//! (`jpeg_resync_to_restart` in jdmarker.c, `jpeg_fill_bit_buffer` and
//! `insufficient_data` in jdhuff.c): the MCU in which the data runs out is
//! finished from inserted zero bits, every later MCU of that segment is left
//! uniform grey (128), and decoding resumes, correctly aligned, at the next
//! restart marker it can trust.

mod stream_builder;

use rusty_jpeg::decode::{Decoder, Error, UnsupportedFeature, Warning};
use stream_builder::*;

// ---------------------------------------------------------------------------
// 1. Zero quantizer entries.
// ---------------------------------------------------------------------------

/// libjpeg-turbo stores a zero DQT entry as is (`get_dqt`, jdmarker.c) and
/// multiplies by it, so that coefficient dequantizes to 0. It does NOT clamp it
/// to 1; that clamp is the encoder's (`jpeg_add_quant_table`, jcparam.c).
#[test]
fn zero_quantizer_entry_decodes_and_zeroes_its_coefficient() {
    let segs = segments(1, 1, 0);
    let decode = |q: &[u8; 64]| luma(&assemble(&header(8, 8, 1, 0, &dqt8(q)), &segs, rst), true);

    // Control: DC quantizer 8 gives 128 + 10.
    let (px, _) = decode(&qtable()).unwrap();
    assert!(px.iter().all(|&s| s == 138), "{px:?}");

    // DC quantizer 0 decodes (it used to fail with "quantization table
    // contains element with a zero value"), and the DC coefficient
    // contributes nothing. Clamping to 1 would give 128 + 10/8 instead.
    let mut q = qtable();
    q[0] = 0;
    let (px, _) = decode(&q).expect("zero DC quantizer must decode");
    assert!(px.iter().all(|&s| s == 128), "{px:?}");

    // A zero AC entry where there is no AC data changes nothing.
    let mut q = qtable();
    q[63] = 0;
    let (px, _) = decode(&q).expect("zero AC quantizer must decode");
    assert!(px.iter().all(|&s| s == 138), "{px:?}");
}

#[test]
fn zero_entry_in_16_bit_quantization_table_decodes() {
    // Pq = 1 with 8-bit samples: outside the spec, accepted as libjpeg does.
    let mut body = vec![0x10];
    for i in 0..64 {
        let v: u16 = if i == 0 { 0 } else { 1 };
        body.extend_from_slice(&v.to_be_bytes());
    }
    let data = assemble(&header(8, 8, 1, 0, &body), &segments(1, 1, 0), rst);
    let (px, _) = luma(&data, true).expect("zero entry in a 16-bit table must decode");
    assert!(px.iter().all(|&s| s == 128), "{px:?}");
}

// ---------------------------------------------------------------------------
// 2. Restart markers out of sequence, and corrupt data posing as a marker.
// ---------------------------------------------------------------------------

#[test]
fn clean_restart_stream_decodes() {
    for l in layouts(1) {
        let clean = l.decode(&l.clean());
        assert_recovered(&clean, &clean, l.mcus, &[], &[]);
    }
}

/// An RST more than two counts away from the expected one is not trusted to
/// resynchronise on: libjpeg discards it and carries on (action 1). This used
/// to fail as "found RST6 where RST2 was expected".
#[test]
fn restart_marker_with_wrong_number_is_discarded() {
    for l in layouts(1) {
        let clean = l.decode(&l.clean());
        // Boundary 10 should carry RST2; it carries RST6 (4 counts away).
        let data = assemble(&l.hdr, &l.segs, |i| if i == 10 { rst(14) } else { rst(i) });
        assert_recovered(&l.decode(&data), &clean, l.mcus, &[], &[]);
    }
}

/// A dropped RST: the decoder skips forward to the next marker, finds an RST
/// one count ahead, and waits for it (action 3). The segment skipped over is
/// lost; everything after it is aligned again. This used to fail as "found
/// RST5 where RST4 was expected".
#[test]
fn missing_restart_marker_loses_one_segment_only() {
    for l in layouts(1) {
        let clean = l.decode(&l.clean());
        // No RST after segment 20. MCU 20 decodes (its data is intact); the
        // decoder then wants RST4, skips segment 21's bytes looking for it,
        // finds RST5 one ahead and leaves it unread: MCU 21 is an empty
        // segment, finished from zero bits as libjpeg does (not checked).
        let data = assemble(&l.hdr, &l.segs, |i| if i == 20 { vec![] } else { rst(i) });
        assert_recovered(&l.decode(&data), &clean, l.mcus, &[], &[21]);
    }
}

/// Two segments and their markers dropped: the RST found is two counts ahead,
/// so two segments read as empty before the stream lines up again.
#[test]
fn restart_marker_two_ahead_waits_two_segments() {
    for l in layouts(1) {
        let clean = l.decode(&l.clean());
        // ... seg30, RST(32), seg33, RST(33) ...: after MCU 30 the decoder
        // wants RST(30) and finds RST(32), two ahead, and leaves it unread.
        // MCU 31 is finished from zero bits (not checked) and MCU 32, still
        // facing the unread marker, is left grey -- where decoding the fill
        // again would have given -1 AC terms. Then RST(32) matches and MCU 33
        // decodes seg33.
        let mut v = l.hdr.clone();
        for (i, s) in l.segs.iter().enumerate() {
            if i == 31 || i == 32 {
                continue;
            }
            v.extend_from_slice(s);
            if i + 1 < l.segs.len() {
                v.extend(if i == 30 { rst(32) } else { rst(i) });
            }
        }
        v.extend_from_slice(&[0xFF, 0xD9]);
        assert_recovered(&l.decode(&v), &clean, l.mcus, &[32], &[31]);
    }
}

/// An RST one count BEHIND the expected one (a repeated marker): libjpeg scans
/// forward to the next marker and decides again (action 2).
#[test]
fn stale_restart_marker_skips_forward() {
    for l in layouts(1) {
        let clean = l.decode(&l.clean());
        // Boundary 40 carries RST(39) instead of RST(40): behind, so skip
        // forward over segment 41's bytes to RST(41), one ahead: wait. MCU 41
        // is an empty segment (not checked); MCU 42 onwards are aligned again.
        let data = assemble(&l.hdr, &l.segs, |i| if i == 40 { rst(39) } else { rst(i) });
        assert_recovered(&l.decode(&data), &clean, l.mcus, &[], &[41]);
    }
}

/// Corrupt entropy data containing `FF 05`: a reserved code below SOF0 that
/// the decoder calls "RES". It used to fail the image as "found marker RES
/// inside scan where RSTn was expected". The bit reader stops at it; the rest
/// of that segment is lost; at the restart boundary the bogus marker is
/// skipped (action 2) and the real RST resumes with the next segment.
#[test]
fn reserved_marker_code_inside_a_segment_is_skipped() {
    for l in layouts(1) {
        let clean = l.decode(&l.clean());
        let mut segs = l.segs.clone();
        segs[50] = vec![0xFF, 0x05, 0x12, 0x34];
        let data = assemble(&l.hdr, &segs, rst);
        // MCU 50 is decoded from the zero fill (libjpeg does the same for the
        // MCU in which the data runs out), so its AC terms are arbitrary: not
        // checked. Every other MCU must be exact.
        assert_recovered(&l.decode(&data), &clean, l.mcus, &[], &[50]);
    }
}

/// The same defect without restart markers. libjpeg-turbo stops decoding real
/// data at the bogus marker, fills the rest of the scan with grey, and outputs
/// the image; it complains about the unknown marker only afterwards. We used
/// to refuse the image ("RES marker found where not allowed").
#[test]
fn reserved_marker_code_without_restarts_greys_the_rest_of_the_scan() {
    for ncomp in [1u8, 3] {
        let l = Layout::new(ncomp, 0);
        let clean = l.decode(&l.clean());
        // Keep the whole bytes that MCUs 0..k occupy and put `FF 05` and junk
        // after them. MCU k-1's last bits (if not byte-aligned) are cut off.
        let k = 50;
        let mut bw = BitWriter::new();
        put_mcus(&mut bw, 0..k, l.ncomp);
        let prefix = bw.out;
        assert!(l.segs[0].starts_with(&prefix));
        let mut v = l.hdr.clone();
        v.extend_from_slice(&prefix);
        v.extend_from_slice(&[0xFF, 0x05, 0x99, 0x88, 0x77, 0xFF, 0xD9]);
        // The data runs out in MCU k-1 (or, if it was byte-aligned, at the
        // start of MCU k). That MCU is finished from the zero fill, so MCUs
        // k-1 and k are not checked. Everything after is grey, where decoding
        // the fill would have given -1 AC terms.
        let grey: Vec<usize> = (k + 1..l.mcus).collect();
        assert_recovered(&l.decode(&v), &clean, l.mcus, &grey, &[k - 1, k]);
    }
}

/// The scan's data ends early and EOI follows: every remaining restart finds
/// EOI and leaves it unread (action 3), so the rest of the image is grey,
/// apart from the first MCU after the cut, which is finished from zero bits.
/// This used to fail as "found marker EOI inside scan where RST3 was expected".
#[test]
fn eoi_inside_restart_scan_greys_the_remainder() {
    for l in layouts(1) {
        let clean = l.decode(&l.clean());
        let data = assemble(&l.hdr, &l.segs[..60], rst);
        let grey: Vec<usize> = (61..l.mcus).collect();
        assert_recovered(&l.decode(&data), &clean, l.mcus, &grey, &[60]);
    }
}

/// `FF FF 00` inside entropy data is not valid, but libjpeg-turbo reads it as
/// one FF data byte (`jpeg_fill_bit_buffer`). We used to fail with "FF 00
/// found where marker was expected".
#[test]
fn padded_stuffed_ff_inside_entropy_data_is_one_ff_byte() {
    // Two 8x8 luma blocks, DC differences +1024 and -1024: category 11, whose
    // DC code `111111110` makes the first entropy byte FF, stuffed as FF 00.
    let mut bw = BitWriter::new();
    put_block(&mut bw, 1024);
    put_block(&mut bw, -1024);
    let seg = bw.finish();
    assert_eq!(&seg[..2], &[0xFF, 0x00]);
    let hdr = header(16, 8, 1, 0, &dqt8(&qtable()));
    let clean = assemble(&hdr, std::slice::from_ref(&seg), rst);
    let mut padded = hdr.clone();
    padded.push(0xFF);
    padded.extend_from_slice(&seg);
    padded.extend_from_slice(&[0xFF, 0xD9]);
    let want = luma(&clean, true).unwrap();
    assert!(block(&want, 0).iter().all(|&s| s == 255));
    assert!(block(&want, 1).iter().all(|&s| s == 128));
    assert_eq!(luma(&padded, true).expect("FF FF 00 must decode"), want);
}

/// Extra bytes at the end of a segment, before its RST: libjpeg discards them
/// (with a warning) when it looks for the marker. We used to give up when the
/// marker was not within the next few bytes.
#[test]
fn junk_before_restart_marker_is_skipped() {
    for l in layouts(1) {
        let clean = l.decode(&l.clean());
        let data = assemble(&l.hdr, &l.segs, |i| {
            let mut v = if i == 7 { vec![0x11; 40] } else { vec![] };
            v.extend(rst(i));
            v
        });
        assert_recovered(&l.decode(&data), &clean, l.mcus, &[], &[]);
    }
}

/// An RST between marker segments: libjpeg-turbo's `read_markers` ignores it
/// anywhere. We used to fail with "RST found outside of entropy-coded data".
#[test]
fn stray_restart_marker_between_segments_is_ignored() {
    let l = Layout::new(3, 1);
    let clean = l.clean();
    // After SOI (2 bytes) and the DQT segment (69 bytes).
    let mut v = clean[..2 + 69].to_vec();
    v.extend_from_slice(&[0xFF, 0xD3]);
    v.extend_from_slice(&clean[2 + 69..]);
    assert_eq!(l.decode(&v), l.decode(&clean));
}

// ---------------------------------------------------------------------------
// 3. Two components.
// ---------------------------------------------------------------------------

/// A 16x16 frame with the given components and scans, entropy data all-zero
/// blocks (enough of it for any of the layouts used).
fn sof_stream(sof: u8, comps: &[(u8, u8)], scans: &[&[u8]]) -> Vec<u8> {
    let mut v = vec![0xFF, 0xD8];
    segment(&mut v, 0xDB, &dqt8(&qtable()));
    let mut body = vec![8, 0, 16, 0, 16, comps.len() as u8];
    for (i, &(hs, vs)) in comps.iter().enumerate() {
        body.extend_from_slice(&[i as u8 + 1, (hs << 4) | vs, 0]);
    }
    segment(&mut v, sof, &body);
    segment(&mut v, 0xC4, &dht());
    for ids in scans {
        let mut sos = vec![ids.len() as u8];
        for &id in *ids {
            sos.extend_from_slice(&[id, 0x00]);
        }
        let (ss, se) = match sof {
            0xC2 => (0, 0), // progressive DC-only first scan
            0xC3 => (1, 0), // lossless, predictor 1
            _ => (0, 63),
        };
        sos.extend_from_slice(&[ss, se, 0]);
        segment(&mut v, 0xDA, &sos);
        let mut bw = BitWriter::new();
        for _ in 0..16 * 16 * 4 {
            bw.put(0, 2); // DC category 0 / zero lossless difference
            if se == 63 {
                bw.put(0b10, 2); // EOB
            }
        }
        v.extend(bw.finish());
    }
    v.extend_from_slice(&[0xFF, 0xD9]);
    v
}

/// A 2-component frame (grey + alpha, as JPEG-in-TIFF can carry) must be a
/// clean `Unsupported(ComponentCount(2))` from every entry point, never a
/// panic -- including `info()`, whose fallback arms were `panic!()`.
#[test]
fn two_component_frames_are_a_clean_error() {
    for sof in [0xC0u8, 0xC1, 0xC2, 0xC3] {
        for comps in [[(1u8, 1u8), (1, 1)], [(2, 2), (1, 1)], [(1, 1), (2, 1)]] {
            let data = sof_stream(sof, &comps, &[&[1, 2]]);
            let check = |r: Result<(), Error>| match r {
                Err(Error::Unsupported(UnsupportedFeature::ComponentCount(2))) => {}
                other => panic!("SOF{} {comps:?}: {other:?}", sof & 0xF),
            };
            for single in [true, false] {
                let mut d = Decoder::new(&data[..]);
                d.set_single_threaded(single);
                check(d.decode().map(|_| ()));
                assert!(d.info().is_none());
                let mut d = Decoder::new(&data[..]);
                d.set_single_threaded(single);
                check(d.decode_planar().map(|_| ()));
                assert!(d.info().is_none());
            }
            let mut d = Decoder::new(&data[..]);
            check(d.read_info());
            assert!(d.info().is_none());
            check(Decoder::new(&data[..]).scale(4, 4).map(|_| ()));
        }
    }
}

/// Two-component SCANS inside 3- and 4-component frames are ordinary JPEG
/// and must decode.
#[test]
fn two_component_scans_decode() {
    /// (SOF marker, per-component sampling factors, component ids per scan)
    type Case = (u8, &'static [(u8, u8)], &'static [&'static [u8]]);
    let cases: [Case; 5] = [
        (0xC0, &[(1, 1), (1, 1), (1, 1)], &[&[1, 2], &[3]]),
        (0xC0, &[(2, 2), (1, 1), (1, 1)], &[&[1], &[2, 3]]),
        (0xC0, &[(2, 1), (1, 1), (1, 1)], &[&[2, 3], &[1]]),
        (0xC0, &[(1, 1), (1, 1), (1, 1), (1, 1)], &[&[1, 2], &[3, 4]]),
        (0xC2, &[(2, 2), (1, 1), (1, 1)], &[&[1, 2], &[3]]),
    ];
    for (sof, comps, scans) in cases {
        let data = sof_stream(sof, comps, scans);
        for single in [true, false] {
            let mut d = Decoder::new(&data[..]);
            d.set_single_threaded(single);
            let px = d
                .decode()
                .unwrap_or_else(|e| panic!("{comps:?} {scans:?}: {e}"));
            assert_eq!(px.len(), 16 * 16 * comps.len());
            let mut d = Decoder::new(&data[..]);
            d.set_single_threaded(single);
            assert_eq!(d.decode_planar().unwrap().components.len(), comps.len());
        }
    }
}

// ---------------------------------------------------------------------------
// 4. A bit sequence that is no Huffman code.
// ---------------------------------------------------------------------------

/// The entropy data of `segments`, except that in MCU `bad_mcu` the EOB of the
/// luma block is replaced by seventeen 1-bits -- which no code of the AC table
/// (`0`, `10`) begins -- and, if `bad_dc`, so is the DC code of the block
/// after it (a chroma block, DC difference 0).
fn segments_with_bad_code(l: &Layout, dri: usize, bad_mcu: usize, bad_dc: bool) -> Vec<Vec<u8>> {
    let per = if dri == 0 { l.mcus } else { dri };
    let (eob, eob_len) = code_for(&AC_BITS, &AC_VALS, 0x00);
    (0..l.mcus)
        .step_by(per)
        .map(|k| {
            let mut bw = BitWriter::new();
            let mut pred = 0;
            for m in k..(k + per).min(l.mcus) {
                put_diff(&mut bw, luma_dc(m) - pred);
                pred = luma_dc(m);
                if m == bad_mcu {
                    bw.put(0x1FFFF, 17);
                } else {
                    bw.put(eob, eob_len);
                }
                for c in 1..l.ncomp {
                    if m == bad_mcu && bad_dc && c == 1 {
                        bw.put(0x1FFFF, 17);
                        bw.put(eob, eob_len);
                    } else {
                        put_block(&mut bw, 0);
                    }
                }
            }
            bw.finish()
        })
        .collect()
}

/// libjpeg-turbo does not refuse an image for a bad Huffman code. Its decoder
/// reads on until its length search passes 16 bits -- 17 bits consumed --
/// warns "Corrupt JPEG data: bad Huffman code", and takes the symbol to be 0
/// (`jpeg_huff_decode`, jdhuff.c). Symbol 0 is an end-of-block in an AC table
/// and a zero difference in a DC table, so a stream with exactly seventeen
/// stray bits where such a code belongs decodes to the very same image. This
/// used to fail as "failed to decode huffman code".
#[test]
fn bad_huffman_code_reads_as_symbol_zero_after_seventeen_bits() {
    for (ncomp, dri, bad_dc) in [
        (1u8, 0u16, false),
        (3, 0, false),
        (3, 0, true),
        (1, 4, false),
        (3, 4, true),
    ] {
        let l = Layout::new(ncomp, dri);
        let clean = l.decode(&l.clean());
        let segs = segments_with_bad_code(&l, dri as usize, 50, bad_dc);
        let data = assemble(&l.hdr, &segs, rst);
        assert_ne!(data, l.clean());
        assert_recovered(&l.decode(&data), &clean, l.mcus, &[], &[]);

        for single in [true, false] {
            let mut d = Decoder::new(&data[..]);
            d.set_single_threaded(single);
            d.decode().unwrap();
            assert_eq!(d.warnings(), [Warning::BadHuffmanCode]);
            assert!(!d.is_truncated());

            let mut d = Decoder::new(&data[..]);
            d.set_single_threaded(single);
            d.set_strict(true);
            match d.decode() {
                Err(Error::Format(m)) => assert!(m.contains("huffman"), "{m}"),
                other => panic!("strict: {:?}", other.map(|px| px.len())),
            }
        }
    }
}

/// Bad codes that leave the decoder out of step: a segment overwritten with
/// 1-bits. Its MCUs are lost (the first is decoded from whatever the bits
/// mean, the rest are grey once the data runs out), and the next restart
/// marker puts the decoder back in step, as libjpeg-turbo's does.
#[test]
fn bad_huffman_codes_cost_one_restart_interval() {
    for l in layouts(4) {
        let clean = l.decode(&l.clean());
        let mut segs = l.segs.clone();
        // Stuffed FF bytes: 24 one-bits of entropy data, no marker.
        segs[20] = vec![0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00];
        let data = assemble(&l.hdr, &segs, rst);
        // Segment 20 is MCUs 80..84.
        assert_recovered(&l.decode(&data), &clean, l.mcus, &[81, 82, 83], &[80]);
        let mut d = Decoder::new(&data[..]);
        d.decode().unwrap();
        assert_eq!(
            d.warnings(),
            [Warning::SegmentEndedEarly, Warning::BadHuffmanCode]
        );
    }
}

/// Whole bytes of entropy data that no MCU used, before the next marker:
/// libjpeg-turbo warns "extraneous bytes before marker" and decodes on. Two
/// stray bytes are still inside the bit reader's accumulator when the scan or
/// restart interval ends, so this is the `unread_bytes` check, not the marker
/// search that follows it.
#[test]
fn extraneous_bytes_before_marker_warn() {
    for dri in [0u16, 4] {
        for l in layouts(dri) {
            let clean = l.decode(&l.clean());
            let mut segs = l.segs.clone();
            // The only segment, or one in the middle (ended by an RST).
            let at = if dri == 0 { 0 } else { 20 };
            segs[at].extend_from_slice(&[0x12, 0x34]);
            let data = assemble(&l.hdr, &segs, rst);
            assert_eq!(l.decode(&data), clean, "ncomp={} dri={dri}", l.ncomp);

            for single in [true, false] {
                let mut d = Decoder::new(&data[..]);
                d.set_single_threaded(single);
                d.decode().unwrap();
                assert_eq!(d.warnings(), [Warning::ExtraneousBytes], "dri={dri}");

                let mut d = Decoder::new(&data[..]);
                d.set_single_threaded(single);
                d.set_strict(true);
                assert!(d.decode().is_err(), "strict, dri={dri}");
            }
        }
    }
}

/// In an AC refinement scan a newly non-zero coefficient always has size 1.
/// libjpeg-turbo warns "bad Huffman code" about any other size and reads it
/// as 1 (`decode_mcu_AC_refine`); strict mode refuses the image.
#[test]
fn refinement_coefficient_of_size_two_is_a_bad_code() {
    // AC table: `0` is EOB, `1` is run 0 / size 2.
    const BITS: [u8; 16] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const VALS: [u8; 2] = [0x00, 0x02];
    let mut dht = vec![0x00];
    dht.extend_from_slice(&DC_BITS);
    dht.extend_from_slice(&DC_VALS);
    dht.push(0x10);
    dht.extend_from_slice(&BITS);
    dht.extend_from_slice(&VALS);

    let blocks = 4;
    let mut data = frame_with(0xC2, 16, 16, 1, 0, &dqt8(&qtable()), &dht);
    // DC first scan, then an AC first scan at Al=1 that leaves every band empty.
    data.extend(scan_header(&[1], 0, 0, 0x00));
    let mut bw = BitWriter::new();
    for _ in 0..blocks {
        put_diff(&mut bw, 0);
    }
    data.extend(bw.finish());
    data.extend(scan_header(&[1], 1, 63, 0x01));
    let mut bw = BitWriter::new();
    for _ in 0..blocks {
        bw.put(0, 1); // EOB
    }
    data.extend(bw.finish());
    // AC refinement (Ah=1, Al=0): each block gets a new coefficient coded
    // with size 2, then EOB.
    data.extend(scan_header(&[1], 1, 63, 0x10));
    let mut bw = BitWriter::new();
    for _ in 0..blocks {
        bw.put(1, 1); // run 0 / size 2
        bw.put(1, 1); // sign: positive
        bw.put(0, 1); // EOB
    }
    data.extend(bw.finish());
    data.extend_from_slice(&[0xFF, 0xD9]);

    for single in [true, false] {
        let mut d = Decoder::new(&data[..]);
        d.set_single_threaded(single);
        d.decode().unwrap();
        assert_eq!(d.warnings(), [Warning::BadHuffmanCode]);

        let mut d = Decoder::new(&data[..]);
        d.set_single_threaded(single);
        d.set_strict(true);
        match d.decode() {
            Err(Error::Format(m)) => assert!(m.contains("huffman"), "{m}"),
            other => panic!("strict: {:?}", other.map(|px| px.len())),
        }
    }
}
