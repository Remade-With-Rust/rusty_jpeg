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

use rusty_jpeg::decode::{Decoder, Error, UnsupportedFeature};

// ---------------------------------------------------------------------------
// A minimal baseline JPEG writer.
// ---------------------------------------------------------------------------

/// Annex K.3 luminance DC table: code `00` is category 0.
const DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
/// A two-symbol AC table chosen so that ZERO BITS ARE NOT HARMLESS: `0` is
/// run 0 / size 1 and `10` is EOB. Decoding the zero fill inserted after a
/// marker yields a block of -1 coefficients, not an empty block, so a test can
/// tell "left grey" from "decoded the fill".
const AC_BITS: [u8; 16] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const AC_VALS: [u8; 2] = [0x01, 0x00];

/// Canonical Huffman code (Annex C) of `sym` as `(code, length)`.
fn code_for(bits: &[u8; 16], vals: &[u8], sym: u8) -> (u32, u32) {
    let mut code = 0u32;
    let mut k = 0;
    for (i, &n) in bits.iter().enumerate() {
        for _ in 0..n {
            if vals[k] == sym {
                return (code, i as u32 + 1);
            }
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    panic!("symbol {sym} not in table");
}

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
    /// Bits written, excluding stuffing and padding.
    total: usize,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            n: 0,
            total: 0,
        }
    }
    fn put(&mut self, value: u32, len: u32) {
        self.total += len as usize;
        for i in (0..len).rev() {
            self.acc = (self.acc << 1) | ((value >> i) & 1);
            self.n += 1;
            if self.n == 8 {
                let b = self.acc as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0x00);
                }
                self.acc = 0;
                self.n = 0;
            }
        }
    }
    /// Pad to a byte boundary with 1-bits, as a segment ends before an RST.
    fn finish(mut self) -> Vec<u8> {
        while self.n != 0 {
            self.put(1, 1);
        }
        self.out
    }
}

fn segment(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(body);
}

/// DC quantizer 8 and AC quantizers 1, so a DC-only block with DC value `d`
/// decodes to the flat sample value `128 + d`.
fn qtable() -> [u8; 64] {
    let mut q = [1u8; 64];
    q[0] = 8;
    q
}

fn dqt8(q: &[u8; 64]) -> Vec<u8> {
    let mut b = vec![0x00];
    b.extend_from_slice(q);
    b
}

fn dht() -> Vec<u8> {
    let mut b = vec![0x00];
    b.extend_from_slice(&DC_BITS);
    b.extend_from_slice(&DC_VALS);
    b.push(0x10);
    b.extend_from_slice(&AC_BITS);
    b.extend_from_slice(&AC_VALS);
    b
}

/// SOI, DQT, SOF0, DHT, optional DRI and SOS for an 8-bit baseline image whose
/// components all use 1x1 sampling, quantization table 0 and Huffman tables 0.
fn header(w: u16, h: u16, ncomp: u8, dri: u16, dqt_body: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, 0xD8];
    segment(&mut v, 0xDB, dqt_body);
    let mut sof = vec![8];
    sof.extend_from_slice(&h.to_be_bytes());
    sof.extend_from_slice(&w.to_be_bytes());
    sof.push(ncomp);
    for c in 0..ncomp {
        sof.extend_from_slice(&[c + 1, 0x11, 0]);
    }
    segment(&mut v, 0xC0, &sof);
    segment(&mut v, 0xC4, &dht());
    if dri != 0 {
        segment(&mut v, 0xDD, &dri.to_be_bytes());
    }
    let mut sos = vec![ncomp];
    for c in 0..ncomp {
        sos.extend_from_slice(&[c + 1, 0x00]);
    }
    sos.extend_from_slice(&[0, 63, 0]);
    segment(&mut v, 0xDA, &sos);
    v
}

/// Write one DC-only block: DC difference `diff`, then EOB.
fn put_block(bw: &mut BitWriter, diff: i32) {
    let cat = 32 - diff.unsigned_abs().leading_zeros();
    let (c, l) = code_for(&DC_BITS, &DC_VALS, cat as u8);
    bw.put(c, l);
    if cat > 0 {
        let extra = if diff > 0 {
            diff as u32
        } else {
            (diff - 1) as u32 & ((1 << cat) - 1)
        };
        bw.put(extra, cat);
    }
    let (c, l) = code_for(&AC_BITS, &AC_VALS, 0x00);
    bw.put(c, l);
}

/// The DC value of the luma block of MCU `k`: distinct from its neighbours,
/// so a block decoded out of place is visible. Sample value `128 + luma_dc(k)`.
fn luma_dc(k: usize) -> i32 {
    10 + 5 * (k % 16) as i32
}

/// Write MCUs `range` (luma `luma_dc`, chroma DC 0) as one entropy segment,
/// with the DC prediction starting from 0.
fn put_mcus(bw: &mut BitWriter, range: core::ops::Range<usize>, ncomp: u8) {
    let mut pred = 0;
    for m in range {
        put_block(bw, luma_dc(m) - pred);
        pred = luma_dc(m);
        for _ in 1..ncomp {
            put_block(bw, 0);
        }
    }
}

/// The entropy-coded data of an image with `mcus` MCUs and restart interval
/// `dri`: one byte-aligned segment per interval, or a single one if `dri` is 0.
fn segments(mcus: usize, ncomp: u8, dri: usize) -> Vec<Vec<u8>> {
    let per = if dri == 0 { mcus } else { dri };
    (0..mcus)
        .step_by(per)
        .map(|k| {
            let mut bw = BitWriter::new();
            put_mcus(&mut bw, k..(k + per).min(mcus), ncomp);
            bw.finish()
        })
        .collect()
}

/// Header, then each segment followed by `between(i)` (normally RST(i)),
/// then EOI.
fn assemble(header: &[u8], segs: &[Vec<u8>], between: impl Fn(usize) -> Vec<u8>) -> Vec<u8> {
    let mut v = header.to_vec();
    for (i, s) in segs.iter().enumerate() {
        v.extend_from_slice(s);
        if i + 1 < segs.len() {
            v.extend(between(i));
        }
    }
    v.extend_from_slice(&[0xFF, 0xD9]);
    v
}

fn rst(i: usize) -> Vec<u8> {
    vec![0xFF, 0xD0 + (i % 8) as u8]
}

/// A decoded luma plane: samples (stride = width), width.
type Plane = (Vec<u8>, usize);

fn luma(data: &[u8], single_threaded: bool) -> Result<Plane, Error> {
    let mut d = Decoder::new(data);
    d.set_single_threaded(single_threaded);
    let p = d.decode_planar()?;
    let c = &p.components[0];
    let mut out = Vec::with_capacity(c.width * c.height);
    for y in 0..c.height {
        out.extend_from_slice(&c.data[y * c.stride..y * c.stride + c.width]);
    }
    Ok((out, c.width))
}

/// The 8x8 luma block of MCU `k` (MCUs are 8x8: all sampling is 1x1).
fn block(plane: &Plane, k: usize) -> Vec<u8> {
    let (data, w) = plane;
    let (bx, by) = (k % (w / 8), k / (w / 8));
    (0..8)
        .flat_map(|y| {
            let o = (by * 8 + y) * w + bx * 8;
            data[o..o + 8].to_vec()
        })
        .collect()
}

/// Asserts the damaged decode equals the clean one at every MCU except those
/// in `grey`, where every sample must be exactly 128, and in `skip`.
fn assert_recovered(damaged: &Plane, clean: &Plane, mcus: usize, grey: &[usize], skip: &[usize]) {
    for k in 0..mcus {
        if skip.contains(&k) {
            continue;
        }
        let got = block(damaged, k);
        if grey.contains(&k) {
            assert!(
                got.iter().all(|&s| s == 128),
                "MCU {k} should be uniform grey (a lost segment), got {got:?}"
            );
        } else {
            assert_eq!(
                got,
                block(clean, k),
                "MCU {k} differs from the clean decode"
            );
            assert!(
                got.iter().all(|&s| s as i32 == 128 + luma_dc(k)),
                "MCU {k} does not show its own value: {got:?}"
            );
        }
    }
}

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

/// A 136x136 image with restart interval `dri`. Grayscale exercises the
/// buffered, non-interleaved path; three components the interleaved one (the
/// fused entropy+IDCT path when single-threaded). 136x136 is over the
/// multithreaded worker's threshold, so the other threading mode exercises it.
struct Layout {
    ncomp: u8,
    mcus: usize,
    hdr: Vec<u8>,
    segs: Vec<Vec<u8>>,
}

impl Layout {
    fn new(ncomp: u8, dri: u16) -> Self {
        let (w, h) = (136u16, 136u16);
        let mcus = (w as usize / 8) * (h as usize / 8);
        Layout {
            ncomp,
            mcus,
            hdr: header(w, h, ncomp, dri, &dqt8(&qtable())),
            segs: segments(mcus, ncomp, dri as usize),
        }
    }

    fn clean(&self) -> Vec<u8> {
        assemble(&self.hdr, &self.segs, rst)
    }

    /// Decode `data` in both threading modes, which must agree, and check the
    /// packed (colour-converted) output path accepts it too.
    fn decode(&self, data: &[u8]) -> Plane {
        let a = luma(data, true).unwrap_or_else(|e| panic!("ncomp={}: {e}", self.ncomp));
        let b = luma(data, false).unwrap_or_else(|e| panic!("ncomp={}: {e}", self.ncomp));
        assert_eq!(a, b, "ncomp={}: threading modes disagree", self.ncomp);
        Decoder::new(data)
            .decode()
            .unwrap_or_else(|e| panic!("ncomp={} packed: {e}", self.ncomp));
        a
    }
}

fn layouts(dri: u16) -> [Layout; 2] {
    [Layout::new(1, dri), Layout::new(3, dri)]
}

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
