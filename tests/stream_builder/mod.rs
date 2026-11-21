//! A minimal JPEG writer for the damaged-stream and truncated-stream tests.
//!
//! Every stream those tests decode is built here, byte by byte, so each test
//! states exactly which defect its stream carries. The Huffman tables are
//! chosen so that the zero bits a decoder inserts once a segment's data has
//! run out do NOT decode to an empty block (see `AC_BITS`): a test can
//! therefore tell "left grey" from "decoded the fill".
#![allow(dead_code)]

use rusty_jpeg::decode::{Decoder, Error};

// ---------------------------------------------------------------------------
// A minimal baseline JPEG writer.
// ---------------------------------------------------------------------------

/// Annex K.3 luminance DC table: code `00` is category 0.
pub const DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
pub const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
/// A two-symbol AC table chosen so that ZERO BITS ARE NOT HARMLESS: `0` is
/// run 0 / size 1 and `10` is EOB. Decoding the zero fill inserted after a
/// marker yields a block of -1 coefficients, not an empty block, so a test can
/// tell "left grey" from "decoded the fill".
pub const AC_BITS: [u8; 16] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
pub const AC_VALS: [u8; 2] = [0x01, 0x00];

/// Canonical Huffman code (Annex C) of `sym` as `(code, length)`.
pub fn code_for(bits: &[u8; 16], vals: &[u8], sym: u8) -> (u32, u32) {
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

pub struct BitWriter {
    pub out: Vec<u8>,
    pub acc: u32,
    pub n: u32,
    /// Bits written, excluding stuffing and padding.
    pub total: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            n: 0,
            total: 0,
        }
    }
    pub fn put(&mut self, value: u32, len: u32) {
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
    pub fn finish(mut self) -> Vec<u8> {
        while self.n != 0 {
            self.put(1, 1);
        }
        self.out
    }
}

pub fn segment(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(body);
}

/// DC quantizer 8 and AC quantizers 1, so a DC-only block with DC value `d`
/// decodes to the flat sample value `128 + d`.
pub fn qtable() -> [u8; 64] {
    let mut q = [1u8; 64];
    q[0] = 8;
    q
}

pub fn dqt8(q: &[u8; 64]) -> Vec<u8> {
    let mut b = vec![0x00];
    b.extend_from_slice(q);
    b
}

pub fn dht() -> Vec<u8> {
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
pub fn header(w: u16, h: u16, ncomp: u8, dri: u16, dqt_body: &[u8]) -> Vec<u8> {
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
pub fn put_block(bw: &mut BitWriter, diff: i32) {
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
pub fn luma_dc(k: usize) -> i32 {
    10 + 5 * (k % 16) as i32
}

/// Write MCUs `range` (luma `luma_dc`, chroma DC 0) as one entropy segment,
/// with the DC prediction starting from 0.
pub fn put_mcus(bw: &mut BitWriter, range: core::ops::Range<usize>, ncomp: u8) {
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
pub fn segments(mcus: usize, ncomp: u8, dri: usize) -> Vec<Vec<u8>> {
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
pub fn assemble(header: &[u8], segs: &[Vec<u8>], between: impl Fn(usize) -> Vec<u8>) -> Vec<u8> {
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

pub fn rst(i: usize) -> Vec<u8> {
    vec![0xFF, 0xD0 + (i % 8) as u8]
}

/// A decoded luma plane: samples (stride = width), width.
pub type Plane = (Vec<u8>, usize);

pub fn luma(data: &[u8], single_threaded: bool) -> Result<Plane, Error> {
    dump(data);
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
pub fn block(plane: &Plane, k: usize) -> Vec<u8> {
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
pub fn assert_recovered(
    damaged: &Plane,
    clean: &Plane,
    mcus: usize,
    grey: &[usize],
    skip: &[usize],
) {
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

/// A 136x136 image with restart interval `dri`. Grayscale exercises the
/// buffered, non-interleaved path; three components the interleaved one (the
/// fused entropy+IDCT path when single-threaded). 136x136 is over the
/// multithreaded worker's threshold, so the other threading mode exercises it.
pub struct Layout {
    pub ncomp: u8,
    pub mcus: usize,
    pub hdr: Vec<u8>,
    pub segs: Vec<Vec<u8>>,
}

impl Layout {
    pub fn new(ncomp: u8, dri: u16) -> Self {
        let (w, h) = (136u16, 136u16);
        let mcus = (w as usize / 8) * (h as usize / 8);
        Layout {
            ncomp,
            mcus,
            hdr: header(w, h, ncomp, dri, &dqt8(&qtable())),
            segs: segments(mcus, ncomp, dri as usize),
        }
    }

    pub fn clean(&self) -> Vec<u8> {
        assemble(&self.hdr, &self.segs, rst)
    }

    /// Decode `data` in both threading modes, which must agree, and check the
    /// packed (colour-converted) output path accepts it too.
    pub fn decode(&self, data: &[u8]) -> Plane {
        let a = luma(data, true).unwrap_or_else(|e| panic!("ncomp={}: {e}", self.ncomp));
        let b = luma(data, false).unwrap_or_else(|e| panic!("ncomp={}: {e}", self.ncomp));
        assert_eq!(a, b, "ncomp={}: threading modes disagree", self.ncomp);
        Decoder::new(data)
            .decode()
            .unwrap_or_else(|e| panic!("ncomp={} packed: {e}", self.ncomp));
        a
    }
}

pub fn layouts(dri: u16) -> [Layout; 2] {
    [Layout::new(1, dri), Layout::new(3, dri)]
}

// ---------------------------------------------------------------------------
// Frames with more than one scan: progressive, lossless, non-interleaved.
// ---------------------------------------------------------------------------

/// SOI, DQT, a frame header with marker `sof` (8-bit samples, every component
/// 1x1 sampling and quantization table 0), DHT and an optional DRI: everything
/// before the first scan header.
pub fn frame(sof: u8, w: u16, h: u16, ncomp: u8, dri: u16, dqt_body: &[u8]) -> Vec<u8> {
    frame_with(sof, w, h, ncomp, dri, dqt_body, &dht())
}

/// [`frame`] with the DHT segment body given.
pub fn frame_with(
    sof: u8,
    w: u16,
    h: u16,
    ncomp: u8,
    dri: u16,
    dqt_body: &[u8],
    dht_body: &[u8],
) -> Vec<u8> {
    let mut v = vec![0xFF, 0xD8];
    segment(&mut v, 0xDB, dqt_body);
    let mut body = vec![8];
    body.extend_from_slice(&h.to_be_bytes());
    body.extend_from_slice(&w.to_be_bytes());
    body.push(ncomp);
    for c in 0..ncomp {
        body.extend_from_slice(&[c + 1, 0x11, 0]);
    }
    segment(&mut v, sof, &body);
    segment(&mut v, 0xC4, dht_body);
    if dri != 0 {
        segment(&mut v, 0xDD, &dri.to_be_bytes());
    }
    v
}

/// A scan header for the components `ids` (Huffman tables 0), with spectral
/// selection `ss..=se` (in a lossless frame `ss` is the predictor) and the
/// successive-approximation byte `ah_al`.
pub fn scan_header(ids: &[u8], ss: u8, se: u8, ah_al: u8) -> Vec<u8> {
    let mut body = vec![ids.len() as u8];
    for &id in ids {
        body.extend_from_slice(&[id, 0x00]);
    }
    body.extend_from_slice(&[ss, se, ah_al]);
    let mut v = Vec::new();
    segment(&mut v, 0xDA, &body);
    v
}

/// Write a DC difference, or a lossless sample difference: its magnitude
/// category under the DC table, then that many bits. No EOB.
pub fn put_diff(bw: &mut BitWriter, diff: i32) {
    put_diff_with(bw, &DC_BITS, &DC_VALS, diff)
}

/// [`put_diff`] under the Huffman table `bits` / `vals`.
pub fn put_diff_with(bw: &mut BitWriter, bits: &[u8; 16], vals: &[u8], diff: i32) {
    let cat = 32 - diff.unsigned_abs().leading_zeros();
    let (c, l) = code_for(bits, vals, cat as u8);
    bw.put(c, l);
    if cat > 0 {
        let extra = if diff > 0 {
            diff as u32
        } else {
            (diff - 1) as u32 & ((1 << cat) - 1)
        };
        bw.put(extra, cat);
    }
}

/// Every component plane of the decoded image, cropped to its width, plus the
/// decoder's warnings. Decoded in both threading modes, which must agree.
pub fn planes(data: &[u8]) -> Result<(Vec<Plane>, Vec<rusty_jpeg::decode::Warning>), Error> {
    dump(data);
    let decode = |single: bool| {
        let mut d = Decoder::new(data);
        d.set_single_threaded(single);
        let p = d.decode_planar()?;
        let planes: Vec<Plane> = p
            .components
            .iter()
            .map(|c| {
                let mut out = Vec::with_capacity(c.width * c.height);
                for y in 0..c.height {
                    out.extend_from_slice(&c.data[y * c.stride..y * c.stride + c.width]);
                }
                (out, c.width)
            })
            .collect();
        Ok::<_, Error>((planes, d.warnings().to_vec()))
    };
    let single = decode(true);
    let threaded = decode(false);
    match (single, threaded) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b, "threading modes disagree");
            Ok(a)
        }
        (Err(a), Err(b)) => {
            assert_eq!(format!("{a}"), format!("{b}"), "threading modes disagree");
            Err(a)
        }
        (a, b) => panic!(
            "threading modes disagree: {:?} / {:?}",
            a.map(|x| x.1),
            b.map(|x| x.1)
        ),
    }
}

/// With `RUSTY_JPEG_DUMP_STREAMS=<dir>` set, every stream these helpers decode
/// is also written to `<dir>` (named by a hash of its bytes), so the whole set
/// can be put through another decoder -- libjpeg-turbo's `djpeg`, say -- and
/// the outputs compared. That is how the expectations in these tests were
/// checked against libjpeg-turbo.
pub fn dump(data: &[u8]) {
    if let Some(dir) = std::env::var_os("RUSTY_JPEG_DUMP_STREAMS") {
        let hash = data.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        let path = std::path::Path::new(&dir).join(format!("{hash:016x}.jpg"));
        std::fs::write(path, data).expect("RUSTY_JPEG_DUMP_STREAMS: cannot write");
    }
}
