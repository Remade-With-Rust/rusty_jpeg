//! Arithmetic entropy decoding: the QM-coder of T.81 Annex D and the DCT
//! coefficient models of Annexes F and G.
//!
//! A port of libjpeg-turbo's `jdarith.c` / `jaricom.c` (with their "floating
//! cut-point" arrangement of the C register), down to what it does with corrupt
//! data, so that a damaged stream decodes to what `djpeg` shows.
//!
//! No Huffman code is involved, but `HuffmanDecoder` also holds the state of a
//! scan's data source -- the marker it ran into, whether the data ended -- and
//! the restart resynchronisation works on that, so one is embedded for it.

use crate::decode::decoder::{process_restart, UNZIGZAG};
use crate::decode::error::{Error, Result, Warnings};
use crate::decode::huffman::HuffmanDecoder;
use crate::decode::marker::Marker;
use crate::decode::{Buffered, Source};
use core::ops::Range;

/// Conditioning table destinations (B.2.4.3, `Tb`).
pub(crate) const TABLES: usize = 16;

/// The conditioning parameters a DAC segment sets (B.2.4.3), with the defaults
/// that apply until one does (F.1.4.4.1.4 and F.1.4.4.2.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Conditioning {
    /// Bounds `(L, U)` of the "small difference" DC category.
    pub(crate) dc: [(u8, u8); TABLES],
    /// `Kx`: the last zigzag position of the low-frequency AC category.
    pub(crate) ac: [u8; TABLES],
}

impl Default for Conditioning {
    fn default() -> Self {
        Conditioning {
            dc: [(0, 1); TABLES],
            ac: [5; TABLES],
        }
    }
}

/// Statistics bins per DC and per AC table: 49 and 245 (Tables F.4 and F.5),
/// rounded up as libjpeg does.
const DC_BINS: usize = 64;
const AC_BINS: usize = 256;

/// First of the DC magnitude-category bins, Table F.4 (`X1`).
const DC_X1: usize = 20;
/// First of the AC magnitude-category bins for low and high frequencies,
/// Table F.5 (`X2`).
const AC_X2_LOW: usize = 189;
const AC_X2_HIGH: usize = 217;
/// From a magnitude-category bin to the bin of its bit pattern.
const M_OFFSET: usize = 14;

/// The state of the "fixed probability 0.5" bin, whose estimate never moves.
const FIXED: u8 = 113;

/// The decoding registers.
pub(crate) struct ArithDecoder {
    c: u32,
    a: u32,
    /// Bits of `c` below the cut point; negative while the first bytes of a
    /// segment are still to come.
    ct: i32,
    /// The segment's data has ended at a marker. Unlike in a Huffman segment
    /// that is legal before the last MCU (D.1.8: trailing zero bytes are not
    /// sent), and the decoder supplies zero bytes from there on.
    ended: bool,
    /// A coefficient ran past the end of its band or its magnitude past 15
    /// bits. Nothing more is decoded before the next restart marker.
    bad: bool,
    /// A `bad` was met at any time in the scan.
    saw_bad_code: bool,
    source: HuffmanDecoder,
}

impl ArithDecoder {
    pub(crate) fn new(strict: bool) -> ArithDecoder {
        ArithDecoder {
            c: 0,
            a: 0,
            ct: -16,
            ended: false,
            bad: false,
            saw_bad_code: false,
            source: HuffmanDecoder::new(strict),
        }
    }

    /// Nothing more to decode in this segment (libjpeg's `ct == -1`).
    pub(crate) fn failed(&self) -> bool {
        self.bad
    }

    pub(crate) fn saw_bad_code(&self) -> bool {
        self.saw_bad_code
    }

    /// The marker and end-of-data state of the scan.
    pub(crate) fn source(&mut self) -> &mut HuffmanDecoder {
        &mut self.source
    }

    /// Cross a restart marker (see `process_restart`) and start the next
    /// segment. The caller resets the statistics and the DC predictions.
    pub(crate) fn restart<R: Source>(
        &mut self,
        reader: &mut Buffered<R>,
        expected: u8,
        warnings: &mut Warnings,
    ) -> Result<()> {
        let consumed = process_restart(reader, &mut self.source, expected, warnings)?;
        self.c = 0;
        self.a = 0;
        self.ct = -16;
        // A marker left unread means the coming segment is empty.
        self.ended = !consumed;
        self.bad = false;
        Ok(())
    }

    #[cold]
    fn fail(&mut self) {
        self.bad = true;
        self.saw_bad_code = true;
    }

    /// The next byte of the segment, zero once it has ended.
    #[inline]
    fn byte<R: Source>(&mut self, reader: &mut Buffered<R>) -> Result<u8> {
        if self.ended {
            return Ok(0);
        }
        match reader.next_byte() {
            Ok(0xFF) => self.byte_after_ff(reader),
            Ok(byte) => Ok(byte),
            Err(error) => self.end_of_data(error),
        }
    }

    #[cold]
    fn byte_after_ff<R: Source>(&mut self, reader: &mut Buffered<R>) -> Result<u8> {
        loop {
            match reader.next_byte() {
                // Fill bytes.
                Ok(0xFF) => {}
                // Byte stuffing.
                Ok(0) => return Ok(0xFF),
                Ok(code) => {
                    self.ended = true;
                    self.source
                        .set_pending_marker(Marker::from_u8(code).unwrap_or(Marker::EOI));
                    return Ok(0);
                }
                Err(error) => return self.end_of_data(error),
            }
        }
    }

    #[cold]
    fn end_of_data(&mut self, error: Error) -> Result<u8> {
        self.source.end_of_data(error)?;
        self.ended = true;
        Ok(0)
    }

    /// One binary decision with the estimate in `state` (Figure D.16, with the
    /// renormalisation of D.2.6 done before the decision that needs it).
    #[inline]
    fn decode<R: Source>(&mut self, reader: &mut Buffered<R>, state: &mut u8) -> Result<bool> {
        if self.a < 0x8000 {
            self.renormalise(reader)?;
        }

        let mut sv = *state;
        let (qe, lps, mps) = QE[usize::from(sv & 0x7F)];
        let qe = u32::from(qe);

        // `0x8000 <= a <= 0x10000`, `qe < 0x8000` and `0 <= ct <= 7`.
        self.a -= qe;
        let split = self.a << self.ct;
        if self.c >= split {
            self.c -= split;
            // Conditional LPS exchange.
            if self.a < qe {
                *state = (sv & 0x80) ^ mps;
            } else {
                *state = (sv & 0x80) ^ lps;
                sv ^= 0x80;
            }
            self.a = qe;
        } else if self.a < 0x8000 {
            // Conditional MPS exchange.
            if self.a < qe {
                *state = (sv & 0x80) ^ lps;
                sv ^= 0x80;
            } else {
                *state = (sv & 0x80) ^ mps;
            }
        }
        Ok(sv & 0x80 != 0)
    }

    #[inline]
    fn renormalise<R: Source>(&mut self, reader: &mut Buffered<R>) -> Result<()> {
        while self.a < 0x8000 {
            self.ct -= 1;
            if self.ct < 0 {
                let byte = self.byte(reader)?;
                // `c < a << ct` once the first two bytes are in, so this stays
                // within 25 bits.
                self.c = (self.c << 8) | u32::from(byte);
                self.ct += 8;
                if self.ct < 0 {
                    self.ct += 1;
                    if self.ct == 0 {
                        // Both initial bytes are in: `a` becomes 0x10000.
                        self.a = 0x8000;
                    }
                }
            }
            self.a <<= 1;
        }
        Ok(())
    }

    /// Figure F.24: the bits of a magnitude below its leading one, `m`, and
    /// from them the value, `v + 1` with its sign.
    #[inline]
    fn magnitude<R: Source>(
        &mut self,
        reader: &mut Buffered<R>,
        bin: &mut u8,
        mut m: i32,
        negative: bool,
    ) -> Result<i32> {
        let mut v = m;
        loop {
            m >>= 1;
            if m == 0 {
                break;
            }
            if self.decode(reader, bin)? {
                v |= m;
            }
        }
        v += 1;
        Ok(if negative { -v } else { v })
    }

    /// A value's magnitude category beyond its first steps (Figure F.23): `m`
    /// doubles for every 1 decoded, each in the next bin from `at`. `false`
    /// when it overflows; `m` stays below `0x8000`, so `at` moves at most 14
    /// bins.
    #[inline]
    fn category<R: Source>(
        &mut self,
        reader: &mut Buffered<R>,
        bins: &mut [u8],
        at: &mut usize,
        m: &mut i32,
    ) -> Result<bool> {
        while self.decode(reader, &mut bins[*at])? {
            *m <<= 1;
            if *m == 0x8000 {
                self.fail();
                return Ok(false);
            }
            *at += 1;
        }
        Ok(true)
    }
}

/// The statistics of one scan's tables.
pub(crate) struct Stats {
    dc: [[u8; DC_BINS]; TABLES],
    ac: [[u8; AC_BINS]; TABLES],
}

impl Stats {
    pub(crate) fn new() -> Stats {
        Stats {
            dc: [[0; DC_BINS]; TABLES],
            ac: [[0; AC_BINS]; TABLES],
        }
    }

    /// Forget what a DC and an AC table have learned.
    pub(crate) fn reset(&mut self, dc_table: usize, ac_table: usize) {
        self.dc[dc_table] = [0; DC_BINS];
        self.ac[ac_table] = [0; AC_BINS];
    }
}

/// One component's DC prediction and conditioning state.
#[derive(Clone, Copy, Default)]
pub(crate) struct DcState {
    prediction: i32,
    /// First bin of the context the previous difference selects (Table F.4).
    context: usize,
}

/// Sections F.2.4.1 and F.1.4.4.1: the DC value of a block. `None` when the
/// magnitude of the difference overflowed.
pub(crate) fn decode_dc<R: Source>(
    d: &mut ArithDecoder,
    reader: &mut Buffered<R>,
    stats: &mut Stats,
    table: usize,
    (low, high): (u8, u8),
    dc: &mut DcState,
) -> Result<Option<i32>> {
    let bins = &mut stats.dc[table];
    let mut at = dc.context;
    if !d.decode(reader, &mut bins[at])? {
        dc.context = 0;
        return Ok(Some(dc.prediction));
    }
    let negative = d.decode(reader, &mut bins[at + 1])?;
    let sign = usize::from(negative);
    at += 2 + sign;
    let mut m = i32::from(d.decode(reader, &mut bins[at])?);
    if m != 0 {
        at = DC_X1;
        if !d.category(reader, bins, &mut at, &mut m)? {
            return Ok(None);
        }
    }
    // F.1.4.4.1.2. `low` and `high` are nibbles.
    dc.context = if m < (1i32 << low) >> 1 {
        0
    } else if m > (1i32 << high) >> 1 {
        12 + sign * 4
    } else {
        4 + sign * 4
    };
    let diff = d.magnitude(reader, &mut bins[at + M_OFFSET], m, negative)?;
    dc.prediction = (dc.prediction + diff) & 0xFFFF;
    Ok(Some(dc.prediction))
}

/// Section F.2.4.2: the AC coefficients of a block in `band` (zigzag
/// positions, `1 <= start`, `end <= 64`), scaled by `2^al`; also the first
/// pass of a progressive band. `nonzero` is kept as `Decoder::nonzero` says.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_ac<R: Source>(
    d: &mut ArithDecoder,
    reader: &mut Buffered<R>,
    stats: &mut Stats,
    table: usize,
    kx: u8,
    band: Range<u8>,
    al: u8,
    coefficients: &mut [i16; 64],
    nonzero: &mut u64,
) -> Result<()> {
    let bins = &mut stats.ac[table];
    let end = usize::from(band.end.min(64));
    let mut fixed = FIXED;
    let mut k = usize::from(band.start.max(1));
    while k < end {
        let mut at = 3 * (k - 1);
        if d.decode(reader, &mut bins[at])? {
            // End of block.
            break;
        }
        while !d.decode(reader, &mut bins[at + 1])? {
            at += 3;
            k += 1;
            if k >= end {
                d.fail();
                return Ok(());
            }
        }
        let negative = d.decode(reader, &mut fixed)?;
        at += 2;
        let mut m = i32::from(d.decode(reader, &mut bins[at])?);
        if m != 0 && d.decode(reader, &mut bins[at])? {
            m <<= 1;
            at = if k <= usize::from(kx) {
                AC_X2_LOW
            } else {
                AC_X2_HIGH
            };
            if !d.category(reader, bins, &mut at, &mut m)? {
                return Ok(());
            }
        }
        let v = d.magnitude(reader, &mut bins[at + M_OFFSET], m, negative)?;
        // Truncated to 16 bits, as libjpeg's `JCOEF` is.
        let value = (v << al) as i16;
        coefficients[usize::from(UNZIGZAG[k & 63]) & 63] = value;
        *nonzero = (*nonzero & !(1 << k)) | (u64::from(value != 0) << k);
        k += 1;
    }
    Ok(())
}

/// Section G.1.3.1: one more bit of a block's DC value.
pub(crate) fn refine_dc<R: Source>(
    d: &mut ArithDecoder,
    reader: &mut Buffered<R>,
    al: u8,
    coefficients: &mut [i16; 64],
) -> Result<()> {
    let mut fixed = FIXED;
    if d.decode(reader, &mut fixed)? {
        coefficients[0] |= 1 << al;
    }
    Ok(())
}

/// Section G.1.3.3: one more bit of a block's AC coefficients in `band`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_ac<R: Source>(
    d: &mut ArithDecoder,
    reader: &mut Buffered<R>,
    stats: &mut Stats,
    table: usize,
    band: Range<u8>,
    al: u8,
    coefficients: &mut [i16; 64],
    nonzero: &mut u64,
) -> Result<()> {
    let bins = &mut stats.ac[table];
    let end = usize::from(band.end.min(64));
    let bit = 1i16 << al;
    let mut fixed = FIXED;
    // `EOBx`, the end of the block as the previous scans left it: no
    // end-of-block decision is coded up to there. (Bit 0 is never set in
    // `nonzero`; setting it here makes the answer 0 for an empty block.)
    let below_end = !0u64 >> (64 - end.max(1));
    let previous_end = 63 - ((*nonzero & below_end) | 1).leading_zeros() as usize;

    let mut k = usize::from(band.start.max(1));
    while k < end {
        let mut at = 3 * (k - 1);
        if k > previous_end && d.decode(reader, &mut bins[at])? {
            break;
        }
        loop {
            let coefficient = &mut coefficients[usize::from(UNZIGZAG[k & 63]) & 63];
            if *coefficient != 0 {
                if d.decode(reader, &mut bins[at + 2])? {
                    // Wrapping as libjpeg's 16-bit arithmetic does; never to
                    // zero, `bit` being at most `1 << 13`.
                    *coefficient = if *coefficient < 0 {
                        coefficient.wrapping_sub(bit)
                    } else {
                        coefficient.wrapping_add(bit)
                    };
                }
                break;
            }
            if d.decode(reader, &mut bins[at + 1])? {
                *coefficient = if d.decode(reader, &mut fixed)? {
                    -bit
                } else {
                    bit
                };
                *nonzero |= 1 << k;
                break;
            }
            at += 3;
            k += 1;
            if k >= end {
                d.fail();
                return Ok(());
            }
        }
        k += 1;
    }
    Ok(())
}

/// Table D.3: `Qe`, the next state after an LPS (with bit 7 set where the
/// sense of the MPS switches) and the next state after an MPS. The last entry
/// is the fixed estimate of 0.5 (T.851, section 10.3).
#[rustfmt::skip]
static QE: [(u16, u8, u8); 128] = {
    const STATES: [(u16, u8, u8); 114] = [
        (0x5a1d, 0x81, 1), (0x2586, 0x0e, 2), (0x1114, 0x10, 3), (0x080b, 0x12, 4),
        (0x03d8, 0x14, 5), (0x01da, 0x17, 6), (0x00e5, 0x19, 7), (0x006f, 0x1c, 8),
        (0x0036, 0x1e, 9), (0x001a, 0x21, 10), (0x000d, 0x23, 11), (0x0006, 0x09, 12),
        (0x0003, 0x0a, 13), (0x0001, 0x0c, 13), (0x5a7f, 0x8f, 15), (0x3f25, 0x24, 16),
        (0x2cf2, 0x26, 17), (0x207c, 0x27, 18), (0x17b9, 0x28, 19), (0x1182, 0x2a, 20),
        (0x0cef, 0x2b, 21), (0x09a1, 0x2d, 22), (0x072f, 0x2e, 23), (0x055c, 0x30, 24),
        (0x0406, 0x31, 25), (0x0303, 0x33, 26), (0x0240, 0x34, 27), (0x01b1, 0x36, 28),
        (0x0144, 0x38, 29), (0x00f5, 0x39, 30), (0x00b7, 0x3b, 31), (0x008a, 0x3c, 32),
        (0x0068, 0x3e, 33), (0x004e, 0x3f, 34), (0x003b, 0x20, 35), (0x002c, 0x21, 9),
        (0x5ae1, 0xa5, 37), (0x484c, 0x40, 38), (0x3a0d, 0x41, 39), (0x2ef1, 0x43, 40),
        (0x261f, 0x44, 41), (0x1f33, 0x45, 42), (0x19a8, 0x46, 43), (0x1518, 0x48, 44),
        (0x1177, 0x49, 45), (0x0e74, 0x4a, 46), (0x0bfb, 0x4b, 47), (0x09f8, 0x4d, 48),
        (0x0861, 0x4e, 49), (0x0706, 0x4f, 50), (0x05cd, 0x30, 51), (0x04de, 0x32, 52),
        (0x040f, 0x32, 53), (0x0363, 0x33, 54), (0x02d4, 0x34, 55), (0x025c, 0x35, 56),
        (0x01f8, 0x36, 57), (0x01a4, 0x37, 58), (0x0160, 0x38, 59), (0x0125, 0x39, 60),
        (0x00f6, 0x3a, 61), (0x00cb, 0x3b, 62), (0x00ab, 0x3d, 63), (0x008f, 0x3d, 32),
        (0x5b12, 0xc1, 65), (0x4d04, 0x50, 66), (0x412c, 0x51, 67), (0x37d8, 0x52, 68),
        (0x2fe8, 0x53, 69), (0x293c, 0x54, 70), (0x2379, 0x56, 71), (0x1edf, 0x57, 72),
        (0x1aa9, 0x57, 73), (0x174e, 0x48, 74), (0x1424, 0x48, 75), (0x119c, 0x4a, 76),
        (0x0f6b, 0x4a, 77), (0x0d51, 0x4b, 78), (0x0bb6, 0x4d, 79), (0x0a40, 0x4d, 48),
        (0x5832, 0xd0, 81), (0x4d1c, 0x58, 82), (0x438e, 0x59, 83), (0x3bdd, 0x5a, 84),
        (0x34ee, 0x5b, 85), (0x2eae, 0x5c, 86), (0x299a, 0x5d, 87), (0x2516, 0x56, 71),
        (0x5570, 0xd8, 89), (0x4ca9, 0x5f, 90), (0x44d9, 0x60, 91), (0x3e22, 0x61, 92),
        (0x3824, 0x63, 93), (0x32b4, 0x63, 94), (0x2e17, 0x5d, 86), (0x56a8, 0xdf, 96),
        (0x4f46, 0x65, 97), (0x47e5, 0x66, 98), (0x41cf, 0x67, 99), (0x3c3d, 0x68, 100),
        (0x375e, 0x63, 93), (0x5231, 0x69, 102), (0x4c0f, 0x6a, 103), (0x4639, 0x6b, 104),
        (0x415e, 0x67, 99), (0x5627, 0xe9, 106), (0x50e7, 0x6c, 107), (0x4b85, 0x6d, 103),
        (0x5597, 0x6e, 109), (0x504f, 0x6f, 107), (0x5a10, 0xee, 111), (0x5522, 0x70, 109),
        (0x59eb, 0xf0, 111), (0x5a1d, 0x71, 113),
    ];
    // Padded to 128 so that a 7-bit index needs no bounds check; no state
    // leads to the padding.
    let mut table = [STATES[113]; 128];
    let mut i = 0;
    while i < STATES.len() {
        table[i] = STATES[i];
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The bounds the decoder's unchecked-looking arithmetic relies on.
    #[test]
    fn states_are_closed_and_estimates_small() {
        for &(qe, lps, mps) in &QE {
            assert!(qe > 0 && qe < 0x8000);
            assert!(usize::from(lps & 0x7F) < 114 && mps < 114);
        }
        assert_eq!(QE[usize::from(FIXED)], (0x5a1d, FIXED, FIXED));
    }

    /// The worked example of T.81 K.4: decoding the compressed test sequence
    /// gives back the 256 input bits.
    #[test]
    fn annex_k4_test_sequence() {
        const CODED: [u8; 30] = [
            0x65, 0x5B, 0x51, 0x44, 0xF7, 0x96, 0x9D, 0x51, 0x78, 0x55, 0xBF, 0xFF, 0x00, 0xFC,
            0x51, 0x84, 0xC7, 0xCE, 0xF9, 0x39, 0x00, 0x28, 0x7D, 0x46, 0x70, 0x8E, 0xCB, 0xC0,
            0xF6, 0xFF,
        ];
        const BITS: [u8; 32] = [
            0x00, 0x02, 0x00, 0x51, 0x00, 0x00, 0x00, 0xC0, 0x03, 0x52, 0x87, 0x2A, 0xAA, 0xAA,
            0xAA, 0xAA, 0x82, 0xC0, 0x20, 0x00, 0xFC, 0xD7, 0x9E, 0xF6, 0x74, 0xEA, 0xAB, 0xF7,
            0x69, 0x7E, 0xE7, 0x4C,
        ];
        let mut data = CODED.to_vec();
        data.push(0xD9);
        let mut reader = Buffered::new(&data[..]);
        let mut d = ArithDecoder::new(false);
        let mut state = 0u8;
        for (i, &byte) in BITS.iter().enumerate() {
            let mut got = 0u8;
            for _ in 0..8 {
                got = got << 1 | u8::from(d.decode(&mut reader, &mut state).unwrap());
            }
            assert_eq!(got, byte, "byte {i}");
        }
    }
}
