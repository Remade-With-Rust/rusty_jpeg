use crate::decode::error::{Error, Result};
use crate::decode::marker::Marker;
use crate::decode::parser::ScanInfo;
use crate::decode::Source;
use alloc::borrow::ToOwned;
use alloc::vec;
use alloc::vec::Vec;

/// Width of the first-level lookup (`LUT_BITS`), for callers holding the
/// accumulator themselves (see `HuffmanDecoder::state`).
pub const FAST_BITS: u8 = LUT_BITS;

/// Width of the first-level lookup. 9 bits, not the 8 jpeg-decoder used:
/// measured against 8, both 9 and 10 took the dense photos 3-6% faster
/// (more AC codes resolve in one lookup together with their value) and were
/// neutral on sparse 1080p video frames; 9 keeps the two tables at 1.5 KB.
const LUT_BITS: u8 = 9;

/// The run `fast_ac` reports for an end-of-block code. A real entry's
/// run is at most 15, so this cannot collide with one.
pub const AC_EOB: u8 = 0xFF;
/// The `run_size` marking an EOB entry in `ac_lut`. Real entries pack
/// `run << 4 | (code size + magnitude)` with the size at most 8, so their low
/// nibble never reaches 0xF.
const AC_LUT_EOB: u8 = 0xFF;

#[derive(Debug)]
pub struct HuffmanDecoder {
    bits: u64,
    num_bits: u8,
    marker: Option<Marker>,
    /// Zero bits appended after `marker` was reached, since the last `reset`.
    /// They sit at the tail of `bits`, behind every real bit, so a decode has
    /// consumed one exactly when `num_bits < fill_bits` (see `ran_out`).
    /// Touched only on the byte-at-a-time refill path, never per symbol.
    fill_bits: u32,
    /// Refuse what the tolerant decoder recovers from: the end of the data
    /// inside a scan and a bad Huffman code are errors, as they were before
    /// the recovery existed (`Decoder::set_strict`).
    strict: bool,
    /// The data source ended inside this scan. `marker` is then a synthetic
    /// EOI, as libjpeg-turbo's data source inserts one (`fill_input_buffer`,
    /// jdatasrc.c). Never cleared: the data does not come back.
    eof: bool,
    /// A bit sequence that is no code of its table was read as symbol 0.
    bad_code: bool,
}

impl HuffmanDecoder {
    pub fn new(strict: bool) -> HuffmanDecoder {
        HuffmanDecoder {
            bits: 0,
            num_bits: 0,
            marker: None,
            fill_bits: 0,
            strict,
            eof: false,
            bad_code: false,
        }
    }

    // Section F.2.2.3
    // Figure F.16
    pub fn decode<R: Source>(
        &mut self,
        reader: &mut crate::decode::Buffered<R>,
        table: &HuffmanTable,
    ) -> Result<u8> {
        crate::prof::bump(crate::prof::Count::DecSymbols, 1);
        if self.num_bits < 16 {
            self.read_bits(reader)?;
        }

        let (value, size) = table.lut[self.peek_bits(LUT_BITS) as usize & ((1 << LUT_BITS) - 1)];

        if size > 0 {
            crate::prof::bump(crate::prof::Count::DecLutHit, 1);
            self.consume_bits(size);
            Ok(value)
        } else {
            crate::prof::bump(crate::prof::Count::DecLutMiss, 1);
            let bits = self.peek_bits(16);

            for i in LUT_BITS..16 {
                let code = (bits >> (15 - i)) as i32;

                if code <= table.maxcode[i as usize] {
                    self.consume_bits(i + 1);

                    let index = (code + table.delta[i as usize]) as usize;
                    return Ok(table.values[index]);
                }
            }

            // libjpeg-turbo reads a 17th bit before it gives up.
            if self.num_bits < 17 {
                self.read_bits(reader)?;
            }
            self.bad_huffman_code()
        }
    }

    /// [`decode`](Self::decode), with its common case inlined at the call
    /// site: the accumulator already holds the 16 bits `decode` refills to,
    /// and the code fits the first-level table. Anything else goes to
    /// `decode`, which then refills and resolves exactly as it always has --
    /// with 16 bits present it would not have refilled, so both paths consume
    /// the same bits and return the same symbol.
    ///
    /// For the refinement scans, whose every symbol went through the
    /// out-of-line `decode` (1.23M calls at 36.6 Ir per 2000x1250 4:2:0
    /// progressive decode pair).
    #[inline(always)]
    pub fn decode_inline<R: Source>(
        &mut self,
        reader: &mut crate::decode::Buffered<R>,
        table: &HuffmanTable,
    ) -> Result<u8> {
        if self.num_bits >= 16 {
            let (value, size) =
                table.lut[self.peek_bits(LUT_BITS) as usize & ((1 << LUT_BITS) - 1)];
            if size > 0 {
                crate::prof::bump(crate::prof::Count::DecSymbols, 1);
                crate::prof::bump(crate::prof::Count::DecLutHit, 1);
                self.consume_bits(size);
                return Ok(value);
            }
        }
        self.decode(reader, table)
    }

    /// No code of the table matches the next 16 bits: libjpeg-turbo's
    /// `JWRN_HUFF_BAD_CODE` (`jpeg_huff_decode`, jdhuff.c, and the
    /// `HUFF_DECODE_FAST` macro). It has by then consumed 17 bits -- its
    /// length search runs into the sentinel at 17 -- and it returns symbol 0
    /// "as the safest result", then carries on. What follows is garbage until
    /// the next restart marker (or the end of the scan), but it is the same
    /// garbage libjpeg-turbo shows, and the image is not refused for it.
    #[cold]
    fn bad_huffman_code(&mut self) -> Result<u8> {
        if self.strict {
            return Err(Error::Format("failed to decode huffman code".to_owned()));
        }
        self.bad_code = true;
        // `decode` made sure the accumulator holds them.
        self.consume_bits(17);
        Ok(0)
    }

    /// The accumulator `(bits, num_bits)`, for a caller that keeps it in
    /// registers across a loop of table lookups (`decode_block`'s AC loop).
    /// Through `&mut self` LLVM kept it in memory: a load, a store and two
    /// read-modify-writes of `num_bits` per symbol, because the refill call
    /// in the same loop might touch it. The caller hands it back with
    /// [`set_state`](Self::set_state) before any other method runs.
    #[inline(always)]
    pub fn state(&self) -> (u64, u8) {
        (self.bits, self.num_bits)
    }

    /// See [`state`](Self::state).
    #[inline(always)]
    pub fn set_state(&mut self, bits: u64, num_bits: u8) {
        self.bits = bits;
        self.num_bits = num_bits;
    }

    /// Refill (`read_bits`) for a caller holding the state: it must have
    /// handed it back first and must take it again after.
    #[inline(always)]
    pub fn refill<R: Source>(&mut self, reader: &mut crate::decode::Buffered<R>) -> Result<()> {
        self.read_bits(reader)
    }

    /// One AC symbol and its value in one lookup, on a caller-held
    /// accumulator holding at least `LUT_BITS` bits (see `ac_lut`):
    /// `Some((value, run))`, with run `AC_EOB` for an end-of-block, or `None`
    /// when the code and its magnitude bits do not fit the table -- the
    /// caller then hands the state back and takes `decode` + `receive_extend`,
    /// which read exactly the same bits. (This was `decode_fast_ac`, which
    /// did the same through `&mut self`.)
    #[inline(always)]
    pub fn fast_ac(table: &HuffmanTable, bits: &mut u64, num_bits: &mut u8) -> Option<(i16, u8)> {
        debug_assert!(*num_bits >= LUT_BITS);
        let (value, run_size) = table.ac_lut[(*bits >> (64 - LUT_BITS)) as usize];
        if run_size == AC_LUT_EOB {
            crate::prof::bump(crate::prof::Count::DecFastAcHit, 1);
            *bits <<= value as u8;
            *num_bits -= value as u8;
            return Some((0, AC_EOB));
        }
        if run_size != 0 {
            crate::prof::bump(crate::prof::Count::DecFastAcHit, 1);
            let size = run_size & 0x0f;
            *bits <<= size;
            *num_bits -= size;
            return Some((value, run_size >> 4));
        }
        crate::prof::bump(crate::prof::Count::DecFastAcMiss, 1);
        None
    }

    /// A DC difference in one lookup: the symbol and its magnitude bits
    /// together, already extended (see `HuffmanTable::ac_lut`). `None` when
    /// they do not fit the table; the caller then takes `decode` +
    /// `receive_extend`, which read exactly the same bits.
    #[inline]
    pub fn decode_fast_dc<R: Source>(
        &mut self,
        reader: &mut crate::decode::Buffered<R>,
        table: &HuffmanTable,
    ) -> Result<Option<i16>> {
        if self.num_bits < LUT_BITS {
            self.read_bits(reader)?;
        }
        let (diff, size) = table.ac_lut[self.peek_bits(LUT_BITS) as usize & ((1 << LUT_BITS) - 1)];
        if size == 0 {
            return Ok(None);
        }
        crate::prof::bump(crate::prof::Count::DecSymbols, 1);
        self.consume_bits(size);
        Ok(Some(diff))
    }

    #[inline]
    pub fn get_bits<R: Source>(
        &mut self,
        reader: &mut crate::decode::Buffered<R>,
        count: u8,
    ) -> Result<u16> {
        if self.num_bits < count {
            self.read_bits(reader)?;
        }

        let bits = self.peek_bits(count);
        self.consume_bits(count);

        Ok(bits)
    }

    #[inline]
    pub fn receive_extend<R: Source>(
        &mut self,
        reader: &mut crate::decode::Buffered<R>,
        count: u8,
    ) -> Result<i16> {
        crate::prof::bump(crate::prof::Count::DecReceiveExtend, 1);
        let value = self.get_bits(reader, count)?;
        Ok(extend(value, count))
    }

    pub fn reset(&mut self) {
        self.bits = 0;
        self.num_bits = 0;
        self.fill_bits = 0;
    }

    /// A code that is in the table but cannot occur where it was read (a
    /// refinement scan's new coefficient with a size other than 1).
    /// libjpeg-turbo warns "bad Huffman code" and goes on.
    #[cold]
    pub fn unexpected_code(&mut self) -> Result<()> {
        if self.strict {
            return Err(Error::Format("unexpected huffman code".to_owned()));
        }
        self.bad_code = true;
        Ok(())
    }

    /// Whether the data source ended inside this scan (see `eof`).
    pub fn hit_end_of_data(&self) -> bool {
        self.eof
    }

    /// Whether a bad Huffman code was read as symbol 0 in this scan.
    pub fn saw_bad_code(&self) -> bool {
        self.bad_code
    }

    /// Whole bytes of real data (not the zero fill) still unread in the
    /// accumulator. A valid segment leaves none when its last MCU is done --
    /// only padding bits, fewer than eight -- so any that remain are data the
    /// scan did not account for, which libjpeg-turbo counts into its
    /// "extraneous bytes before marker" warning (`discarded_bytes`).
    pub fn unread_bytes(&self) -> u32 {
        u32::from(self.num_bits).saturating_sub(self.fill_bits) / 8
    }

    /// The data source has ended: from here on the scan sees an EOI, exactly
    /// as if `FF D9` followed the last byte. libjpeg-turbo's stdio and memory
    /// sources do this literally ("insert a fake EOI marker", jdatasrc.c).
    /// Any other error, and every error in strict mode, is passed on.
    #[cold]
    pub fn end_of_data(&mut self, error: Error) -> Result<()> {
        match error {
            Error::UnexpectedEof if !self.strict => {
                self.marker = Some(Marker::EOI);
                self.eof = true;
                Ok(())
            }
            error => Err(error),
        }
    }

    /// libjpeg's `insufficient_data`: a symbol has been decoded from the zero
    /// bits inserted after the entropy-coded segment ran into a marker, so the
    /// segment's real data is exhausted. Sticky until the next `reset`.
    #[inline]
    pub fn ran_out(&self) -> bool {
        self.fill_bits > u32::from(self.num_bits)
    }

    /// Take the marker the bit reader has already run into, WITHOUT reading
    /// any further. (`take_marker` refills first, which pulls bytes.)
    pub fn take_pending_marker(&mut self) -> Option<Marker> {
        self.marker.take()
    }

    /// Leave `marker` unread: the reader treats it as sitting at the current
    /// input position, so the next refill yields zero bits and the next
    /// `take_marker` returns it again (libjpeg's non-zero `unread_marker`).
    pub fn set_pending_marker(&mut self, marker: Marker) {
        self.marker = Some(marker);
    }

    pub fn take_marker<R: Source>(
        &mut self,
        reader: &mut crate::decode::Buffered<R>,
    ) -> Result<Option<Marker>> {
        self.read_bits(reader).map(|_| self.marker.take())
    }

    #[inline]
    fn peek_bits(&mut self, count: u8) -> u16 {
        debug_assert!(count <= 16);
        debug_assert!(self.num_bits >= count);

        ((self.bits >> (64 - count)) & ((1 << count) - 1)) as u16
    }

    #[inline]
    fn consume_bits(&mut self, count: u8) {
        debug_assert!(self.num_bits >= count);

        self.bits <<= count as usize;
        self.num_bits -= count;
    }

    fn read_bits<R: Source>(&mut self, reader: &mut crate::decode::Buffered<R>) -> Result<()> {
        crate::prof::bump(crate::prof::Count::DecRefills, 1);

        // Bulk refill: absorb every byte the accumulator has room for in ONE
        // step, when the buffer holds them and none is 0xFF.
        //
        // The byte loop below places byte i at bit `56 - num_bits - 8i`, i.e. it
        // packs them big-endian from the top down — which is exactly
        // `u64::from_be_bytes(window) >> num_bits`. Counters said 6.8 bytes per
        // refill, so this replaces ~7 iterations (each with a branch, a shift
        // and an 0xFF test) with one load, one scan and one shift.
        //
        // The 0xFF test cannot be skipped: a 0xFF inside the entropy stream is
        // either byte-stuffing or a marker, and both need the slow path. Bailing
        // to it costs nothing because nothing has been consumed yet.
        if self.marker.is_none() {
            let want = ((64 - self.num_bits) / 8) as usize;
            let avail = reader.buffered();
            // With a whole 8-byte window buffered (all but the last few bytes
            // of each buffer fill), take it in ONE constant-length load and
            // test the bytes for 0xFF in-register. The general form below
            // copied `want` bytes -- a runtime length, so a `memcpy` call --
            // and scanned them with `contains`, a `memchr` call: two calls per
            // refill, ~27k refills per 2000x1250 4:4:4 frame.
            if want > 0 && avail.len() >= 8 {
                let window = u64::from_be_bytes([
                    avail[0], avail[1], avail[2], avail[3], avail[4], avail[5], avail[6], avail[7],
                ]);
                // The top `want` bytes: those the accumulator has room for.
                let keep = !(!0u64).checked_shr(want as u32 * 8).unwrap_or(0);
                if ff_bytes(window) & keep == 0 {
                    self.bits |= (window & keep) >> self.num_bits;
                    self.num_bits += (want * 8) as u8;
                    reader.consume(want);
                    crate::prof::bump(crate::prof::Count::DecBulkRefills, 1);
                    return Ok(());
                }
            } else if want > 0 && avail.len() >= want && !avail[..want].contains(&0xFF) {
                let mut window = [0u8; 8];
                window[..want].copy_from_slice(&avail[..want]);
                self.bits |= u64::from_be_bytes(window) >> self.num_bits;
                self.num_bits += (want * 8) as u8;
                reader.consume(want);
                crate::prof::bump(crate::prof::Count::DecBulkRefills, 1);
                return Ok(());
            }
        }

        while self.num_bits <= 56 {
            crate::prof::bump(crate::prof::Count::DecBytesRead, 1);
            // Fill with zero bits if we have reached the end.
            let byte = match self.marker {
                Some(_) => {
                    self.fill_bits = self.fill_bits.saturating_add(8);
                    0
                }
                None => match reader.next_byte() {
                    Ok(byte) => byte,
                    Err(error) => {
                        self.end_of_data(error)?;
                        continue;
                    }
                },
            };

            if byte == 0xFF {
                // The data ending after an FF (a marker or a stuffed FF cut in
                // two) leaves that FF as a fill byte before the synthetic EOI,
                // which is how libjpeg-turbo reads `FF` + its fake `FF D9`.
                let mut next_byte = match reader.next_byte() {
                    Ok(byte) => byte,
                    Err(error) => {
                        self.end_of_data(error)?;
                        continue;
                    }
                };

                // Check for byte stuffing.
                if next_byte != 0x00 {
                    // We seem to have reached the end of entropy-coded data and encountered a
                    // marker. Since we can't put data back into the reader, we have to continue
                    // reading to identify the marker so we can pass it on.

                    // Section B.1.1.2
                    // "Any marker may optionally be preceded by any number of fill bytes, which are bytes assigned code X’FF’."
                    while next_byte == 0xFF {
                        next_byte = match reader.next_byte() {
                            Ok(byte) => byte,
                            Err(error) => {
                                self.end_of_data(error)?;
                                break;
                            }
                        };
                    }
                    if self.marker.is_some() {
                        // The data ended in the run of fill bytes.
                        continue;
                    }

                    if next_byte != 0x00 {
                        self.marker = Some(Marker::from_u8(next_byte).unwrap());
                        continue;
                    }
                    // `FF FF .. FF 00` is not valid, but libjpeg-turbo reads it
                    // as a single FF data byte (`jpeg_fill_bit_buffer`,
                    // jdhuff.c) rather than failing the image, and so do we.
                }
            }

            self.bits |= (byte as u64) << (56 - self.num_bits);
            self.num_bits += 8;
        }

        Ok(())
    }
}

/// The high bit of every byte of `x` that is 0xFF, and no other bit.
///
/// Exact, not the usual "has a zero byte" approximation, whose borrow can flag
/// a byte above a real hit: `(b & 0x7F) + 0x7F` carries into bit 7 for every
/// low part except 0x7F, so bit 7 ends up clear only for a byte of 0xFF.
#[inline]
fn ff_bytes(x: u64) -> u64 {
    const LOW7: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    let y = !x;
    !(((y & LOW7) + LOW7) | y | LOW7)
}

// Section F.2.2.1
// Figure F.12
fn extend(value: u16, count: u8) -> i16 {
    let vt = 1 << (count as u16 - 1);

    if value < vt {
        value as i16 + (-1 << count as i16) + 1
    } else {
        value as i16
    }
}

#[cfg(test)]
mod ff_bytes_tests {
    /// Every byte value in every lane, alone and next to every other value:
    /// exactly the 0xFF bytes are flagged.
    #[test]
    fn ff_bytes_flags_exactly_the_ff_bytes() {
        for lane in 0..8 {
            for a in 0..=255u64 {
                for b in [0u64, 1, 0x7F, 0x80, 0xFE, 0xFF] {
                    let mut x = 0x0102_0304_0506_0708u64 ^ (b * 0x0101_0101_0101_0101);
                    x &= !(0xFF << (8 * lane));
                    x |= a << (8 * lane);
                    let want = (0..8).fold(0u64, |m, i| {
                        if (x >> (8 * i)) & 0xFF == 0xFF {
                            m | (0x80 << (8 * i))
                        } else {
                            m
                        }
                    });
                    assert_eq!(super::ff_bytes(x), want, "x = {x:#018x}");
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HuffmanTableClass {
    DC,
    AC,
}

pub struct HuffmanTable {
    values: Vec<u8>,
    delta: [i32; 16],
    maxcode: [i32; 16],

    lut: [(u8, u8); 1 << LUT_BITS],
    /// Combined run/value decoding of the SAME bit window as `lut`. Kept as a
    /// separate small array on purpose: folding both into one 8-byte-padded
    /// entry removed a redundant peek but quadrupled the primary table to 2 KB,
    /// and measured 1703 -> 1969 ms. The redundancy was worth less than the
    /// cache footprint.
    ///
    /// A DC table fills it the same way with run 0: the DC difference already
    /// extended, so `decode_fast_dc` replaces a symbol decode plus a separate
    /// `receive_extend` (refill check, read, extend) with one lookup whenever
    /// the code and its magnitude bits fit.
    ac_lut: [(i16, u8); 1 << LUT_BITS],
}

impl HuffmanTable {
    pub fn new(bits: &[u8; 16], values: &[u8], class: HuffmanTableClass) -> Result<HuffmanTable> {
        let (huffcode, huffsize) = derive_huffman_codes(bits)?;

        // Section F.2.2.3
        // Figure F.15
        // delta[i] is set to VALPTR(I) - MINCODE(I)
        let mut delta = [0i32; 16];
        let mut maxcode = [-1i32; 16];
        let mut j = 0;

        for i in 0..16 {
            if bits[i] != 0 {
                delta[i] = j as i32 - huffcode[j] as i32;
                j += bits[i] as usize;
                maxcode[i] = huffcode[j - 1] as i32;
            }
        }

        // Build a lookup table for faster decoding.
        let mut lut = [(0u8, 0u8); 1 << LUT_BITS];

        for (i, &size) in huffsize
            .iter()
            .enumerate()
            .filter(|&(_, &size)| size <= LUT_BITS)
        {
            let bits_remaining = LUT_BITS - size;
            let start = (huffcode[i] << bits_remaining) as usize;

            let val = (values[i], size);
            for b in &mut lut[start..][..1 << bits_remaining] {
                *b = val;
            }
        }

        // Build a lookup table for small AC coefficients which both decodes the value and does the
        // equivalent of receive_extend.
        let mut ac_lut = [(0i16, 0u8); 1 << LUT_BITS];
        if matches!(class, HuffmanTableClass::DC) {
            for (i, &(value, size)) in lut.iter().enumerate() {
                // `size > 0`: a real code (an empty slot also reads value 0).
                // Categories above 11 are invalid and stay on the slow path,
                // which reports them.
                if size > 0 && value <= 11 && size + value <= LUT_BITS {
                    let diff = if value == 0 {
                        0
                    } else {
                        let bits =
                            (((i << size) & ((1 << LUT_BITS) - 1)) >> (LUT_BITS - value)) as u16;
                        extend(bits, value)
                    };
                    ac_lut[i] = (diff, size + value);
                }
            }
        }
        if matches!(class, HuffmanTableClass::AC) {
            for (i, &(value, size)) in lut.iter().enumerate() {
                let run_length = value >> 4;
                let magnitude_category = value & 0x0f;

                // EOB (0x00): store the code size to consume. `size > 0`
                // distinguishes a real code from an empty slot, which also
                // reads value 0.
                if value == 0x00 && size > 0 {
                    ac_lut[i] = (size as i16, AC_LUT_EOB);
                    continue;
                }

                if magnitude_category > 0 && size + magnitude_category <= LUT_BITS {
                    let unextended_ac_value = (((i << size) & ((1 << LUT_BITS) - 1))
                        >> (LUT_BITS - magnitude_category))
                        as u16;
                    ac_lut[i] = (
                        extend(unextended_ac_value, magnitude_category),
                        (run_length << 4) | (size + magnitude_category),
                    );
                }
            }
        }

        Ok(HuffmanTable {
            values: values.to_vec(),
            delta,
            maxcode,
            lut,
            ac_lut,
        })
    }
}

// Section C.2
fn derive_huffman_codes(bits: &[u8; 16]) -> Result<(Vec<u16>, Vec<u8>)> {
    // Figure C.1
    let huffsize = bits
        .iter()
        .enumerate()
        .fold(Vec::new(), |mut acc, (i, &value)| {
            acc.extend(core::iter::repeat_n((i + 1) as u8, value as usize));
            acc
        });

    // Figure C.2
    let mut huffcode = vec![0u16; huffsize.len()];
    let mut code_size = huffsize[0];
    let mut code = 0u32;

    for (i, &size) in huffsize.iter().enumerate() {
        while code_size < size {
            code <<= 1;
            code_size += 1;
        }

        if code >= (1u32 << size) {
            return Err(Error::Format("bad huffman code length".to_owned()));
        }

        huffcode[i] = code as u16;
        code += 1;
    }

    Ok((huffcode, huffsize))
}

// https://www.loc.gov/preservation/digital/formats/fdd/fdd000063.shtml
// "Avery Lee, writing in the rec.video.desktop newsgroup in 2001, commented that "MJPEG, or at
//  least the MJPEG in AVIs having the MJPG fourcc, is restricted JPEG with a fixed -- and
//  *omitted* -- Huffman table. The JPEG must be YCbCr colorspace, it must be 4:2:2, and it must
//  use basic Huffman encoding, not arithmetic or progressive.... You can indeed extract the
//  MJPEG frames and decode them with a regular JPEG decoder, but you have to prepend the DHT
//  segment to them, or else the decoder won't have any idea how to decompress the data.
//  The exact table necessary is given in the OpenDML spec.""
pub fn fill_default_mjpeg_tables(
    scan: &ScanInfo,
    dc_huffman_tables: &mut [Option<HuffmanTable>],
    ac_huffman_tables: &mut [Option<HuffmanTable>],
) {
    // Section K.3.3

    if dc_huffman_tables[0].is_none() && scan.dc_table_indices.contains(&0) {
        // Table K.3
        dc_huffman_tables[0] = Some(
            HuffmanTable::new(
                &[
                    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00,
                ],
                &[
                    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
                ],
                HuffmanTableClass::DC,
            )
            .unwrap(),
        );
    }
    if dc_huffman_tables[1].is_none() && scan.dc_table_indices.contains(&1) {
        // Table K.4
        dc_huffman_tables[1] = Some(
            HuffmanTable::new(
                &[
                    0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
                    0x00, 0x00, 0x00,
                ],
                &[
                    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
                ],
                HuffmanTableClass::DC,
            )
            .unwrap(),
        );
    }
    if ac_huffman_tables[0].is_none() && scan.ac_table_indices.contains(&0) {
        // Table K.5
        ac_huffman_tables[0] = Some(
            HuffmanTable::new(
                &[
                    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00,
                    0x00, 0x01, 0x7D,
                ],
                &[
                    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13,
                    0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42,
                    0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A,
                    0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35,
                    0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
                    0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67,
                    0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84,
                    0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
                    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3,
                    0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
                    0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1,
                    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
                    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
                ],
                HuffmanTableClass::AC,
            )
            .unwrap(),
        );
    }
    if ac_huffman_tables[1].is_none() && scan.ac_table_indices.contains(&1) {
        // Table K.6
        ac_huffman_tables[1] = Some(
            HuffmanTable::new(
                &[
                    0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00,
                    0x01, 0x02, 0x77,
                ],
                &[
                    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51,
                    0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1,
                    0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24,
                    0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A,
                    0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
                    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66,
                    0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82,
                    0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
                    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA,
                    0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
                    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
                    0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4,
                    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
                ],
                HuffmanTableClass::AC,
            )
            .unwrap(),
        );
    }
}

/// Kani proofs for the magnitude-category arithmetic shared by the decoder's
/// `extend` and the encoder's `get_code`.
#[cfg(kani)]
mod proofs {
    use super::extend;

    /// `extend` is total and matches T.81 F.2.2.1 for every category the
    /// decoder can pass (1..=15) and every `count`-bit value.
    #[kani::proof]
    fn extend_matches_spec() {
        let count: u8 = kani::any_where(|c| (1..=15).contains(c));
        let value: u16 = kani::any_where(|v| u32::from(*v) < 1u32 << count);
        let got = i32::from(extend(value, count));
        let want = if u32::from(value) < 1u32 << (count - 1) {
            i32::from(value) - (1i32 << count) + 1
        } else {
            i32::from(value)
        };
        assert_eq!(got, want);
    }

    /// Encoder/decoder agreement: `extend(get_code(v)) == v` for every
    /// nonzero `|v| < 2^14`. At `|v| >= 2^14` (category 15) `get_code`'s
    /// `(1i16 << 15) - 1` overflows; the encoder never gets there for 8-bit
    /// samples (categories stop at 11), so the bound is the reachable range.
    #[kani::proof]
    fn get_code_extend_roundtrip() {
        let v: i16 = kani::any_where(|v: &i16| *v != 0 && v.unsigned_abs() < 1 << 14);
        let (size, bits) = crate::encode::get_code(v);
        assert!((1..=15).contains(&size));
        assert!(u32::from(bits) < 1u32 << size);
        assert_eq!(extend(bits, size), v);
    }

    /// `get_code(0)` is category 0, as DC/AC coding requires.
    #[kani::proof]
    fn get_code_zero() {
        assert_eq!(crate::encode::get_code(0).0, 0);
    }
}
