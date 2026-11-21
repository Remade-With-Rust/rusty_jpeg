//! Interblock smoothing of an incomplete progressive image: a port of
//! libjpeg-turbo 3.x's `smoothing_ok` and `decompress_smooth_data`
//! (jdcoefct.c), and of the `coef_bits` bookkeeping behind them
//! (`start_pass_phuff_decoder`, jdphuff.c).
//!
//! A progressive JPEG sends each coefficient in instalments. When the data
//! stops early, some coefficients are missing or coarse, and an image rendered
//! from them as they stand is a mosaic of flat or blurred 8x8 blocks.
//! libjpeg-turbo does not show that: wherever one of the first nine AC
//! coefficients of a block is still zero and not yet known to full precision,
//! it estimates the coefficient from the DC values of the block and its 24
//! neighbours (and, where no AC data arrived at all, replaces the DC value by
//! a weighted average), then runs the IDCT. That is the picture every
//! libjpeg-based viewer shows for a cut-short progressive download, so it is
//! the picture to match.
//!
//! Only the decoder's handling of a stream that ENDS EARLY uses this; see
//! `Decoder::decode_planes` for why a complete stream never does.

use crate::decode::decoder::MAX_COMPONENTS;
use crate::decode::parser::{Component, ScanInfo};
use alloc::sync::Arc;

/// libjpeg's `SAVED_COEFS`: the DC coefficient and the first nine AC
/// coefficients in zigzag order -- the ones smoothing can estimate.
const SAVED_COEFS: usize = 10;

/// Natural-order positions of the first nine AC coefficients in zigzag order
/// (libjpeg's `Q01_POS` .. `Q30_POS`): AC01, AC10, AC20, AC11, AC02, AC03,
/// AC12, AC21, AC30.
const POSITIONS: [usize; SAVED_COEFS] = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24];

type CoefBits = [[i8; SAVED_COEFS]; MAX_COMPONENTS];

/// How far the scans read so far have taken each component: libjpeg's
/// `cinfo->coef_bits`, its shadow copy, `input_scan_number` and
/// `last_good_iMCU_row`.
pub(crate) struct Progression {
    /// Per component and coefficient: the point transform (`Al`) of the last
    /// scan that coded it -- 0 means known to full precision -- or -1 if no
    /// scan has.
    coef_bits: CoefBits,
    /// `coef_bits` as it stood before the component's latest scan began.
    prev_coef_bits: CoefBits,
    /// Scans begun.
    scans: u32,
    /// The last MCU row of the latest scan that still had data to decode.
    pub(crate) last_good_row: u16,
}

impl Default for Progression {
    fn default() -> Self {
        Progression {
            coef_bits: [[-1; SAVED_COEFS]; MAX_COMPONENTS],
            prev_coef_bits: [[-1; SAVED_COEFS]; MAX_COMPONENTS],
            scans: 0,
            last_good_row: 0,
        }
    }
}

impl Progression {
    /// A progressive scan begins: `start_pass_phuff_decoder`.
    pub(crate) fn start_scan(&mut self, scan: &ScanInfo) {
        self.scans += 1;
        let ss = usize::from(scan.spectral_selection.start);
        let se = usize::from(scan.spectral_selection.end) - 1;
        for &component in &scan.component_indices {
            // libjpeg saves `MIN(Ss, 1) ..= MAX(Se, 9)`; only 0..=9 is kept.
            for coefi in ss.min(1)..SAVED_COEFS {
                self.prev_coef_bits[component][coefi] = if self.scans > 1 {
                    self.coef_bits[component][coefi]
                } else {
                    0
                };
            }
            // Coefficients `ss ..= se` of the band, those of them that are kept.
            let al = scan.successive_approximation_low as i8;
            for bits in self.coef_bits[component]
                .iter_mut()
                .take(se.min(SAVED_COEFS - 1) + 1)
                .skip(ss)
            {
                *bits = al;
            }
        }
    }

    /// `smoothing_ok`: whether smoothing applies, and the state it works from.
    ///
    /// It applies when every component has its quantization table, with
    /// non-zero entries for the ten coefficients involved (they are divisors);
    /// when every component's DC is at least partly known; and when at least
    /// one of the nine AC coefficients of some component is not yet exact.
    pub(crate) fn smoothing(
        &self,
        components: &[Component],
        quantization_tables: &[Option<Arc<[u16; 64]>>; 4],
    ) -> Option<Smoothing> {
        let mut useful = false;
        let mut smoothing = Smoothing {
            coef_bits: self.coef_bits,
            prev_coef_bits: [[-1; SAVED_COEFS]; MAX_COMPONENTS],
            last_good_row: self.last_good_row,
        };
        for (ci, component) in components.iter().enumerate() {
            let table = quantization_tables[component.quantization_table_index].as_ref()?;
            if POSITIONS.iter().any(|&position| table[position] == 0) {
                return None;
            }
            if self.coef_bits[ci][0] < 0 {
                return None;
            }
            if self.scans > 1 {
                smoothing.prev_coef_bits[ci] = self.prev_coef_bits[ci];
            }
            useful |= self.coef_bits[ci][1..].iter().any(|&bits| bits != 0);
        }
        useful.then_some(smoothing)
    }
}

/// The latched state one smoothed output pass works from (libjpeg's
/// `coef_bits_latch`).
pub(crate) struct Smoothing {
    coef_bits: CoefBits,
    prev_coef_bits: CoefBits,
    last_good_row: u16,
}

/// The block grid of one component, as smoothing needs it.
pub(crate) struct Grid<'a> {
    /// Every coefficient of the component: `blocks_wide` blocks per row, 64
    /// coefficients per block, in natural order.
    pub(crate) coefficients: &'a [i16],
    /// Blocks per row in `coefficients`, padding blocks included.
    pub(crate) blocks_wide: usize,
    /// libjpeg's `width_in_blocks` / `height_in_blocks`: the blocks that hold
    /// image samples. Smoothing never reads a padding block to the right of
    /// them, and (in the last MCU row) below them.
    pub(crate) width_in_blocks: usize,
    pub(crate) height_in_blocks: usize,
    /// The component's vertical sampling factor: its block rows per MCU row.
    pub(crate) v_samp: usize,
    /// MCU rows in the image.
    pub(crate) mcu_rows: usize,
}

impl Grid<'_> {
    fn dc(&self, row: usize, column: usize) -> i64 {
        i64::from(self.coefficients[(row * self.blocks_wide + column) * 64])
    }
}

impl Smoothing {
    /// Smooth MCU row `mcu_row` of component `ci`: `row` holds a copy of that
    /// MCU row's coefficients (as `grid.coefficients` has them) and is
    /// modified in place. This is the body of `decompress_smooth_data` for one
    /// iMCU row of one component, up to (not including) the IDCT.
    pub(crate) fn smooth_mcu_row(
        &self,
        ci: usize,
        grid: &Grid,
        quantization_table: &[u16; 64],
        mcu_row: usize,
        row: &mut [i16],
    ) {
        // "If the current scan is incomplete, then we use the component-
        // dependent info from the previous scan."
        let coef_bits = if mcu_row > usize::from(self.last_good_row) {
            &self.prev_coef_bits[ci]
        } else {
            &self.coef_bits[ci]
        };
        // "We only do DC interpolation if no AC coefficient data is available."
        let change_dc = coef_bits[1..].iter().all(|&bits| bits == -1);
        let q: [i64; SAVED_COEFS] =
            core::array::from_fn(|i| i64::from(quantization_table[POSITIONS[i]]));

        // Block rows of this MCU row that hold image samples.
        let block_rows = if mcu_row + 1 < grid.mcu_rows {
            grid.v_samp
        } else {
            match grid.height_in_blocks % grid.v_samp {
                0 => grid.v_samp,
                rows => rows,
            }
        };
        // libjpeg computes both of these with the CURRENT row's `block_rows`,
        // which is what clamps the neighbourhood at the bottom of the image.
        let image_block_rows = block_rows * grid.mcu_rows;
        let last_column = grid.width_in_blocks - 1;

        for block_row in 0..block_rows {
            let image_block_row = mcu_row * block_rows + block_row;
            // Rows of the block grid: this one, two above and two below,
            // repeating the nearest one at the top and bottom edges.
            let this = mcu_row * grid.v_samp + block_row;
            let prev = if image_block_row > 0 { this - 1 } else { this };
            let prev_prev = if image_block_row > 1 { this - 2 } else { prev };
            let next = if image_block_row + 1 < image_block_rows {
                this + 1
            } else {
                this
            };
            let next_next = if image_block_row + 2 < image_block_rows {
                this + 2
            } else {
                next
            };
            let rows = [prev_prev, prev, this, next, next_next];

            // The 5x5 neighbourhood of DC values, DC01..DC25 row by row, as a
            // sliding register; "initialize all 25 here so as to do the right
            // thing on narrow pics".
            let mut dc = [[0i64; 5]; 5];
            for (values, &r) in dc.iter_mut().zip(&rows) {
                *values = [grid.dc(r, 0); 5];
            }

            for column in 0..=last_column {
                if column == 0 && column < last_column {
                    for (values, &r) in dc.iter_mut().zip(&rows) {
                        values[3] = grid.dc(r, 1);
                        values[4] = values[3];
                    }
                }
                if column + 1 < last_column {
                    for (values, &r) in dc.iter_mut().zip(&rows) {
                        values[4] = grid.dc(r, column + 2);
                    }
                }
                let [[d01, d02, d03, d04, d05], [d06, d07, d08, d09, d10], [d11, d12, d13, d14, d15], [d16, d17, d18, d19, d20], [d21, d22, d23, d24, d25]] =
                    dc;

                let offset = (block_row * grid.blocks_wide + column) * 64;
                let block: &mut [i16; 64] = (&mut row[offset..offset + 64]).try_into().unwrap();

                // "An estimate is applied only if the coefficient is still
                // zero and is not known to be fully accurate."
                let mut estimate = |coefi: usize, sum: i64| {
                    let al = coef_bits[coefi];
                    let position = POSITIONS[coefi];
                    if al != 0 && block[position] == 0 {
                        block[position] = predict(q[0] * sum, q[coefi], al) as i16;
                    }
                };

                // AC01
                estimate(
                    1,
                    if change_dc {
                        -d01 - d02 + d04 + d05 - 3 * d06 + 13 * d07 - 13 * d09 + 3 * d10 - 3 * d11
                            + 38 * d12
                            - 38 * d14
                            + 3 * d15
                            - 3 * d16
                            + 13 * d17
                            - 13 * d19
                            + 3 * d20
                            - d21
                            - d22
                            + d24
                            + d25
                    } else {
                        -7 * d11 + 50 * d12 - 50 * d14 + 7 * d15
                    },
                );
                // AC10
                estimate(
                    2,
                    if change_dc {
                        -d01 - 3 * d02 - 3 * d03 - 3 * d04 - d05 - d06
                            + 13 * d07
                            + 38 * d08
                            + 13 * d09
                            - d10
                            + d16
                            - 13 * d17
                            - 38 * d18
                            - 13 * d19
                            + d20
                            + d21
                            + 3 * d22
                            + 3 * d23
                            + 3 * d24
                            + d25
                    } else {
                        -7 * d03 + 50 * d08 - 50 * d18 + 7 * d23
                    },
                );
                // AC20
                estimate(
                    3,
                    if change_dc {
                        d03 + 2 * d07 + 7 * d08 + 2 * d09 - 5 * d12 - 14 * d13 - 5 * d14
                            + 2 * d17
                            + 7 * d18
                            + 2 * d19
                            + d23
                    } else {
                        -d03 + 13 * d08 - 24 * d13 + 13 * d18 - d23
                    },
                );
                // AC11
                estimate(
                    4,
                    if change_dc {
                        -d01 + d05 + 9 * d07 - 9 * d09 - 9 * d17 + 9 * d19 + d21 - d25
                    } else {
                        d10 + d16 - 10 * d17 + 10 * d19 - d02 - d20 + d22 - d24 + d04 - d06
                            + 10 * d07
                            - 10 * d09
                    },
                );
                // AC02
                estimate(
                    5,
                    if change_dc {
                        2 * d07 - 5 * d08 + 2 * d09 + d11 + 7 * d12 - 14 * d13
                            + 7 * d14
                            + d15
                            + 2 * d17
                            - 5 * d18
                            + 2 * d19
                    } else {
                        -d11 + 13 * d12 - 24 * d13 + 13 * d14 - d15
                    },
                );
                if change_dc {
                    // AC03
                    estimate(6, d07 - d09 + 2 * d12 - 2 * d14 + d17 - d19);
                    // AC12
                    estimate(7, d07 - 3 * d08 + d09 - d17 + 3 * d18 - d19);
                    // AC21
                    estimate(8, d07 - d09 - 3 * d12 + 3 * d14 + d17 - d19);
                    // AC30
                    estimate(9, d07 + 2 * d08 + d09 - d17 - 2 * d18 - d19);
                    // The DC value itself: a Gaussian-like average of the
                    // neighbourhood. Unconditional, and never clamped.
                    let sum = -2 * d01 - 6 * d02 - 8 * d03 - 6 * d04 - 2 * d05 - 6 * d06
                        + 6 * d07
                        + 42 * d08
                        + 6 * d09
                        - 6 * d10
                        - 8 * d11
                        + 42 * d12
                        + 152 * d13
                        + 42 * d14
                        - 8 * d15
                        - 6 * d16
                        + 6 * d17
                        + 42 * d18
                        + 6 * d19
                        - 6 * d20
                        - 2 * d21
                        - 6 * d22
                        - 8 * d23
                        - 6 * d24
                        - 2 * d25;
                    block[0] = predict(q[0] * sum, q[0], 0) as i16;
                }

                // "Advance for next column."
                for values in dc.iter_mut() {
                    values.copy_within(1.., 0);
                }
            }
        }
    }
}

/// `num / (256 * q)`, rounded to nearest, half away from zero; then, for a
/// coefficient whose bits above `al` are already known to be zero, limited to
/// what the missing low bits can hold. (`al <= 0` -- nothing known, or the DC
/// average -- is not limited.)
fn predict(num: i64, q: i64, al: i8) -> i64 {
    let magnitude = ((q << 7) + num.abs()) / (q << 8);
    let magnitude = if al > 0 && magnitude >= (1 << al) {
        (1 << al) - 1
    } else {
        magnitude
    };
    if num >= 0 {
        magnitude
    } else {
        -magnitude
    }
}
