//! Scans of arithmetic-coded frames (SOF9 and SOF10).

use crate::decode::arith::{self, ArithDecoder, DcState, Stats};
use crate::decode::decoder::{Decoder, MAX_COMPONENTS};
use crate::decode::error::{Error, Result, Warning};
use crate::decode::marker::Marker;
use crate::decode::parser::{FrameInfo, ScanInfo};
use crate::decode::Source;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;

impl<R: Source> Decoder<R> {
    /// Decode one scan into the stored coefficients (`self.coefficients`),
    /// sequential and progressive alike. Returns the marker that ended it, if
    /// the decoder ran into one.
    pub(super) fn decode_scan_arithmetic(
        &mut self,
        frame: &FrameInfo,
        scan: &ScanInfo,
    ) -> Result<Option<Marker>> {
        let components = &scan.component_indices;
        assert!(components.len() <= MAX_COMPONENTS);
        if components.iter().any(|&i| {
            self.quantization_tables[frame.components[i].quantization_table_index].is_none()
        }) {
            return Err(Error::Format("use of unset quantization table".to_owned()));
        }

        let band = scan.spectral_selection.clone();
        let al = scan.successive_approximation_low;
        let codes_dc = band.start == 0;
        let codes_ac = band.end > 1;
        let refinement = scan.successive_approximation_high != 0;

        // The MCUs of the scan, and what one holds of each component (4.8.2):
        // in a scan of one component a single block, and only those blocks
        // that carry samples of the image.
        let interleaved = components.len() > 1;
        let (mcus_wide, mcus_high) = if interleaved {
            (frame.mcu_size.width, frame.mcu_size.height)
        } else {
            frame.image_blocks(components[0])
        };

        let mut d = ArithDecoder::new(self.warnings.strict);
        let mut stats = Box::new(Stats::new());
        let mut dc = [DcState::default(); MAX_COMPONENTS];
        let mut mcus_left_until_restart = self.restart_interval;
        let mut expected_rst_num = 0;

        for mcu_y in 0..usize::from(mcus_high) {
            for mcu_x in 0..usize::from(mcus_wide) {
                if self.restart_interval > 0 {
                    if mcus_left_until_restart == 0 {
                        d.restart(&mut self.reader, expected_rst_num, &mut self.warnings)?;
                        for (&dc_table, &ac_table) in
                            scan.dc_table_indices.iter().zip(&scan.ac_table_indices)
                        {
                            stats.reset(dc_table, ac_table);
                        }
                        dc = Default::default();
                        expected_rst_num = (expected_rst_num + 1) % 8;
                        mcus_left_until_restart = self.restart_interval;
                    }
                    mcus_left_until_restart -= 1;
                }

                'mcu: for (i, &ci) in components.iter().enumerate() {
                    let component = &frame.components[ci];
                    let (across, down) = if interleaved {
                        (
                            usize::from(component.horizontal_sampling_factor),
                            usize::from(component.vertical_sampling_factor),
                        )
                    } else {
                        (1, 1)
                    };
                    let dc_table = scan.dc_table_indices[i];
                    let ac_table = scan.ac_table_indices[i];
                    for v in 0..down {
                        for h in 0..across {
                            // The rest of a segment that went wrong is left as
                            // it is: zero, or as earlier scans had it.
                            if d.failed() {
                                break 'mcu;
                            }
                            let block = (mcu_y * down + v)
                                * usize::from(component.block_size.width)
                                + mcu_x * across
                                + h;
                            let coefficients: &mut [i16; 64] = (&mut self.coefficients[ci]
                                [block * 64..block * 64 + 64])
                                .try_into()
                                .unwrap();
                            let nonzero = &mut self.nonzero[ci][block];

                            if codes_dc && refinement {
                                arith::refine_dc(&mut d, &mut self.reader, al, coefficients)?;
                            } else if codes_dc {
                                match arith::decode_dc(
                                    &mut d,
                                    &mut self.reader,
                                    &mut stats,
                                    dc_table,
                                    self.conditioning.dc[dc_table],
                                    &mut dc[i],
                                )? {
                                    Some(value) => coefficients[0] = (value << al) as i16,
                                    None => break 'mcu,
                                }
                            }
                            if codes_ac && refinement {
                                arith::refine_ac(
                                    &mut d,
                                    &mut self.reader,
                                    &mut stats,
                                    ac_table,
                                    band.clone(),
                                    al,
                                    coefficients,
                                    nonzero,
                                )?;
                            } else if codes_ac {
                                arith::decode_ac(
                                    &mut d,
                                    &mut self.reader,
                                    &mut stats,
                                    ac_table,
                                    self.conditioning.ac[ac_table],
                                    band.clone(),
                                    al,
                                    coefficients,
                                    nonzero,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        // libjpeg's arithmetic decoder never reports exhausted data (its
        // `insufficient_data` stays FALSE), running into a marker early being
        // legal. So the smoothing of a truncated progressive image takes every
        // row of the scan for decoded.
        self.progression.last_good_row = frame.mcu_size.height.saturating_sub(1);

        if d.saw_bad_code() {
            self.warnings.raise(Warning::BadArithmeticCode)?;
        }
        let marker = d.source().take_pending_marker();
        let marker = self.skip_stray_markers(marker)?;
        if d.source().hit_end_of_data() {
            self.warnings.raise(Warning::Truncated)?;
        }
        Ok(marker)
    }
}
