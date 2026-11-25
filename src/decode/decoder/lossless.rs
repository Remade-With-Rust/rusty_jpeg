use crate::decode::decoder::{process_restart, Decoder, MAX_COMPONENTS};
use crate::decode::error::{Error, Result};
use crate::decode::huffman::HuffmanDecoder;
use crate::decode::marker::Marker;
use crate::decode::parser::Predictor;
use crate::decode::parser::{Component, FrameInfo, ScanInfo};
use crate::decode::Source;
use alloc::borrow::ToOwned;
use alloc::vec;
use alloc::vec::Vec;

impl<R: Source> Decoder<R> {
    /// decode_scan_lossless
    pub fn decode_scan_lossless(
        &mut self,
        frame: &FrameInfo,
        scan: &ScanInfo,
    ) -> Result<(Option<Marker>, Vec<Vec<u16>>)> {
        let ncomp = scan.component_indices.len();
        let npixel = frame.image_size.height as usize * frame.image_size.width as usize;
        assert!(ncomp <= MAX_COMPONENTS);
        let mut results = vec![vec![0u16; npixel]; ncomp];

        let components: Vec<Component> = scan
            .component_indices
            .iter()
            .map(|&i| frame.components[i].clone())
            .collect();

        // Verify that all required huffman tables has been set.
        if scan
            .dc_table_indices
            .iter()
            .any(|&i| self.dc_huffman_tables[i].is_none())
        {
            return Err(Error::Format(
                "scan makes use of unset dc huffman table".to_owned(),
            ));
        }

        let mut huffman = HuffmanDecoder::new(self.warnings.strict);
        let mut mcus_left_until_restart = self.restart_interval;
        let mut expected_rst_num = 0;

        let width = frame.image_size.width as usize;
        let height = frame.image_size.height as usize;

        // libjpeg-turbo's lossless decoder works a row at a time
        // (`decompress_data`, jddiffct.c; `decode_mcus`, jdlhuff.c), and so do
        // the two things below that a damaged or cut-short stream needs.
        //
        // `insufficient`: the segment's data is exhausted (see `decode_scan`).
        // The row in which it runs out is finished from the inserted zero
        // bits; every later row of the segment is not decoded at all and comes
        // out uniform mid-grey (`row_filled`).
        //
        // `row_restarts`: a restart marker was crossed at the start of the
        // row, so its prediction starts afresh like the first row of the scan
        // (H.1.1: "at the beginning of each restart interval"). Both vectors
        // stay empty, and cost nothing, for a stream that never needs them.
        let mut insufficient = false;
        let mut row_filled: Vec<bool> = Vec::new();
        let mut row_restarts: Vec<bool> = Vec::new();

        let mut differences = vec![Vec::with_capacity(npixel); ncomp];
        for mcu_y in 0..height {
            // Checked before a restart below resets the bit reader.
            if huffman.ran_out() && !insufficient {
                insufficient = true;
                self.segment_ended_early(&huffman)?;
            }
            let mut fill_row = false;

            for mcu_x in 0..width {
                if self.restart_interval > 0 {
                    if mcus_left_until_restart == 0 {
                        // Out-of-step restart markers resynchronise as in the
                        // DCT path (see `process_restart`).
                        let consumed = process_restart(
                            &mut self.reader,
                            &mut huffman,
                            expected_rst_num,
                            &mut self.warnings,
                        )?;

                        expected_rst_num = (expected_rst_num + 1) % 8;
                        mcus_left_until_restart = self.restart_interval;
                        // A fresh segment has data again -- unless the marker
                        // was left unread, in which case it is empty.
                        if consumed {
                            insufficient = false;
                        }
                        // libjpeg-turbo refuses a lossless restart interval
                        // that is not a whole number of rows. Such a restart,
                        // in the middle of a row, is left as it was here: the
                        // bit reader is reset and the prediction is not.
                        if mcu_x == 0 {
                            row_restarts.resize(height, false);
                            row_restarts[mcu_y] = true;
                        }
                    }

                    mcus_left_until_restart -= 1;
                }

                if mcu_x == 0 && insufficient {
                    fill_row = true;
                    row_filled.resize(height, false);
                    row_filled[mcu_y] = true;
                }
                if fill_row {
                    for difference in differences.iter_mut() {
                        difference.push(0);
                    }
                    continue;
                }

                for (i, _component) in components.iter().enumerate() {
                    let dc_table = self.dc_huffman_tables[scan.dc_table_indices[i]]
                        .as_ref()
                        .unwrap();
                    let value = huffman.decode(&mut self.reader, dc_table)?;
                    let diff = match value {
                        0 => 0,
                        1..=15 => huffman.receive_extend(&mut self.reader, value)? as i32,
                        16 => 32768,
                        _ => {
                            // Section F.1.2.1.1
                            // Table F.1
                            return Err(Error::Format(
                                "invalid DC difference magnitude category".to_owned(),
                            ));
                        }
                    };
                    differences[i].push(diff);
                }
            }
        }

        // H.1.2.1. The first sample of the scan, and of each restart interval,
        // is predicted by 2^(P-Pt-1); the rest of that row by the sample to
        // its left (Ra). In every other row the first sample is predicted by
        // the one above (Rb) and the rest by the scan's predictor.
        let initial = if scan.predictor_selection == Predictor::Ra
            || frame.precision > 1 + scan.point_transform
        {
            1i32 << (frame.precision - scan.point_transform - 1)
        } else {
            0
        };
        // What libjpeg-turbo's undifferencer gives a row with no data: zero
        // differences from a freshly reset prediction, so the initial value in
        // every sample.
        let grey = (initial & 0xFFFF) as u16;
        for (result, difference) in results.iter_mut().zip(&differences) {
            for y in 0..height {
                let row = y * width;
                if row_filled.get(y).copied().unwrap_or(false) {
                    result[row..row + width].fill(grey);
                    continue;
                }
                let first_row = y == 0 || row_restarts.get(y).copied().unwrap_or(false);
                for x in 0..width {
                    let prediction = if x == 0 {
                        if first_row {
                            initial
                        } else {
                            result[row - width] as i32
                        }
                    } else {
                        let ra = result[row + x - 1] as i32;
                        if first_row {
                            ra
                        } else {
                            let rb = result[row - width + x] as i32;
                            let rc = result[row - width + x - 1] as i32;
                            predict(ra, rb, rc, scan.predictor_selection)
                        }
                    };
                    let sample = ((prediction + difference[row + x]) & 0xFFFF) as u16; // modulo 2^16
                    result[row + x] = sample;
                }
            }
            // The point transform (H.1.2.1): what is coded, and predicted, is
            // the sample divided by 2^Pt; the output is that scaled back up.
            // The scaling has to wait until the whole component is predicted
            // -- applying it sample by sample, as this did, fed the scaled
            // values back into the prediction, and an image coded with Pt > 0
            // decoded wrongly.
            if scan.point_transform > 0 {
                for sample in result.iter_mut() {
                    *sample <<= scan.point_transform;
                }
            }
        }

        let marker = self.finish_scan(&mut huffman)?;
        Ok((marker, results))
    }
}

/// Table H.1: the prediction for a sample that has all three neighbours.
fn predict(ra: i32, rb: i32, rc: i32, predictor: Predictor) -> i32 {
    match predictor {
        Predictor::NoPrediction => 0,
        Predictor::Ra => ra,
        Predictor::Rb => rb,
        Predictor::Rc => rc,
        Predictor::RaRbRc1 => ra + rb - rc,
        Predictor::RaRbRc2 => ra + ((rb - rc) >> 1),
        Predictor::RaRbRc3 => rb + ((ra - rc) >> 1),
        Predictor::RaRb => (ra + rb) / 2,
    }
}

/// The image's samples, interleaved.
pub fn compute_image_lossless(frame: &FrameInfo, mut data: Vec<Vec<u16>>) -> Result<Vec<u16>> {
    if data.is_empty() || data.iter().any(Vec::is_empty) {
        return Err(Error::Format("not all components have data".to_owned()));
    }
    let output_size = frame.output_size;
    let ncomp = frame.components.len();

    if ncomp == 1 {
        return Ok(data.remove(0));
    }
    let mut decoded: Vec<u16> =
        vec![0u16; ncomp * output_size.width as usize * output_size.height as usize];
    for (x, chunk) in decoded.chunks_mut(ncomp).enumerate() {
        for (sample, component_data) in chunk.iter_mut().zip(&data) {
            *sample = component_data[x];
        }
    }
    Ok(decoded)
}

/// Samples as the bytes `decode` returns them in.
pub(super) fn convert_to_u8(frame: &FrameInfo, data: Vec<u16>) -> Vec<u8> {
    if frame.precision == 8 {
        data.iter().map(|x| *x as u8).collect()
    } else {
        // we output native endian, which is the standard for image-rs
        let mut bytes = vec![0u8; data.len() * 2];
        for (pair, sample) in bytes.chunks_exact_mut(2).zip(&data) {
            pair.copy_from_slice(&sample.to_ne_bytes());
        }
        bytes
    }
}
