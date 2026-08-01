use super::{RowData, Worker};
use crate::decode::decoder::MAX_COMPONENTS;
use crate::decode::error::Result;
use crate::decode::idct::dequantize_and_idct_block;
use crate::decode::parser::Component;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;

pub struct ImmediateWorker {
    offsets: [usize; MAX_COMPONENTS],
    results: Vec<Vec<u8>>,
    components: Vec<Option<Component>>,
    quantization_tables: Vec<Option<Arc<[u16; 64]>>>,
    /// Last coefficient buffer this worker finished with, offered back to the
    /// caller so it can be refilled rather than reallocated.
    spare: Option<Vec<i16>>,
}

impl Default for ImmediateWorker {
    fn default() -> Self {
        ImmediateWorker {
            offsets: [0; MAX_COMPONENTS],
            results: vec![Vec::new(); MAX_COMPONENTS],
            components: vec![None; MAX_COMPONENTS],
            quantization_tables: vec![None; MAX_COMPONENTS],
            spare: None,
        }
    }
}

/// `RUSTY_JPEG_ABLATE=planeinit` — see `start_immediate`.
pub(crate) fn ablate_plane_init() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("RUSTY_JPEG_ABLATE")
            .map(|v| v.split(',').any(|t| t == "planeinit"))
            .unwrap_or(false)
    })
}

impl ImmediateWorker {
    pub fn start_immediate(&mut self, data: RowData) {
        let _s = crate::prof::scope(crate::prof::Stage::DecPlaneInit);
        assert!(self.results[data.index].is_empty());
        // `RUSTY_JPEG_ABLATE=planeinit` skips sizing/zeroing the output plane.
        // Only valid together with `stores`, since nothing may write into it.
        if crate::decode::idct::ablate_stores() && ablate_plane_init() {
            self.components[data.index] = Some(data.component);
            self.quantization_tables[data.index] = Some(data.quantization_table);
            return;
        }

        self.offsets[data.index] = 0;
        // Refill a recycled allocation when the caller supplied one: its pages
        // are already resident, which is the entire cost being avoided.
        if let Some(buf) = data.recycled {
            self.results[data.index] = buf;
            self.results[data.index].clear();
        }
        self.results[data.index].resize(
            data.component.block_size.width as usize
                * data.component.block_size.height as usize
                * data.component.dct_scale
                * data.component.dct_scale,
            0u8,
        );
        self.components[data.index] = Some(data.component);
        self.quantization_tables[data.index] = Some(data.quantization_table);
    }

    pub fn append_row_immediate(&mut self, (index, data): (usize, Vec<i16>)) {
        // Convert coefficients from a MCU row to samples.

        let component = self.components[index].as_ref().unwrap();
        let quantization_table = self.quantization_tables[index].as_ref().unwrap();
        let block_count =
            component.block_size.width as usize * component.vertical_sampling_factor as usize;
        let line_stride = component.block_size.width as usize * component.dct_scale;

        assert_eq!(data.len(), block_count * 64);

        // Two horizontally adjacent full blocks can share one AVX2 instruction
        // stream (see `arch::avx2`). Worth pairing only when BOTH need a real
        // transform: a DC-only block is a fill, which is cheaper than half a
        // vectorized IDCT, and ~31% of blocks on photographic content are.
        let blocks_wide = component.block_size.width as usize;
        let pair_idct = if component.dct_scale == 8 && !crate::decode::idct::ablate_idct() {
            crate::decode::arch::get_dequantize_and_idct_block_8x8_pair()
        } else {
            None
        };

        let mut i = 0;
        while i < block_count {
            let x = (i % blocks_wide) * component.dct_scale;
            let y = (i / blocks_wide) * component.dct_scale;
            let coefficients: &[i16; 64] = data[i * 64..(i + 1) * 64].try_into().unwrap();

            crate::prof::bump(crate::prof::Count::DecBlocks, 1);
            let dc_only = crate::decode::idct::is_dc_only(coefficients);
            if dc_only {
                crate::prof::bump(crate::prof::Count::DecDcOnlyBlocks, 1);
            }

            // Pair with the next block when it exists, sits on the SAME block
            // row (so its origin is exactly `dct_scale` bytes along), and also
            // needs a full transform.
            if let Some(idct_pair) = pair_idct {
                if !dc_only && i + 1 < block_count && (i % blocks_wide) + 1 < blocks_wide {
                    let next: &[i16; 64] =
                        data[(i + 1) * 64..(i + 2) * 64].try_into().unwrap();
                    if !crate::decode::idct::is_dc_only(next) {
                        crate::prof::bump(crate::prof::Count::DecBlocks, 1);
                        crate::prof::bump(crate::prof::Count::DecIdctPairs, 1);
                        let _s = crate::prof::scope(crate::prof::Stage::DecIdct);
                        let output =
                            &mut self.results[index][self.offsets[index] + y * line_stride + x..];
                        #[allow(unsafe_code)]
                        unsafe {
                            idct_pair(
                                coefficients,
                                next,
                                quantization_table,
                                line_stride,
                                output,
                                component.dct_scale,
                            );
                        }
                        i += 2;
                        continue;
                    }
                }
            }

            let output = &mut self.results[index][self.offsets[index] + y * line_stride + x..];
            let _s = crate::prof::scope(crate::prof::Stage::DecIdct);
            if dc_only && component.dct_scale == 8 {
                // Already scanned above; going through `dequantize_and_idct_block`
                // here would scan all 63 AC coefficients a second time, and on
                // photographic content ~31% of blocks land in this branch.
                crate::decode::idct::fill_dc_only(
                    coefficients,
                    quantization_table,
                    line_stride,
                    output,
                );
            } else {
                dequantize_and_idct_block(
                    component.dct_scale,
                    coefficients,
                    quantization_table,
                    line_stride,
                    output,
                );
            }
            i += 1;
        }

        self.offsets[index] += block_count * component.dct_scale * component.dct_scale;

        // This worker consumed the row synchronously, so the buffer is free
        // now. Keep it for the caller to refill instead of dropping it and
        // making them allocate + zero a replacement for every MCU row.
        self.spare = Some(data);
    }

    pub fn get_result_immediate(&mut self, index: usize) -> Vec<u8> {
        mem::take(&mut self.results[index])
    }
}

impl Worker for ImmediateWorker {
    fn reclaim_buffer(&mut self) -> Option<Vec<i16>> {
        self.spare.take()
    }

    fn start(&mut self, data: RowData) -> Result<()> {
        self.start_immediate(data);
        Ok(())
    }
    fn append_row(&mut self, row: (usize, Vec<i16>)) -> Result<()> {
        self.append_row_immediate(row);
        Ok(())
    }
    fn get_result(&mut self, index: usize) -> Result<Vec<u8>> {
        Ok(self.get_result_immediate(index))
    }
}
