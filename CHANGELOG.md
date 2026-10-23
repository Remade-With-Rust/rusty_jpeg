# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.3.3...v0.4.0) - 2026-09-02

The ESP32 release: `no_std` + `alloc`, a decoder that reads a slice, an
encoder that writes into a caller buffer (`SliceWriter`), packed YUYV
input, and a fix for optimized Huffman tables with restart intervals. The
full account is in [CHANGES.md](CHANGES.md#040).

## [Unreleased]

## [0.4.6](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.4.5...v0.4.6) - 2026-10-23

### Other

- release v0.4.6
- *(decode)* single-band output renders straight into a zeroed image

## [0.4.6](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.4.5...v0.4.6) - 2026-10-23

### Other

- *(decode)* single-band output renders straight into a zeroed image

## [0.4.5](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.4.4...v0.4.5) - 2026-10-23

### Other

- *(decode)* reuse component planes across decodes on the same thread

## [0.4.4](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.4.3...v0.4.4) - 2026-10-22

### Fixed

- *(decode)* select_worker compared width x width, not width x height

### Other

- *(decode)* don't thread a decode the process can only run on one core
- *(decode)* one-lookup DC differences
- *(decode)* render the RGB image into uninitialised capacity instead of zero-filling it first
- *(decode)* 9-bit first-level Huffman lookup (was 8)
- *(decode)* end-of-block in the fast AC table, and carry "DC only" from the entropy decoder
- *(decode)* AVX2 4:2:0 fancy upsampling, 16 sample pairs per step
- *(decode)* run the RGB output stage in row bands across threads in the default (threaded) mode
- *(decode)* AVX2 YCbCr->RGB, 16 pixels per step, bit-identical to SSSE3
- *(decode)* row-band RGB output stage — static upsampler dispatch, one set of line buffers, luma read in place, vectorizable H2V2/H2V1
- satisfy clippy 1.99 (div_ceil, single-element loop)
- 3,686 active installs

## [0.4.3](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.4.2...v0.4.3) - 2026-10-17

### Other

- bump rusty_alloc-api to =2.2.6 ([#11](https://github.com/Remade-With-Rust/rusty_jpeg/pull/11))
- Merge pull request #7 from Remade-With-Rust/chore/release-plz-0-5-139
- 3,262 active installs

## [0.4.2](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.4.1...v0.4.2) - 2026-10-08

### Other

- bump rusty_alloc-api to =2.2.5 ([#9](https://github.com/Remade-With-Rust/rusty_jpeg/pull/9))
- 2,784 active installs
- 1,439 active installs
- 1,435 active installs
- 1,168 active installs

## [0.4.1](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.4.0...v0.4.1) - 2026-09-03

### Other

- the PSNR floor is 20 dB; the synthetic picture is deliberately harsh (24.7 dB at q75, both paths equal)
- YUYV input keeps the RGB path's PSNR (decode-encode-decode, corpus + synthetic)
- Merge origin/main (release-plz, v0.3.3, In-the-wild block) into 0.4.0

## [0.3.3](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.3.2...v0.3.3) - 2026-08-28

### Other

- add release-plz so merged dependency bumps actually reach crates.io ([#1](https://github.com/Remade-With-Rust/rusty_jpeg/pull/1))
- Sync rusty_jpeg 0.3.2 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.3.1 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.3.0 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.2.3 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.2.2 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.2.1 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.2.0 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.7 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.7 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.7 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.6 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.5 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.5 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.4 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.3 from remade_ffmpeg_rs
- Sync rusty_jpeg 0.1.2 from remade_ffmpeg_rs
- Build warning-free
- rusty_jpeg 0.1.2 — pure-Rust JPEG/MJPEG decoder + encoder
