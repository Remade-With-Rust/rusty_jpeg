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

## [0.4.1](https://github.com/Remade-With-Rust/rusty_jpeg/compare/v0.4.0...v0.4.1) - 2026-09-03

### Other

- release v0.4.1 ([#4](https://github.com/Remade-With-Rust/rusty_jpeg/pull/4))
- the PSNR floor is 20 dB; the synthetic picture is deliberately harsh (24.7 dB at q75, both paths equal)
- YUYV input keeps the RGB path's PSNR (decode-encode-decode, corpus + synthetic)
- Merge origin/main (release-plz, v0.3.3, In-the-wild block) into 0.4.0

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
