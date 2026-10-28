"""Regenerate tests/fixtures/progressive_libjpeg.jpg.

The pixels are a synthetic 32x32 RGB gradient computed below; no external
image is involved. libjpeg (through Pillow) encodes it as a progressive JPEG,
which emits a Huffman table segment per scan, so the first, DC-only scan
names AC table 0 before any DHT defines it. tests/progressive.rs depends on
that layout, and this crate's own encoder cannot produce it.

Usage: python generate_progressive_libjpeg.py <out.jpg>
"""
import sys

from PIL import Image

W = H = 32
img = Image.new("RGB", (W, H))
img.putdata([(x * 255 // (W - 1), y * 255 // (H - 1), (x + y) * 255 // (W + H - 2))
             for y in range(H) for x in range(W)])
img.save(sys.argv[1], "JPEG", quality=75, progressive=True)
