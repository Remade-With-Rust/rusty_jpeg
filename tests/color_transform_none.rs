//! `ColorTransform::None` must return the stored components interleaved.
//!
//! Regression test for a report from ArtCraft (rusty_jpeg 0.4.6): four-channel
//! JPEGs decoded with `ColorTransform::None` came back as per-row planes
//! (`C..C M..M Y..Y K..K`) in a buffer typed as interleaved CMYK32.

use rusty_jpeg::decode::{ColorTransform, Decoder};

const SOLID: &[u8] = include_bytes!("fixtures/solid_cmyk_adobe.jpg");

#[test]
fn none_returns_interleaved_stored_components() {
    let mut d = Decoder::new(SOLID);
    d.set_color_transform(ColorTransform::None);
    let px = d.decode().unwrap();
    let info = d.info().unwrap();
    assert_eq!(px.len(), info.width as usize * info.height as usize * 4);
    // Hand-built: one 8x8 block per component, stored C M Y K = 0 255 255 255.
    for (i, p) in px.chunks_exact(4).enumerate() {
        assert_eq!(p, [0, 255, 255, 255], "pixel {i}");
    }
}

#[test]
fn none_is_the_inverse_of_the_default_adobe_cmyk_output() {
    // For Adobe-inverted CMYK the default transform emits 255 - stored, so the
    // raw components are exactly 255 - default.
    let raw = {
        let mut d = Decoder::new(SOLID);
        d.set_color_transform(ColorTransform::None);
        d.decode().unwrap()
    };
    let default = Decoder::new(SOLID).decode().unwrap();
    assert_eq!(raw.len(), default.len());
    for (r, d) in raw.iter().zip(&default) {
        assert_eq!(*r, 255 - *d);
    }
}
