#![no_main]
//! 12-bit, arithmetic-coded and two-component lossless frames.
//!
//! Arbitrary bytes rarely get past the headers of these, so this target writes
//! valid headers for a mode picked by the input and hands the rest of it over
//! as everything that follows the frame header: scan data, and whatever
//! segments the fuzzer finds its way to.
use libfuzzer_sys::fuzz_target;
use rusty_jpeg::decode::Decoder;

fn segment(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&(body.len() as u16 + 2).to_be_bytes());
    out.extend_from_slice(body);
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let (head, rest) = data.split_at(8);
    let sof = [0xC1, 0xC2, 0xC3, 0xC9, 0xCA][usize::from(head[0] % 5)];
    let lossless = sof == 0xC3;
    let precision = match (lossless, head[1] % 2) {
        (true, _) => 2 + head[1] % 15,
        (false, 0) => 8,
        (false, _) => 12,
    };
    let components = match (lossless, head[2] % 4) {
        (true, n) => n + 1,
        (false, 0) => 1,
        (false, 1) => 4,
        (false, _) => 3,
    };
    let mut jpeg = vec![0xFF, 0xD8];
    let mut dqt = vec![0];
    dqt.extend((0..64).map(|i| head[3].wrapping_add(i)));
    segment(&mut jpeg, 0xDB, &dqt);
    let mut frame = vec![precision, 0, head[4] % 64 + 1, 0, head[5] % 64 + 1, components];
    for c in 0..components {
        let factors = match (lossless, c) {
            (false, 0) => [0x11, 0x21, 0x12, 0x22, 0x41][usize::from(head[6] % 5)],
            _ => 0x11,
        };
        frame.extend_from_slice(&[c + 1, factors, 0]);
    }
    segment(&mut jpeg, sof, &frame);
    // One table of each class: codes of 1 to 16 bits, all categories.
    for class in [0x00, 0x10] {
        let mut dht = vec![class, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        dht.extend((0..15).map(|i| if class == 0 { i } else { i * 17 + head[7] % 16 }));
        segment(&mut jpeg, 0xC4, &dht);
    }
    segment(&mut jpeg, 0xDD, &[0, head[7] % 5]);
    // A first scan, so that plain scan data is enough; progressive ones follow
    // from the input.
    let mut scan = vec![components];
    for c in 0..components {
        scan.extend_from_slice(&[c + 1, 0]);
    }
    scan.extend_from_slice(&match sof {
        0xC3 => [head[7] % 8, 0, 0],
        0xC2 | 0xCA => [0, 0, 0],
        _ => [0, 63, 0],
    });
    segment(&mut jpeg, 0xDA, &scan);
    jpeg.extend_from_slice(rest);

    let mut d = Decoder::new(&jpeg[..]);
    d.set_max_decoding_buffer_size(64 * 1024 * 1024);
    d.set_single_threaded(true);
    d.set_strict(head[3] & 0x40 != 0);
    if let Ok(samples) = d.decode_u16() {
        let h = d.frame_header().unwrap();
        assert_eq!(
            samples.len(),
            usize::from(h.width) * usize::from(h.height) * usize::from(h.components)
        );
        assert!(h.precision == 16 || lossless || samples.iter().all(|&s| s >> h.precision == 0));
    }
});
