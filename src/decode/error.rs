use alloc::boxed::Box;
use alloc::fmt;
use alloc::string::String;
use alloc::vec::Vec;
use core::error::Error as CoreError;
use core::result;
#[cfg(feature = "std")]
use std::io::Error as IoError;

use crate::decode::ColorTransform;

pub type Result<T> = result::Result<T, Error>;

/// An enumeration over JPEG features (currently) unsupported by this library.
///
/// Support for features listed here may be included in future versions of this library.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnsupportedFeature {
    /// Hierarchical JPEG.
    Hierarchical,
    /// Lossless JPEG using arithmetic entropy coding (SOF11). Arithmetic-coded
    /// DCT frames (SOF9, SOF10) are supported.
    ArithmeticEntropyCoding,
    /// Sample precision in bits. 8 and 12 bits are supported in DCT frames, 2 to 16 in lossless ones;
    /// [`Decoder::decode_planar`](crate::decode::Decoder::decode_planar) returns 8-bit planes only.
    SamplePrecision(u8),
    /// Number of components in an image. 1, 3 and 4 components are currently supported, and 2 in
    /// lossless frames.
    ComponentCount(u8),
    /// An image can specify a zero height in the frame header and use the DNL (Define Number of
    /// Lines) marker at the end of the first scan to define the number of lines in the frame.
    DNL,
    /// Subsampling ratio.
    SubsamplingRatio,
    /// A subsampling ratio not representable as an integer.
    NonIntegerSubsamplingRatio,
    /// Colour transform
    ColorTransform(ColorTransform),
}

/// Errors that can occur while decoding a JPEG image.
#[derive(Debug)]
pub enum Error {
    /// The image is not formatted properly. The string contains detailed information about the
    /// error.
    Format(String),
    /// The image makes use of a JPEG feature not (currently) supported by this library.
    Unsupported(UnsupportedFeature),
    /// An I/O error occurred while decoding the image.
    #[cfg(feature = "std")]
    Io(IoError),
    /// The data ended before there was an image to show: inside the headers,
    /// or before the first scan had any data. Raised in place of an
    /// `Io(UnexpectedEof)` on every path, with or without `std`, so a
    /// receiver can tell a truncated frame from a corrupt one.
    ///
    /// Data that ends later than that is decoded as far as it goes and
    /// reported as [`Warning::Truncated`] -- unless the decoder is strict
    /// ([`Decoder::set_strict`](crate::decode::Decoder::set_strict)), in which
    /// case it is this error wherever the data ends short of the
    /// end-of-image marker.
    UnexpectedEof,
    /// An internal error occurred while decoding the image.
    Internal(Box<dyn CoreError + Send + Sync + 'static>), //TODO: not used, can be removed with the next version bump
}

/// A defect the decoder recovered from instead of refusing the image.
///
/// By default the decoder recovers a damaged or incomplete JPEG the way
/// libjpeg-turbo does: it decodes what is there, fills what is not, and goes
/// on. libjpeg-turbo reports each such recovery as a warning; so does this
/// decoder, through [`Decoder::warnings`](crate::decode::Decoder::warnings).
/// A decode that returns `Ok` with no warnings read a complete, well-formed
/// stream. [`Decoder::set_strict`](crate::decode::Decoder::set_strict) turns
/// every one of these into an [`Error`] instead.
///
/// Each kind is reported at most once per decode, however often it occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Warning {
    /// The data ended before the end-of-image marker: a download cut short, a
    /// crashed write. The image is decoded as if an EOI followed the last
    /// byte, and whatever was not reached is filled (see
    /// [`Decoder::warnings`](crate::decode::Decoder::warnings) for how).
    /// libjpeg-turbo: "Premature end of JPEG file".
    Truncated,
    /// An entropy-coded segment ran out of data (it ended at a marker, or at
    /// the end of the data) before all of its MCUs were decoded. The rest of
    /// the segment is uniform grey, or left as earlier scans had it in a
    /// progressive image. Also raised when only the scan's last MCU is short:
    /// that MCU is finished from zero bits, as in libjpeg-turbo, so nothing
    /// else changes. libjpeg-turbo: "Corrupt JPEG data: premature end of data
    /// segment".
    SegmentEndedEarly,
    /// A bit sequence that is no code of the scan's Huffman table. It reads as
    /// symbol 0 and decoding continues, out of step until the next restart
    /// marker (or the end of the scan). libjpeg-turbo: "Corrupt JPEG data:
    /// bad Huffman code".
    BadHuffmanCode,
    /// The marker at a restart boundary was not the restart marker due there;
    /// the decoder resynchronised as libjpeg-turbo's `jpeg_resync_to_restart`
    /// does. libjpeg-turbo: "Corrupt JPEG data: found marker 0x.. instead of
    /// RSTn".
    RestartMarkerMismatch,
    /// Bytes that belong to no marker segment and no scan were skipped while
    /// looking for a marker. libjpeg-turbo: "Corrupt JPEG data: N extraneous
    /// bytes before marker 0x..".
    ExtraneousBytes,
    /// The image ended with no scan data at all for a component. Its plane is
    /// uniform mid-grey, which is what libjpeg-turbo outputs for it.
    MissingScan,
    /// An arithmetic-coded coefficient ran past the end of its block or out of
    /// range. Nothing more of the segment is decoded, up to the next restart
    /// marker (or the end of the scan). libjpeg-turbo: "Corrupt JPEG data: bad
    /// arithmetic code".
    BadArithmeticCode,
}

impl Warning {
    /// The error this warning is in strict mode. A truncated stream is
    /// [`Error::UnexpectedEof`], as it was before the decoder could recover
    /// one, so a receiver can still tell "more data may come" from "corrupt".
    pub(crate) fn into_error(self) -> Error {
        match self {
            Warning::Truncated => Error::UnexpectedEof,
            Warning::BadHuffmanCode => Error::Format("failed to decode huffman code".into()),
            other => Error::Format(alloc::format!("{}", other)),
        }
    }
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            Warning::Truncated => "premature end of JPEG data",
            Warning::SegmentEndedEarly => "premature end of data segment",
            Warning::BadHuffmanCode => "bad Huffman code",
            Warning::BadArithmeticCode => "bad arithmetic code",
            Warning::RestartMarkerMismatch => "restart marker missing or out of sequence",
            Warning::ExtraneousBytes => "extraneous bytes before marker",
            Warning::MissingScan => "no scan data for a component",
        })
    }
}

/// The warnings of one decode, and whether they are errors instead.
#[derive(Debug, Default)]
pub(crate) struct Warnings {
    list: Vec<Warning>,
    pub(crate) strict: bool,
}

impl Warnings {
    /// Record `warning` (once per kind), or fail with it in strict mode.
    #[cold]
    pub(crate) fn raise(&mut self, warning: Warning) -> Result<()> {
        if self.strict {
            return Err(warning.into_error());
        }
        if !self.list.contains(&warning) {
            self.list.push(warning);
        }
        Ok(())
    }

    pub(crate) fn as_slice(&self) -> &[Warning] {
        &self.list
    }

    pub(crate) fn clear(&mut self) {
        self.list.clear();
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Format(ref desc) => write!(f, "invalid JPEG format: {}", desc),
            Error::Unsupported(ref feat) => write!(f, "unsupported JPEG feature: {:?}", feat),
            #[cfg(feature = "std")]
            Error::Io(ref err) => err.fmt(f),
            Error::UnexpectedEof => write!(f, "unexpected end of JPEG data"),
            Error::Internal(ref err) => err.fmt(f),
        }
    }
}

impl CoreError for Error {
    fn source(&self) -> Option<&(dyn CoreError + 'static)> {
        match *self {
            #[cfg(feature = "std")]
            Error::Io(ref err) => Some(err),
            Error::Internal(ref err) => Some(&**err),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<IoError> for Error {
    fn from(err: IoError) -> Error {
        Error::Io(err)
    }
}
