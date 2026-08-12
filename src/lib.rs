//! VENUS Crop TIFF library: folder loading (TIFF stack + whole-stack
//! projections), the crop rectangle and its JSON format, and the per-frame
//! statistics used to verify the crop against every image.
//!
//! The GUI binary (`main.rs`) is a thin shell around these modules; they are
//! exposed here so they can be unit tested without a display.

pub mod app;
pub mod colormap;
pub mod crop;
pub mod loader;
pub mod stats;
pub mod theme;
