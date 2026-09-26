//! The crop region: a single, pixel-aligned rectangle on the image, plus the
//! JSON format used to hand it to other applications (e.g. a marimo notebook
//! that applies the crop to the full stack before processing it).
//!
//! Two representations: [`RectF`] is the free-floating rectangle being drawn
//! or dragged on screen; [`CropRect`] is the integer region actually saved and
//! used for statistics, always clamped inside the image.
//!
//! A third representation, [`OrientedBox`], is the region actually edited and
//! exported: a center, a size and a tilt angle. Untilted it is exactly a
//! [`CropRect`] and keeps every format below unchanged; tilted, the JSON adds
//! `center_x`/`center_y`/`box_width`/`box_height`/`angle_deg` (on the
//! oriented frames, as drawn) and `x`..`height` hold the axis-aligned
//! bounding box of the tilted region in the on-disk frame, so untilted
//! readers still find where the region is. A tilted region is exported
//! straightened: every cropped image is `box_width × box_height` pixels,
//! resampled bilinearly on the box's own grid (see [`crate::extract`]).
//!
//! Coordinate frames: on screen the crop is drawn on the *oriented* frames
//! (Timepix stacks are transposed on load, CCD stacks flipped vertically and horizontally —
//! see [`detector_orientation`]). The JSON handed to other applications, the
//! `--crop` / `--crop-file` inputs and every cropped file written to disk are
//! in the **on-disk** frame of the input files, so a caller can slice the
//! original files with `frame[y:y+height, x:x+width]` without knowing about
//! the display orientation. [`CropRect::to_disk`] / [`CropRect::from_disk`]
//! convert between the two.

use detector_orientation::{Detector, Orientation};

/// A floating-point rectangle in image-pixel space, as edited on screen.
/// Corners may be in any order and outside the image; [`RectF::to_crop`]
/// normalizes, rounds and clamps.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RectF {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl RectF {
    pub fn normalized(self) -> RectF {
        RectF {
            x0: self.x0.min(self.x1),
            y0: self.y0.min(self.y1),
            x1: self.x0.max(self.x1),
            y1: self.y0.max(self.y1),
        }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        let r = self.normalized();
        px >= r.x0 && px <= r.x1 && py >= r.y0 && py <= r.y1
    }

    pub fn translate(&mut self, dx: f32, dy: f32) {
        self.x0 += dx;
        self.x1 += dx;
        self.y0 += dy;
        self.y1 += dy;
    }

    /// The integer crop this rectangle covers on a `img_w` × `img_h` image:
    /// corners rounded to pixel edges and clamped inside the image. `None`
    /// when nothing (at least 1×1 px) is left.
    pub fn to_crop(self, img_w: usize, img_h: usize) -> Option<CropRect> {
        let r = self.normalized();
        let x0 = (r.x0.round().max(0.0) as usize).min(img_w);
        let y0 = (r.y0.round().max(0.0) as usize).min(img_h);
        let x1 = (r.x1.round().max(0.0) as usize).min(img_w);
        let y1 = (r.y1.round().max(0.0) as usize).min(img_h);
        (x1 > x0 && y1 > y0).then(|| CropRect {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        })
    }
}

/// The saved crop: `x`/`y` is the top-left pixel, `width`/`height` the size in
/// pixels; the region is `[x, x+width) × [y, y+height)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CropRect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl CropRect {
    pub fn full(img_w: usize, img_h: usize) -> CropRect {
        CropRect {
            x: 0,
            y: 0,
            width: img_w,
            height: img_h,
        }
    }

    /// This crop, drawn on frames oriented by `orientation` (oriented image
    /// size `img_w × img_h`), expressed in the on-disk frame of the files.
    pub fn to_disk(self, orientation: Orientation, img_w: usize, img_h: usize) -> CropRect {
        let (x0, y0, x1, y1) =
            orientation.to_disk_rect(self.x, self.y, self.x1(), self.y1(), img_w, img_h);
        CropRect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
    }

    /// A crop given in the on-disk frame of the files (on-disk image size
    /// `disk_w × disk_h`), expressed on the oriented frames.
    pub fn from_disk(self, orientation: Orientation, disk_w: usize, disk_h: usize) -> CropRect {
        let (x0, y0, x1, y1) =
            orientation.from_disk_rect(self.x, self.y, self.x1(), self.y1(), disk_w, disk_h);
        CropRect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
    }

    /// One past the right-most column.
    pub fn x1(&self) -> usize {
        self.x + self.width
    }

    /// One past the bottom-most row.
    pub fn y1(&self) -> usize {
        self.y + self.height
    }

    pub fn area(&self) -> usize {
        self.width * self.height
    }

    pub fn to_rectf(self) -> RectF {
        RectF {
            x0: self.x as f32,
            y0: self.y as f32,
            x1: self.x1() as f32,
            y1: self.y1() as f32,
        }
    }

    /// The part of the crop that lies on a `img_w` × `img_h` image, or `None`
    /// when it falls entirely outside.
    pub fn clamp_to(self, img_w: usize, img_h: usize) -> Option<CropRect> {
        let x1 = self.x1().min(img_w);
        let y1 = self.y1().min(img_h);
        (self.x < x1 && self.y < y1).then(|| CropRect {
            x: self.x,
            y: self.y,
            width: x1 - self.x,
            height: y1 - self.y,
        })
    }

    /// Parse the command-line form `X,Y,WIDTH,HEIGHT` (e.g. `100,200,512,512`).
    pub fn parse_arg(s: &str) -> Result<CropRect, String> {
        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(format!("expected X,Y,WIDTH,HEIGHT (got '{s}')"));
        }
        let mut vals = [0usize; 4];
        for (v, p) in vals.iter_mut().zip(&parts) {
            *v = p
                .parse::<usize>()
                .map_err(|e| format!("invalid value '{p}' in '{s}': {e}"))?;
        }
        let [x, y, width, height] = vals;
        if width == 0 || height == 0 {
            return Err(format!("crop width and height must be > 0 (got '{s}')"));
        }
        Ok(CropRect {
            x,
            y,
            width,
            height,
        })
    }

    /// The JSON written by the save button, for a crop drawn on the oriented
    /// frames (oriented image size `img_w × img_h`).
    ///
    /// `x`/`y`/`width`/`height` and `image_width`/`image_height` are in the
    /// **on-disk** frame of the input files (what every caller slices);
    /// `detector`/`orientation` record how the frames were shown, and the
    /// `display_*` keys repeat the crop as drawn, for information. `folder`
    /// is the input the crop was drawn on.
    pub fn to_json(
        &self,
        img_w: usize,
        img_h: usize,
        folder: &str,
        detector: Detector,
        orientation: Orientation,
    ) -> String {
        let escaped: String = folder
            .chars()
            .flat_map(|c| match c {
                '"' | '\\' => vec!['\\', c],
                _ => vec![c],
            })
            .collect();
        let disk = self.to_disk(orientation, img_w, img_h);
        let (disk_w, disk_h) = orientation.dims(img_w, img_h);
        format!(
            "{{\n  \"x\": {},\n  \"y\": {},\n  \"width\": {},\n  \"height\": {},\n  \
             \"image_width\": {disk_w},\n  \"image_height\": {disk_h},\n  \"folder\": \"{escaped}\",\n  \
             \"detector\": \"{}\",\n  \"orientation\": \"{}\",\n  \
             \"display_x\": {},\n  \"display_y\": {},\n  \"display_width\": {},\n  \"display_height\": {},\n  \
             \"display_image_width\": {img_w},\n  \"display_image_height\": {img_h}\n}}\n",
            disk.x,
            disk.y,
            disk.width,
            disk.height,
            detector.label(),
            orientation.label(),
            self.x,
            self.y,
            self.width,
            self.height,
        )
    }

    /// Read a crop back from JSON text: only the `x`, `y`, `width` and
    /// `height` keys (on-disk frame) are used, so files from other tools work
    /// too.
    pub fn from_json_text(text: &str) -> Result<CropRect, String> {
        let get = |key: &str| -> Result<usize, String> {
            json_uint(text, key).ok_or_else(|| format!("no numeric \"{key}\" field found"))
        };
        let crop = CropRect {
            x: get("x")?,
            y: get("y")?,
            width: get("width")?,
            height: get("height")?,
        };
        if crop.width == 0 || crop.height == 0 {
            return Err("crop width and height must be > 0".to_owned());
        }
        Ok(crop)
    }
}

/// Angles this close to a multiple of 90° snap onto it, so a box rotated
/// back by hand becomes exactly axis-aligned again (pixel-exact export).
const ANGLE_SNAP_DEG: f32 = 0.75;

/// The tilted crop box: a center, a size, and a tilt angle, all in image-pixel
/// space. `angle_deg` is the tilt in degrees, positive counter-clockwise as
/// seen on screen (image y points down). At `angle_deg == 0` the box is the
/// axis-aligned rectangle `[cx - w/2, cx + w/2) × [cy - h/2, cy + h/2)`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OrientedBox {
    pub cx: f32,
    pub cy: f32,
    pub width: f32,
    pub height: f32,
    pub angle_deg: f32,
}

impl OrientedBox {
    pub fn from_rectf(r: RectF) -> OrientedBox {
        let n = r.normalized();
        OrientedBox {
            cx: (n.x0 + n.x1) * 0.5,
            cy: (n.y0 + n.y1) * 0.5,
            width: n.x1 - n.x0,
            height: n.y1 - n.y0,
            angle_deg: 0.0,
        }
    }

    pub fn from_crop(c: CropRect) -> OrientedBox {
        Self::from_rectf(c.to_rectf())
    }

    pub fn full(img_w: usize, img_h: usize) -> OrientedBox {
        Self::from_crop(CropRect::full(img_w, img_h))
    }

    /// Unit vectors along the box's width (u) and height (v), in image space.
    /// Positive angles rotate counter-clockwise on screen (y down).
    pub fn axes(&self) -> ((f32, f32), (f32, f32)) {
        let (s, c) = self.angle_deg.to_radians().sin_cos();
        ((c, -s), (s, c))
    }

    /// Corners in order top-left, top-right, bottom-right, bottom-left (of the
    /// untilted box, rotated about the center).
    pub fn corners(&self) -> [(f32, f32); 4] {
        let (u, v) = self.axes();
        let (hw, hh) = (self.width * 0.5, self.height * 0.5);
        let at = |du: f32, dv: f32| {
            (
                self.cx + u.0 * du + v.0 * dv,
                self.cy + u.1 * du + v.1 * dv,
            )
        };
        [at(-hw, -hh), at(hw, -hh), at(hw, hh), at(-hw, hh)]
    }

    /// A point's coordinates in the box frame: distance from the center along
    /// the width (u) and height (v) axes.
    pub fn to_local(&self, px: f32, py: f32) -> (f32, f32) {
        let (u, v) = self.axes();
        let (dx, dy) = (px - self.cx, py - self.cy);
        (dx * u.0 + dy * u.1, dx * v.0 + dy * v.1)
    }

    /// The image point at box-frame coordinates `(lu, lv)`.
    pub fn from_local(&self, lu: f32, lv: f32) -> (f32, f32) {
        let (u, v) = self.axes();
        (
            self.cx + u.0 * lu + v.0 * lv,
            self.cy + u.1 * lu + v.1 * lv,
        )
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        let (lu, lv) = self.to_local(px, py);
        lu.abs() <= self.width * 0.5 && lv.abs() <= self.height * 0.5
    }

    pub fn translate(&mut self, dx: f32, dy: f32) {
        self.cx += dx;
        self.cy += dy;
    }

    /// Axis-aligned bounding rectangle of the (possibly tilted) box.
    pub fn bounding(&self) -> RectF {
        let cs = self.corners();
        let xs = cs.iter().map(|c| c.0);
        let ys = cs.iter().map(|c| c.1);
        RectF {
            x0: xs.clone().fold(f32::INFINITY, f32::min),
            y0: ys.clone().fold(f32::INFINITY, f32::min),
            x1: xs.fold(f32::NEG_INFINITY, f32::max),
            y1: ys.fold(f32::NEG_INFINITY, f32::max),
        }
    }

    /// Size of the exported images: the box size rounded to whole pixels.
    pub fn out_size(&self) -> (usize, usize) {
        (
            (self.width.round().max(1.0)) as usize,
            (self.height.round().max(1.0)) as usize,
        )
    }

    pub fn area(&self) -> f64 {
        self.width as f64 * self.height as f64
    }

    /// Whether the box is exactly axis-aligned (after [`OrientedBox::snapped`]
    /// the angle is exactly 0 when it is).
    pub fn is_axis_aligned(&self) -> bool {
        self.angle_deg == 0.0
    }

    /// The exact integer crop this box covers when it is axis-aligned and
    /// clamped inside the image; `None` for tilted boxes.
    pub fn as_crop(&self, img_w: usize, img_h: usize) -> Option<CropRect> {
        if !self.is_axis_aligned() {
            return None;
        }
        let (hw, hh) = (self.width * 0.5, self.height * 0.5);
        RectF {
            x0: self.cx - hw,
            y0: self.cy - hh,
            x1: self.cx + hw,
            y1: self.cy + hh,
        }
        .to_crop(img_w, img_h)
    }

    /// Angle folded into `(-180, 180]`, snapped onto a multiple of 90° when
    /// within [`ANGLE_SNAP_DEG`]; ±180° becomes 0 (same rectangle).
    fn canonical_angle(mut a: f32) -> f32 {
        a = a.rem_euclid(360.0);
        if a > 180.0 {
            a -= 360.0;
        }
        for snap in [-180.0, -90.0, 0.0, 90.0, 180.0] {
            if (a - snap).abs() <= ANGLE_SNAP_DEG {
                a = snap;
                break;
            }
        }
        if a == -180.0 || a == 180.0 {
            a = 0.0;
        }
        a
    }

    /// The box as saved after an edit: size rounded to whole pixels (≥ 1),
    /// angle folded and snapped onto exact multiples of 90° when close;
    /// axis-aligned boxes additionally snap onto the pixel grid and clamp
    /// inside the image, exactly like the untilted app. `None` when nothing of
    /// the box lies on the image.
    pub fn snapped(&self, img_w: usize, img_h: usize) -> Option<OrientedBox> {
        let angle = Self::canonical_angle(self.angle_deg);
        if angle == 0.0 {
            let b = OrientedBox {
                angle_deg: 0.0,
                ..*self
            };
            return b.as_crop(img_w, img_h).map(OrientedBox::from_crop);
        }
        let b = OrientedBox {
            cx: (self.cx * 2.0).round() / 2.0, // half-pixel grid: integer sizes stay integer
            cy: (self.cy * 2.0).round() / 2.0,
            width: self.width.abs().round().max(1.0),
            height: self.height.abs().round().max(1.0),
            angle_deg: angle,
        };
        // Keep the box only when its bounding rectangle overlaps the image
        // (a tilted box may legitimately stick out over the borders).
        let bb = b.bounding();
        (bb.x1 > 0.0 && bb.y1 > 0.0 && bb.x0 < img_w as f32 && bb.y0 < img_h as f32).then_some(b)
    }

    /// Parse the command-line form `X,Y,WIDTH,HEIGHT[,ANGLE]`: the top-left
    /// corner and size of the untilted box, plus an optional tilt in degrees
    /// applied about its center.
    pub fn parse_arg(s: &str) -> Result<OrientedBox, String> {
        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() == 4 {
            return CropRect::parse_arg(s).map(OrientedBox::from_crop);
        }
        if parts.len() != 5 {
            return Err(format!("expected X,Y,WIDTH,HEIGHT[,ANGLE] (got '{s}')"));
        }
        let base = CropRect::parse_arg(&parts[..4].join(","))?;
        let angle: f32 = parts[4]
            .parse()
            .map_err(|e| format!("invalid angle '{}' in '{s}': {e}", parts[4]))?;
        let mut b = OrientedBox::from_crop(base);
        b.angle_deg = Self::canonical_angle(angle);
        Ok(b)
    }

    /// The axis-aligned bounding box of the region on the image (clamped
    /// inside it), on the oriented frames; `None` when nothing is on the
    /// image. For an untilted box this is the box itself.
    pub fn bounding_crop(&self, img_w: usize, img_h: usize) -> Option<CropRect> {
        if let Some(c) = self.as_crop(img_w, img_h) {
            return Some(c);
        }
        let bb = self.bounding();
        RectF {
            x0: bb.x0.floor(),
            y0: bb.y0.floor(),
            x1: bb.x1.ceil(),
            y1: bb.y1.ceil(),
        }
        .to_crop(img_w, img_h)
    }

    /// The bounding box in the on-disk frame of the files (what folder names
    /// and the `x`..`height` JSON keys carry).
    pub fn disk_bounds(&self, orientation: Orientation, img_w: usize, img_h: usize) -> CropRect {
        self.bounding_crop(img_w, img_h)
            .unwrap_or(CropRect { x: 0, y: 0, width: 1, height: 1 })
            .to_disk(orientation, img_w, img_h)
    }

    /// The JSON written on save. An untilted box writes exactly the
    /// [`CropRect::to_json`] document (on-disk `x`..`height`); a tilted box
    /// adds `center_x`/`center_y`/`box_width`/`box_height`/`angle_deg` on the
    /// oriented frames (as drawn), with `x`..`height` holding its axis-aligned
    /// bounding box in the on-disk frame so untilted readers still find the
    /// region, and `display_*` the same bounding box as drawn.
    pub fn to_json(
        &self,
        img_w: usize,
        img_h: usize,
        folder: &str,
        detector: Detector,
        orientation: Orientation,
    ) -> String {
        if let Some(c) = self.as_crop(img_w, img_h) {
            return c.to_json(img_w, img_h, folder, detector, orientation);
        }
        let escaped: String = folder
            .chars()
            .flat_map(|c| match c {
                '"' | '\\' => vec!['\\', c],
                _ => vec![c],
            })
            .collect();
        let shown = self
            .bounding_crop(img_w, img_h)
            .unwrap_or(CropRect { x: 0, y: 0, width: 1, height: 1 });
        let disk = shown.to_disk(orientation, img_w, img_h);
        let (disk_w, disk_h) = orientation.dims(img_w, img_h);
        format!(
            "{{\n  \"center_x\": {},\n  \"center_y\": {},\n  \"box_width\": {},\n  \
             \"box_height\": {},\n  \"angle_deg\": {},\n  \"x\": {},\n  \"y\": {},\n  \
             \"width\": {},\n  \"height\": {},\n  \"image_width\": {disk_w},\n  \
             \"image_height\": {disk_h},\n  \"folder\": \"{escaped}\",\n  \
             \"detector\": \"{}\",\n  \"orientation\": \"{}\",\n  \
             \"display_x\": {},\n  \"display_y\": {},\n  \"display_width\": {},\n  \
             \"display_height\": {},\n  \"display_image_width\": {img_w},\n  \
             \"display_image_height\": {img_h}\n}}\n",
            self.cx,
            self.cy,
            self.width,
            self.height,
            self.angle_deg,
            disk.x,
            disk.y,
            disk.width,
            disk.height,
            detector.label(),
            orientation.label(),
            shown.x,
            shown.y,
            shown.width,
            shown.height,
        )
    }

    /// Read a box back from JSON: the tilted keys when present, otherwise the
    /// plain rust_crop_tiff `x`/`y`/`width`/`height` form (angle 0).
    pub fn from_json_text(text: &str) -> Result<OrientedBox, String> {
        if let (Some(cx), Some(cy), Some(w), Some(h)) = (
            json_float(text, "center_x"),
            json_float(text, "center_y"),
            json_float(text, "box_width"),
            json_float(text, "box_height"),
        ) {
            if w <= 0.0 || h <= 0.0 {
                return Err("box width and height must be > 0".to_owned());
            }
            return Ok(OrientedBox {
                cx,
                cy,
                width: w,
                height: h,
                angle_deg: Self::canonical_angle(json_float(text, "angle_deg").unwrap_or(0.0)),
            });
        }
        CropRect::from_json_text(text).map(OrientedBox::from_crop)
    }
}

/// First number following `"key" :` in `text`.
fn json_float(text: &str, key: &str) -> Option<f32> {
    let needle = format!("\"{key}\"");
    let mut search = text;
    loop {
        let at = search.find(&needle)?;
        let rest = &search[at + needle.len()..];
        let rest_trim = rest.trim_start();
        if let Some(after_colon) = rest_trim.strip_prefix(':') {
            let after_colon = after_colon.trim_start();
            let num: String = after_colon
                .chars()
                .take_while(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
                .collect();
            if !num.is_empty() {
                return num.parse().ok();
            }
        }
        search = rest; // e.g. the key appeared inside a string value; keep looking
    }
}

/// First non-negative integer following `"key" :` in `text`.
fn json_uint(text: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let mut search = text;
    loop {
        let at = search.find(&needle)?;
        let rest = &search[at + needle.len()..];
        let rest_trim = rest.trim_start();
        if let Some(after_colon) = rest_trim.strip_prefix(':') {
            let after_colon = after_colon.trim_start();
            let digits: String = after_colon.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                return digits.parse().ok();
            }
        }
        search = rest; // e.g. the key appeared inside a string value; keep looking
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectf_rounds_and_clamps() {
        let r = RectF {
            x0: 10.4,
            y0: -3.0,
            x1: 2.6,
            y1: 7.2,
        };
        // Corners in any order; negative clamped to 0.
        assert_eq!(
            r.to_crop(100, 100),
            Some(CropRect {
                x: 3,
                y: 0,
                width: 7,
                height: 7
            })
        );
        // Entirely outside the image → no crop.
        let out = RectF {
            x0: 200.0,
            y0: 5.0,
            x1: 300.0,
            y1: 10.0,
        };
        assert_eq!(out.to_crop(100, 100), None);
    }

    #[test]
    fn parse_arg_roundtrip() {
        let c = CropRect::parse_arg("100, 200, 512, 256").unwrap();
        assert_eq!(
            c,
            CropRect {
                x: 100,
                y: 200,
                width: 512,
                height: 256
            }
        );
        assert!(CropRect::parse_arg("1,2,3").is_err());
        assert!(CropRect::parse_arg("1,2,0,4").is_err());
    }

    #[test]
    fn json_roundtrip() {
        let c = CropRect {
            x: 5,
            y: 6,
            width: 70,
            height: 80,
        };
        let json = c.to_json(2048, 2048, "/some/\"quoted\"/run_1", Detector::Unknown, Orientation::Identity);
        assert_eq!(CropRect::from_json_text(&json).unwrap(), c);
    }

    #[test]
    fn json_ignores_unknown_and_string_fields() {
        let text = r#"{"folder": "with \"width\" inside", "x": 1, "y": 2,
                       "width": 30, "height": 40, "note": "x: 99"}"#;
        assert_eq!(
            CropRect::from_json_text(text).unwrap(),
            CropRect {
                x: 1,
                y: 2,
                width: 30,
                height: 40
            }
        );
    }

    #[test]
    fn clamp_to_smaller_image() {
        let c = CropRect {
            x: 50,
            y: 50,
            width: 100,
            height: 100,
        };
        assert_eq!(
            c.clamp_to(120, 80),
            Some(CropRect {
                x: 50,
                y: 50,
                width: 70,
                height: 30
            })
        );
        assert_eq!(c.clamp_to(40, 40), None);
    }

    #[test]
    fn oriented_box_json_and_arg_roundtrip() {
        // Untilted: exactly the CropRect document.
        let c = CropRect { x: 5, y: 6, width: 70, height: 80 };
        let b = OrientedBox::from_crop(c);
        let json = b.to_json(200, 100, "f", Detector::Unknown, Orientation::Identity);
        assert_eq!(json, c.to_json(200, 100, "f", Detector::Unknown, Orientation::Identity));
        assert_eq!(OrientedBox::from_json_text(&json).unwrap(), b);
        // Tilted: center / size / angle come back, x..height is the bounding box.
        let t = OrientedBox { cx: 50.0, cy: 40.0, width: 30.0, height: 20.0, angle_deg: 30.0 };
        let json = t.to_json(200, 100, "f", Detector::Unknown, Orientation::Identity);
        let back = OrientedBox::from_json_text(&json).unwrap();
        assert!((back.cx - 50.0).abs() < 1e-4 && (back.angle_deg - 30.0).abs() < 1e-4);
        let bb = CropRect::from_json_text(&json).unwrap();
        assert!(bb.x <= 35 && bb.x1() >= 65 && bb.y <= 25 && bb.y1() >= 55);
        assert!(json.contains("\"angle_deg\": 30"));
        // Command line: 4 values (untilted, disk frame) or 5 (with the tilt).
        assert_eq!(OrientedBox::parse_arg("5,6,70,80").unwrap(), b);
        let a = OrientedBox::parse_arg("5,6,70,80,12.5").unwrap();
        assert!((a.angle_deg - 12.5).abs() < 1e-6 && (a.cx - 40.0).abs() < 1e-6);
        assert!(OrientedBox::parse_arg("1,2,3").is_err());
    }
}
