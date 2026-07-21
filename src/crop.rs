//! The crop region: a single, pixel-aligned rectangle on the image, plus the
//! JSON format used to hand it to other applications (e.g. a marimo notebook
//! that applies the crop to the full stack before processing it).
//!
//! Two representations: [`RectF`] is the free-floating rectangle being drawn
//! or dragged on screen; [`CropRect`] is the integer region actually saved and
//! used for statistics, always clamped inside the image.

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

    /// The JSON written by the save button. `image_width`/`image_height` and
    /// `folder` record what the crop was drawn on.
    pub fn to_json(&self, img_w: usize, img_h: usize, folder: &str) -> String {
        let escaped: String = folder
            .chars()
            .flat_map(|c| match c {
                '"' | '\\' => vec!['\\', c],
                _ => vec![c],
            })
            .collect();
        format!(
            "{{\n  \"x\": {},\n  \"y\": {},\n  \"width\": {},\n  \"height\": {},\n  \
             \"image_width\": {img_w},\n  \"image_height\": {img_h},\n  \"folder\": \"{escaped}\"\n}}\n",
            self.x, self.y, self.width, self.height
        )
    }

    /// Read a crop back from JSON text: only the `x`, `y`, `width` and
    /// `height` keys are used, so files from other tools work too.
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
        let json = c.to_json(2048, 2048, "/some/\"quoted\"/run_1");
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
}
