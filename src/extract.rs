//! Pull an [`OrientedBox`] out of a frame: the cookie-cutter core. The output
//! is always a plain `height × width` image (the box size rounded to whole
//! pixels), whatever the tilt — a tilted box is resampled bilinearly on the
//! box's own grid, so its content appears straightened in the output.
//!
//! Output pixels whose sample point falls outside the source image are NaN
//! (a tilted box may legitimately stick out over the borders); the statistics
//! and the viewer already treat NaN as "no data".

use crate::crop::OrientedBox;
use ndarray::{s, Array2};

/// Extract the box from one frame. Axis-aligned boxes on the pixel grid are
/// copied exactly (no resampling); anything else is sampled bilinearly at the
/// center of every output pixel.
pub fn extract(frame: &Array2<f32>, b: &OrientedBox) -> Array2<f32> {
    let (img_h, img_w) = (frame.shape()[0], frame.shape()[1]);
    let (out_w, out_h) = b.out_size();

    // Exact fast path: axis-aligned, pixel-aligned, fully inside the image.
    if b.is_axis_aligned() {
        let x0 = b.cx - b.width * 0.5;
        let y0 = b.cy - b.height * 0.5;
        let on_grid = x0.fract() == 0.0
            && y0.fract() == 0.0
            && b.width.fract() == 0.0
            && b.height.fract() == 0.0;
        if on_grid
            && x0 >= 0.0
            && y0 >= 0.0
            && x0 + b.width <= img_w as f32
            && y0 + b.height <= img_h as f32
        {
            let (x0, y0) = (x0 as usize, y0 as usize);
            return frame.slice(s![y0..y0 + out_h, x0..x0 + out_w]).to_owned();
        }
    }

    // General path: walk the box's own grid. Sample positions step by the
    // real box size over the rounded output size, so a fractional box is
    // covered evenly.
    let (u, v) = b.axes();
    let [top_left, _, _, _] = b.corners();
    let (du, dv) = (b.width / out_w as f32, b.height / out_h as f32);

    Array2::from_shape_fn((out_h, out_w), |(i, j)| {
        let lu = (j as f32 + 0.5) * du;
        let lv = (i as f32 + 0.5) * dv;
        let x = top_left.0 + u.0 * lu + v.0 * lv;
        let y = top_left.1 + u.1 * lu + v.1 * lv;
        bilinear(frame, x, y, img_w, img_h)
    })
}

/// Bilinear sample at the image point `(x, y)` (pixel (r, c) covers
/// `[c, c+1) × [r, r+1)`, its center is `(c+0.5, r+0.5)`). Points outside the
/// image yield NaN; points between the border pixel centers and the edge
/// clamp onto the border pixels.
fn bilinear(frame: &Array2<f32>, x: f32, y: f32, img_w: usize, img_h: usize) -> f32 {
    if x < 0.0 || y < 0.0 || x > img_w as f32 || y > img_h as f32 {
        return f32::NAN;
    }
    // Center-based coordinates, clamped so border samples stay inside.
    let gx = (x - 0.5).clamp(0.0, (img_w - 1) as f32);
    let gy = (y - 0.5).clamp(0.0, (img_h - 1) as f32);
    let x0 = gx.floor() as usize;
    let y0 = gy.floor() as usize;
    let x1 = (x0 + 1).min(img_w - 1);
    let y1 = (y0 + 1).min(img_h - 1);
    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;
    let a = frame[(y0, x0)] * (1.0 - fx) + frame[(y0, x1)] * fx;
    let b = frame[(y1, x0)] * (1.0 - fx) + frame[(y1, x1)] * fx;
    a * (1.0 - fy) + b * fy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crop::CropRect;

    fn ramp(w: usize, h: usize) -> Array2<f32> {
        Array2::from_shape_fn((h, w), |(r, c)| (r * w + c) as f32)
    }

    #[test]
    fn axis_aligned_matches_slice() {
        let f = ramp(10, 8);
        let b = OrientedBox::from_crop(CropRect {
            x: 2,
            y: 1,
            width: 5,
            height: 4,
        });
        let e = extract(&f, &b);
        assert_eq!(e, f.slice(s![1..5, 2..7]).to_owned());
    }

    #[test]
    fn ninety_degrees_rotates_content() {
        // A 4×2 box rotated +90° over a ramp: the output columns must run
        // along the image's y axis.
        let f = ramp(8, 8);
        let b = OrientedBox {
            cx: 4.0,
            cy: 4.0,
            width: 4.0,
            height: 2.0,
            angle_deg: 90.0,
        };
        let e = extract(&f, &b);
        assert_eq!(e.shape(), &[2, 4]);
        // +90° (CCW on screen): the box's u axis points up (−y), so along an
        // output row the sampled image y decreases → values decrease by one
        // image row (8.0) per output column.
        let row: Vec<f32> = e.row(0).to_vec();
        for w in row.windows(2) {
            assert!((w[0] - w[1] - 8.0).abs() < 1e-3, "row: {row:?}");
        }
    }

    #[test]
    fn tilted_uniform_frame_is_uniform_and_finite() {
        let f = Array2::from_elem((20, 20), 3.5f32);
        let b = OrientedBox {
            cx: 10.0,
            cy: 10.0,
            width: 8.0,
            height: 5.0,
            angle_deg: 25.0,
        };
        let e = extract(&f, &b);
        assert_eq!(e.shape(), &[5, 8]);
        for &v in e.iter() {
            assert!((v - 3.5).abs() < 1e-4);
        }
    }

    #[test]
    fn outside_image_is_nan() {
        let f = ramp(10, 10);
        // Half of the box hangs off the left edge of the image.
        let b = OrientedBox {
            cx: 0.0,
            cy: 5.0,
            width: 6.0,
            height: 4.0,
            angle_deg: 10.0,
        };
        let e = extract(&f, &b);
        assert!(e.iter().any(|v| v.is_nan()));
        assert!(e.iter().any(|v| v.is_finite()));
    }
}
