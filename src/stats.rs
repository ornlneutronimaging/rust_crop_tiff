//! Per-frame statistics of the crop, used to check that no image of the stack
//! loses important content:
//!
//! - **edge band** — mean counts in a thin band just inside the crop edge. If
//!   the sample never touches the crop boundary this curve stays flat at the
//!   open-beam level; a dip (transmission) or spike (bright feature) in some
//!   frame means the crop edge cuts through the sample in that frame.
//! - **inside / outside** — mean counts inside and outside the crop, to see
//!   what the crop keeps and what it throws away, frame by frame.

use crate::crop::{CropRect, OrientedBox};
use crate::extract;
use ndarray::{s, Array2};
use rayon::prelude::*;

pub struct CropStats {
    /// Band width (pixels) the edge statistics were computed with.
    pub band: usize,
    /// Mean counts in the band just inside the crop edge, per frame.
    pub edge_mean: Vec<f64>,
    /// Mean counts inside the crop, per frame.
    pub inside_mean: Vec<f64>,
    /// Mean counts outside the crop, per frame (NaN when the crop covers the
    /// whole image).
    pub outside_mean: Vec<f64>,
}

impl CropStats {
    /// Frame whose edge-band mean deviates the most from the median edge-band
    /// mean — the most suspicious image to inspect, whether the feature
    /// crossing the edge is dark (transmission) or bright.
    pub fn most_suspicious_frame(&self) -> Option<usize> {
        let mut sorted: Vec<f64> = self.edge_mean.iter().copied().filter(|v| v.is_finite()).collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
        self.edge_mean
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(_, a), (_, b)| (*a - median).abs().total_cmp(&(*b - median).abs()))
            .map(|(i, _)| i)
    }
}

/// Compute the per-frame crop statistics. `frame_totals[i]` must be the sum
/// and count of the finite pixels of `frames[i]` (precomputed at load time).
///
/// Non-finite pixels (NaN/inf, common in normalized data) are excluded from
/// every sum and every mean divides by the count of finite pixels, so a few
/// bad pixels cannot turn the whole curve NaN (which would blank the plot).
/// A region with no finite pixel yields NaN and is skipped by the plot.
pub fn compute(
    frames: &[Array2<f32>],
    frame_totals: &[(f64, usize)],
    crop: CropRect,
    band: usize,
) -> CropStats {
    if frames.is_empty() {
        return CropStats {
            band,
            edge_mean: Vec::new(),
            inside_mean: Vec::new(),
            outside_mean: Vec::new(),
        };
    }

    // Region strictly inside the edge band; empty when the crop is too small
    // for the band, in which case the whole crop is the band.
    let inner = (crop.width > 2 * band && crop.height > 2 * band).then(|| CropRect {
        x: crop.x + band,
        y: crop.y + band,
        width: crop.width - 2 * band,
        height: crop.height - 2 * band,
    });

    let mean = |sum: f64, n: usize| if n > 0 { sum / n as f64 } else { f64::NAN };
    let per_frame: Vec<(f64, f64, f64)> = frames
        .par_iter()
        .zip(frame_totals)
        .map(|(f, &(total_sum, total_n))| {
            // Sum and count of the finite pixels of a region.
            let sum_of = |r: CropRect| -> (f64, usize) {
                let (mut sum, mut n) = (0.0f64, 0usize);
                for &v in f.slice(s![r.y..r.y1(), r.x..r.x1()]).iter() {
                    if v.is_finite() {
                        sum += v as f64;
                        n += 1;
                    }
                }
                (sum, n)
            };
            let (inside_sum, inside_n) = sum_of(crop);
            let (edge_sum, edge_n) = match inner {
                Some(inner) => {
                    let (inner_sum, inner_n) = sum_of(inner);
                    (inside_sum - inner_sum, inside_n - inner_n)
                }
                None => (inside_sum, inside_n),
            };
            let outside_mean = mean(total_sum - inside_sum, total_n - inside_n);
            (mean(edge_sum, edge_n), mean(inside_sum, inside_n), outside_mean)
        })
        .collect();

    CropStats {
        band,
        edge_mean: per_frame.iter().map(|t| t.0).collect(),
        inside_mean: per_frame.iter().map(|t| t.1).collect(),
        outside_mean: per_frame.iter().map(|t| t.2).collect(),
    }
}

/// Per-frame statistics of a (possibly tilted) box. Axis-aligned boxes use
/// the exact pixel path below; tilted boxes are extracted bilinearly per
/// frame, the edge band is taken on the extracted (straightened) image, and
/// the outside mean is what the frame total leaves over. NaN samples where
/// the box hangs off the image are excluded like any non-finite pixel.
pub fn compute_oriented(
    frames: &[Array2<f32>],
    frame_totals: &[(f64, usize)],
    obox: &OrientedBox,
    band: usize,
) -> CropStats {
    let (img_h, img_w) = match frames.first() {
        Some(f) => (f.shape()[0], f.shape()[1]),
        None => {
            return CropStats {
                band,
                edge_mean: Vec::new(),
                inside_mean: Vec::new(),
                outside_mean: Vec::new(),
            }
        }
    };
    if let Some(crop) = obox.as_crop(img_w, img_h) {
        return compute(frames, frame_totals, crop, band);
    }

    let (out_w, out_h) = obox.out_size();
    let inner = (out_w > 2 * band && out_h > 2 * band).then(|| CropRect {
        x: band,
        y: band,
        width: out_w - 2 * band,
        height: out_h - 2 * band,
    });
    let mean = |sum: f64, n: usize| if n > 0 { sum / n as f64 } else { f64::NAN };
    let per_frame: Vec<(f64, f64, f64)> = frames
        .par_iter()
        .zip(frame_totals)
        .map(|(f, &(total_sum, total_n))| {
            let ex = extract::extract(f, obox);
            let sum_of = |r: CropRect| -> (f64, usize) {
                let (mut sum, mut n) = (0.0f64, 0usize);
                for &v in ex.slice(s![r.y..r.y1(), r.x..r.x1()]).iter() {
                    if v.is_finite() {
                        sum += v as f64;
                        n += 1;
                    }
                }
                (sum, n)
            };
            let (inside_sum, inside_n) = sum_of(CropRect::full(out_w, out_h));
            let (edge_sum, edge_n) = match inner {
                Some(inner) => {
                    let (inner_sum, inner_n) = sum_of(inner);
                    (inside_sum - inner_sum, inside_n - inner_n)
                }
                None => (inside_sum, inside_n),
            };
            // The extracted samples are interpolated, not a subset of the
            // original pixels, so the outside mean is an estimate here.
            let outside_mean = mean(total_sum - inside_sum, total_n.saturating_sub(inside_n));
            (mean(edge_sum, edge_n), mean(inside_sum, inside_n), outside_mean)
        })
        .collect();

    CropStats {
        band,
        edge_mean: per_frame.iter().map(|t| t.0).collect(),
        inside_mean: per_frame.iter().map(|t| t.1).collect(),
        outside_mean: per_frame.iter().map(|t| t.2).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_with(w: usize, h: usize, base: f32, spot: Option<(usize, usize, f32)>) -> Array2<f32> {
        let mut f = Array2::from_elem((h, w), base);
        if let Some((x, y, v)) = spot {
            f[(y, x)] = v;
        }
        f
    }

    fn totals(frames: &[Array2<f32>]) -> Vec<(f64, usize)> {
        frames
            .iter()
            .map(|f| {
                let finite: Vec<f64> =
                    f.iter().filter(|v| v.is_finite()).map(|&v| v as f64).collect();
                (finite.iter().sum(), finite.len())
            })
            .collect()
    }

    #[test]
    fn uniform_frame_has_equal_means() {
        let frames = vec![frame_with(10, 10, 2.0, None)];
        let crop = CropRect {
            x: 2,
            y: 2,
            width: 6,
            height: 6,
        };
        let s = compute(&frames, &totals(&frames), crop, 1);
        assert_eq!(s.inside_mean, vec![2.0]);
        assert_eq!(s.outside_mean, vec![2.0]);
        assert_eq!(s.edge_mean, vec![2.0]);
    }

    #[test]
    fn spot_on_the_edge_band_shows_up() {
        // Two uniform frames; frame 1 has a bright pixel on the crop's edge band.
        let frames = vec![
            frame_with(10, 10, 1.0, None),
            frame_with(10, 10, 1.0, Some((2, 5, 101.0))),
            frame_with(10, 10, 1.0, None),
        ];
        let crop = CropRect {
            x: 2,
            y: 2,
            width: 6,
            height: 6,
        };
        let s = compute(&frames, &totals(&frames), crop, 1);
        assert_eq!(s.edge_mean[0], 1.0);
        // Band area: 36 - 16 = 20 px; one of them is +100.
        assert!((s.edge_mean[1] - (1.0 + 100.0 / 20.0)).abs() < 1e-9);
        assert_eq!(s.most_suspicious_frame(), Some(1));
    }

    #[test]
    fn non_finite_pixels_are_ignored() {
        // A NaN inside the crop and an inf outside must not turn the means
        // NaN (that blanked the right-hand plot on normalized data).
        let mut f = frame_with(10, 10, 2.0, None);
        f[(5, 5)] = f32::NAN;
        f[(0, 0)] = f32::INFINITY;
        let frames = vec![f];
        let crop = CropRect {
            x: 2,
            y: 2,
            width: 6,
            height: 6,
        };
        let s = compute(&frames, &totals(&frames), crop, 1);
        assert_eq!(s.inside_mean, vec![2.0]);
        assert_eq!(s.outside_mean, vec![2.0]);
        assert_eq!(s.edge_mean, vec![2.0]);
    }

    #[test]
    fn full_image_crop_has_no_outside() {
        let frames = vec![frame_with(4, 4, 3.0, None)];
        let s = compute(&frames, &totals(&frames), CropRect::full(4, 4), 1);
        assert!(s.outside_mean[0].is_nan());
        assert_eq!(s.inside_mean, vec![3.0]);
    }

    #[test]
    fn band_wider_than_crop_uses_whole_crop() {
        let frames = vec![frame_with(10, 10, 5.0, None)];
        let crop = CropRect {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        };
        let s = compute(&frames, &totals(&frames), crop, 10);
        assert_eq!(s.edge_mean, vec![5.0]);
    }
}
