//! Per-frame statistics of the crop, used to check that no image of the stack
//! loses important content:
//!
//! - **edge band** — mean counts in a thin band just inside the crop edge. If
//!   the sample never touches the crop boundary this curve stays flat at the
//!   open-beam level; a dip (transmission) or spike (bright feature) in some
//!   frame means the crop edge cuts through the sample in that frame.
//! - **inside / outside** — mean counts inside and outside the crop, to see
//!   what the crop keeps and what it throws away, frame by frame.

use crate::crop::CropRect;
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

/// Compute the per-frame crop statistics. `frame_totals[i]` must be the total
/// counts of `frames[i]` (precomputed at load time).
pub fn compute(
    frames: &[Array2<f32>],
    frame_totals: &[f64],
    crop: CropRect,
    band: usize,
) -> CropStats {
    let (inside_area, outside_area) = match frames.first() {
        Some(f) => {
            let total_px = f.len();
            (crop.area(), total_px - crop.area())
        }
        None => {
            return CropStats {
                band,
                edge_mean: Vec::new(),
                inside_mean: Vec::new(),
                outside_mean: Vec::new(),
            }
        }
    };

    // Region strictly inside the edge band; empty when the crop is too small
    // for the band, in which case the whole crop is the band.
    let inner = (crop.width > 2 * band && crop.height > 2 * band).then(|| CropRect {
        x: crop.x + band,
        y: crop.y + band,
        width: crop.width - 2 * band,
        height: crop.height - 2 * band,
    });

    let per_frame: Vec<(f64, f64, f64)> = frames
        .par_iter()
        .zip(frame_totals)
        .map(|(f, &total)| {
            let sum_of = |r: CropRect| -> f64 {
                f.slice(s![r.y..r.y1(), r.x..r.x1()])
                    .iter()
                    .map(|&v| v as f64)
                    .sum()
            };
            let inside_sum = sum_of(crop);
            let (edge_sum, edge_area) = match inner {
                Some(inner) => (inside_sum - sum_of(inner), inside_area - inner.area()),
                None => (inside_sum, inside_area),
            };
            let inside_mean = inside_sum / inside_area as f64;
            let edge_mean = edge_sum / edge_area as f64;
            let outside_mean = if outside_area > 0 {
                (total - inside_sum) / outside_area as f64
            } else {
                f64::NAN
            };
            (edge_mean, inside_mean, outside_mean)
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

    fn totals(frames: &[Array2<f32>]) -> Vec<f64> {
        frames
            .iter()
            .map(|f| f.iter().map(|&v| v as f64).sum())
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
