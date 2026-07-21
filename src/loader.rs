//! Loading one folder of TIFF images.
//!
//! Every frame is normalised to an `Array2<f32>` with shape `(height, width)`,
//! row-major. Besides the frames themselves, the loader computes the pixel-wise
//! projections used to judge a crop against the whole stack: sum, mean, max,
//! min and standard deviation, plus the total counts of each frame.

use anyhow::{anyhow, bail, Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Everything the application needs about one folder.
pub struct FolderData {
    pub path: PathBuf,
    /// One frame per TIFF page, in sorted file order.
    pub frames: Vec<Array2<f32>>,
    pub width: usize,
    pub height: usize,
    /// Pixel-wise sum of every frame (the "integrated" image).
    pub sum: Array2<f32>,
    /// Pixel-wise mean of every frame.
    pub mean: Array2<f32>,
    /// Brightest value each pixel ever takes.
    pub max: Array2<f32>,
    /// Darkest value each pixel ever takes. For transmission images this is
    /// the union of the sample silhouettes over all frames.
    pub min: Array2<f32>,
    /// Pixel-wise standard deviation across the frames: moving edges and
    /// changing regions stand out.
    pub std: Array2<f32>,
    /// Total counts of each frame, used by the crop statistics.
    pub frame_totals: Vec<f64>,
}

impl FolderData {
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }
}

/// Every TIFF file directly inside `dir`, sorted by name.
pub fn list_tiff_in_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let path = entry?.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if path.is_file() && (ext == "tif" || ext == "tiff") {
            out.push(path);
        }
    }
    if out.is_empty() {
        bail!("No TIFF files found in {}", dir.display());
    }
    out.sort();
    Ok(out)
}

/// Per-pixel accumulator for the projections, merged across worker threads.
struct Acc {
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    max: Vec<f32>,
    min: Vec<f32>,
}

impl Acc {
    fn new(len: usize) -> Acc {
        Acc {
            sum: vec![0.0; len],
            sumsq: vec![0.0; len],
            max: vec![f32::NEG_INFINITY; len],
            min: vec![f32::INFINITY; len],
        }
    }

    fn add_frame(&mut self, frame: &[f32]) {
        for (i, &v) in frame.iter().enumerate() {
            let vd = v as f64;
            self.sum[i] += vd;
            self.sumsq[i] += vd * vd;
            self.max[i] = self.max[i].max(v);
            self.min[i] = self.min[i].min(v);
        }
    }

    fn merge(mut self, other: Acc) -> Acc {
        for i in 0..self.sum.len() {
            self.sum[i] += other.sum[i];
            self.sumsq[i] += other.sumsq[i];
            self.max[i] = self.max[i].max(other.max[i]);
            self.min[i] = self.min[i].min(other.min[i]);
        }
        self
    }
}

/// Load every TIFF of `dir` (in parallel) and compute the projections.
/// `on_progress(files_done, files_total)` is called from worker threads as
/// files finish, so a caller can drive a progress bar.
pub fn load_folder_with_progress<F>(dir: &Path, on_progress: F) -> Result<FolderData>
where
    F: Fn(usize, usize) + Sync,
{
    let paths = list_tiff_in_dir(dir)?;
    let total = paths.len();
    let done = AtomicUsize::new(0);

    // `par_iter().map().collect()` preserves the sorted file order.
    let per_file: Vec<Result<Vec<Array2<f32>>>> = paths
        .par_iter()
        .map(|p| {
            let r = load_tiff(p);
            on_progress(done.fetch_add(1, Ordering::Relaxed) + 1, total);
            r
        })
        .collect();

    let mut frames: Vec<Array2<f32>> = Vec::with_capacity(total);
    let mut dims: Option<(usize, usize)> = None;
    for (path, loaded) in paths.iter().zip(per_file) {
        for frame in loaded? {
            let (h, w) = (frame.shape()[0], frame.shape()[1]);
            match dims {
                None => dims = Some((h, w)),
                Some((dh, dw)) if (dh, dw) != (h, w) => bail!(
                    "Frame size mismatch: {w}x{h} in {} does not match {dw}x{dh}",
                    path.display()
                ),
                _ => {}
            }
            frames.push(frame);
        }
    }
    let (height, width) = dims.ok_or_else(|| anyhow!("No frames were loaded"))?;
    let len = width * height;
    let n = frames.len() as f64;

    let acc = frames
        .par_iter()
        .fold(
            || Acc::new(len),
            |mut acc, f| {
                acc.add_frame(f.as_slice().expect("frames are standard layout"));
                acc
            },
        )
        .reduce(|| Acc::new(len), Acc::merge);

    let shape = (height, width);
    let mut mean = vec![0f32; len];
    let mut std = vec![0f32; len];
    let mut sum32 = vec![0f32; len];
    for i in 0..len {
        let m = acc.sum[i] / n;
        mean[i] = m as f32;
        sum32[i] = acc.sum[i] as f32;
        std[i] = (acc.sumsq[i] / n - m * m).max(0.0).sqrt() as f32;
    }

    let frame_totals: Vec<f64> = frames
        .par_iter()
        .map(|f| f.iter().map(|&v| v as f64).sum())
        .collect();

    Ok(FolderData {
        path: dir.to_path_buf(),
        frames,
        width,
        height,
        sum: Array2::from_shape_vec(shape, sum32)?,
        mean: Array2::from_shape_vec(shape, mean)?,
        max: Array2::from_shape_vec(shape, acc.max)?,
        min: Array2::from_shape_vec(shape, acc.min)?,
        std: Array2::from_shape_vec(shape, std)?,
        frame_totals,
    })
}

/// Read every page of a (possibly multi-page) TIFF file.
fn load_tiff(path: &Path) -> Result<Vec<Array2<f32>>> {
    use tiff::decoder::{Decoder, DecodingResult};

    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("decode TIFF {}", path.display()))?;

    let mut out = Vec::new();
    loop {
        let (w, h) = decoder.dimensions()?;
        let (w, h) = (w as usize, h as usize);

        let data = decoder.read_image()?;
        let values: Vec<f32> = match data {
            DecodingResult::U8(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::U16(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::U32(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::U64(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I8(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I16(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I32(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I64(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
            DecodingResult::F32(v) => v,
            DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        };

        out.push(to_frame(values, w, h)?);

        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }

    Ok(out)
}

/// Turn a flat, row-major buffer into an `(h, w)` array. If the buffer carries
/// several samples per pixel (e.g. RGB TIFF) only the first sample is kept.
fn to_frame(values: Vec<f32>, w: usize, h: usize) -> Result<Array2<f32>> {
    let expected = w * h;
    if values.len() == expected {
        return Ok(Array2::from_shape_vec((h, w), values)?);
    }
    if expected > 0 && values.len() % expected == 0 {
        let spp = values.len() / expected;
        let first: Vec<f32> = (0..expected).map(|i| values[i * spp]).collect();
        return Ok(Array2::from_shape_vec((h, w), first)?);
    }
    bail!("Pixel count {} is not compatible with {w}x{h}", values.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_crop_tiff_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tiff_u16(path: &Path, w: usize, h: usize, value: u16) {
        use tiff::encoder::{colortype, TiffEncoder};
        let file = std::fs::File::create(path).unwrap();
        let mut enc = TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
        let data = vec![value; w * h];
        enc.write_image::<colortype::Gray16>(w as u32, h as u32, &data)
            .unwrap();
    }

    #[test]
    fn folder_load_computes_projections() {
        let dir = tmp_dir("folder");
        write_tiff_u16(&dir.join("img_00001.tif"), 4, 3, 7);
        write_tiff_u16(&dir.join("img_00000.tif"), 4, 3, 5);

        let data = load_folder_with_progress(&dir, |_, _| {}).unwrap();
        assert_eq!(data.n_frames(), 2);
        assert_eq!((data.width, data.height), (4, 3));
        // Sorted order: img_00000 (5) first.
        assert_eq!(data.frames[0][(0, 0)], 5.0);
        assert_eq!(data.sum[(2, 3)], 12.0);
        assert_eq!(data.mean[(0, 0)], 6.0);
        assert_eq!(data.max[(1, 1)], 7.0);
        assert_eq!(data.min[(1, 1)], 5.0);
        assert!((data.std[(0, 0)] - 1.0).abs() < 1e-6, "std of {{5,7}} is 1");
        assert_eq!(data.frame_totals, vec![60.0, 84.0]);
    }

    #[test]
    fn folder_without_tiff_fails() {
        let dir = tmp_dir("empty");
        assert!(load_folder_with_progress(&dir, |_, _| {}).is_err());
    }
}
