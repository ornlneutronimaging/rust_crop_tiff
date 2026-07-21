//! Loading one input: a folder of TIFF images, or a `.npy` stack file (2-D =
//! one frame, 3-D = one frame per plane along axis 0 — the form used when
//! another application such as rust_ct_reconstruction hands its stack over).
//!
//! Every frame is normalised to an `Array2<f32>` with shape `(height, width)`,
//! row-major. Besides the frames themselves, the loader computes the pixel-wise
//! projections used to judge a crop against the whole stack: sum, mean, max,
//! min and standard deviation, plus the total counts of each frame.

use crate::crop::CropRect;
use anyhow::{anyhow, bail, Context, Result};
use ndarray::{s, Array2, Array3};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Everything the application needs about one input (folder or `.npy` stack).
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

/// Whether `path` is something [`load_input_with_progress`] can open: a
/// folder (of TIFF images) or a `.npy` stack file.
pub fn is_supported_input(path: &Path) -> bool {
    path.is_dir()
        || (path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("npy")))
}

/// Load one input — a folder of TIFF images or a `.npy` stack file — and
/// compute the projections. `on_progress(files_done, files_total)` is called
/// from worker threads as files finish, so a caller can drive a progress bar.
pub fn load_input_with_progress<F>(path: &Path, on_progress: F) -> Result<FolderData>
where
    F: Fn(usize, usize) + Sync,
{
    if path.is_dir() {
        return load_folder_with_progress(path, on_progress);
    }
    let frames = load_npy(path)?;
    on_progress(1, 1);
    build_folder_data(path, frames)
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
    build_folder_data(dir, frames)
}

/// Compute the projections and per-frame totals of `frames` (all of the same
/// size) and assemble the [`FolderData`].
fn build_folder_data(path: &Path, frames: Vec<Array2<f32>>) -> Result<FolderData> {
    let (height, width) = match frames.first() {
        Some(f) => (f.shape()[0], f.shape()[1]),
        None => return Err(anyhow!("No frames were loaded")),
    };
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
        path: path.to_path_buf(),
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

/// Read a `.npy` file: a 2-D array becomes one frame, a 3-D array one frame
/// per plane along axis 0. The dtype is probed among the common numeric types.
fn load_npy(path: &Path) -> Result<Vec<Array2<f32>>> {
    use ndarray_npy::ReadNpyExt;
    use std::io::Cursor;

    let bytes = std::fs::read(path).with_context(|| format!("open {}", path.display()))?;

    macro_rules! try_2d {
        ($t:ty) => {
            if let Ok(a) = Array2::<$t>::read_npy(Cursor::new(&bytes[..])) {
                return Ok(vec![a.mapv(|v| v as f32)]);
            }
        };
    }
    macro_rules! try_3d {
        ($t:ty) => {
            if let Ok(a) = Array3::<$t>::read_npy(Cursor::new(&bytes[..])) {
                return Ok(a
                    .outer_iter()
                    .map(|plane| plane.mapv(|v| v as f32))
                    .collect());
            }
        };
    }

    try_2d!(f32);
    try_2d!(f64);
    try_2d!(u8);
    try_2d!(u16);
    try_2d!(i16);
    try_2d!(u32);
    try_2d!(i32);
    try_2d!(u64);
    try_2d!(i64);
    try_3d!(f32);
    try_3d!(f64);
    try_3d!(u8);
    try_3d!(u16);
    try_3d!(i16);
    try_3d!(u32);
    try_3d!(i32);
    try_3d!(u64);
    try_3d!(i64);

    bail!(
        "Unsupported .npy dtype or shape (need a 2-D or 3-D numeric array): {}",
        path.display()
    )
}

/// The cropped stack as one contiguous `float32` array of shape
/// `(n_frames, crop.height, crop.width)` — what `--output-stack` returns to
/// the calling application.
pub fn cropped_stack(frames: &[Array2<f32>], crop: CropRect) -> Array3<f32> {
    let mut out = Array3::<f32>::zeros((frames.len(), crop.height, crop.width));
    for (i, f) in frames.iter().enumerate() {
        out.slice_mut(s![i, .., ..])
            .assign(&f.slice(s![crop.y..crop.y1(), crop.x..crop.x1()]));
    }
    out
}

/// Write the cropped stack to `path` as a NumPy `.npy` file.
pub fn write_cropped_stack(path: &Path, frames: &[Array2<f32>], crop: CropRect) -> Result<()> {
    let stack = cropped_stack(frames, crop);
    ndarray_npy::write_npy(path, &stack)
        .with_context(|| format!("write cropped stack {}", path.display()))
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

    #[test]
    fn npy_3d_stack_loads_as_frames() {
        let dir = tmp_dir("npy3d");
        let path = dir.join("stack.npy");
        // 2 frames of 3x4, values = 100*frame + 10*y + x.
        let a = Array3::<u16>::from_shape_fn((2, 3, 4), |(k, y, x)| {
            (100 * k + 10 * y + x) as u16
        });
        ndarray_npy::write_npy(&path, &a).unwrap();

        assert!(is_supported_input(&path));
        let data = load_input_with_progress(&path, |_, _| {}).unwrap();
        assert_eq!(data.n_frames(), 2);
        assert_eq!((data.width, data.height), (4, 3));
        assert_eq!(data.frames[1][(2, 3)], 123.0);
        assert_eq!(data.sum[(0, 1)], 1.0 + 101.0);
    }

    #[test]
    fn cropped_stack_roundtrips_through_npy() {
        use ndarray_npy::ReadNpyExt;

        let dir = tmp_dir("cropped");
        let path = dir.join("cropped.npy");
        let frames: Vec<Array2<f32>> = (0..2)
            .map(|k| Array2::from_shape_fn((5, 6), |(y, x)| (100 * k + 10 * y + x) as f32))
            .collect();
        let crop = CropRect {
            x: 1,
            y: 2,
            width: 3,
            height: 2,
        };
        write_cropped_stack(&path, &frames, crop).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let back = Array3::<f32>::read_npy(file).unwrap();
        assert_eq!(back.shape(), &[2, 2, 3]);
        // frame 1, crop-local (0, 0) is full-image (y=2, x=1).
        assert_eq!(back[(1, 0, 0)], 121.0);
        assert_eq!(back[(0, 1, 2)], 33.0);
    }
}
