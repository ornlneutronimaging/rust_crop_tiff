//! Loading one input: a folder of TIFF images, or a `.npy` stack file (2-D =
//! one frame, 3-D = one frame per plane along axis 0 — the form used when
//! another application such as rust_ct_reconstruction hands its stack over).
//!
//! Every frame is normalised to an `Array2<f32>` with shape `(height, width)`,
//! row-major, in the *sample* orientation of the detector that wrote it
//! ([`detector_orientation`]: Timepix TIFFs are transposed, CCD TIFFs flipped
//! vertically; `.npy` stacks are already-processed data and stay as they
//! are). The crop is drawn on these oriented frames; everything written back
//! to disk ([`write_cropped_stack`], [`export_cropped_images`]) is put back in
//! the on-disk frame of the input files. Besides the frames themselves, the loader computes the pixel-wise
//! projections used to judge a crop against the whole stack: sum, mean, max,
//! min and standard deviation, plus the total counts of each frame.

use crate::crop::OrientedBox;
#[cfg(test)]
use crate::crop::CropRect;
use crate::extract;
use anyhow::{anyhow, bail, Context, Result};
pub use detector_orientation::{Detector, Orientation, Selection, Source};
use ndarray::{s, Array2, Array3};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Everything the application needs about one input (folder or `.npy` stack).
pub struct FolderData {
    pub path: PathBuf,
    /// Detector the stack was recorded with (automatic guess + user
    /// override), which decides [`FolderData::orientation`].
    pub detector: Selection,
    /// How every frame was re-oriented on load relative to the file on disk
    /// (always [`Orientation::Identity`] for a `.npy` input).
    pub orientation: Orientation,
    /// One frame per TIFF page, in sorted file order, already oriented.
    pub frames: Vec<Array2<f32>>,
    /// Size of the oriented frames.
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
    /// Per frame: sum of the finite pixels and their count, used by the
    /// crop statistics. Non-finite pixels (NaN/inf, common in normalized
    /// data) are excluded so the statistics stay finite.
    pub frame_totals: Vec<(f64, usize)>,
}

impl FolderData {
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    /// Size of the frames as they are on disk.
    pub fn disk_dims(&self) -> (usize, usize) {
        self.orientation.dims(self.width, self.height)
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

/// How far a load has come. A folder counts its files; a `.npy` stack is one
/// file holding the whole stack, so there the frame count (from the header)
/// says what is being loaded and the bytes read drive the bar.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadProgress {
    pub files_done: usize,
    pub files_total: usize,
    /// Frames being loaded: the file count of a folder, the header's axis 0
    /// for a `.npy` stack.
    pub frames_total: usize,
    /// Bytes of a `.npy` stack read so far (0 / 0 for a folder).
    pub bytes_done: u64,
    pub bytes_total: u64,
    /// The `.npy` file name, empty for a folder.
    pub stack_file: String,
}

impl LoadProgress {
    /// Fraction of the work done: bytes for a stack file, files otherwise.
    pub fn fraction(&self) -> f32 {
        if self.bytes_total > 0 {
            (self.bytes_done as f64 / self.bytes_total as f64).clamp(0.0, 1.0) as f32
        } else if self.files_total > 0 {
            self.files_done as f32 / self.files_total as f32
        } else {
            0.0
        }
    }

    /// The progress bar text: `1385 images from sample_stack.npy — 1.2 GB of
    /// 3.3 GB`, or `240 / 1385 files`.
    pub fn text(&self) -> String {
        if self.stack_file.is_empty() {
            format!("{} / {} files", self.files_done, self.files_total)
        } else {
            format!(
                "{} image{} from {} — {} of {}",
                self.frames_total,
                if self.frames_total == 1 { "" } else { "s" },
                self.stack_file,
                format_bytes(self.bytes_done),
                format_bytes(self.bytes_total),
            )
        }
    }
}

/// Bytes for the progress text: `512 B`, `8.4 MB`, `3.31 GB`.
pub fn format_bytes(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

/// The frame count announced by a `.npy` header (axis 0 of a 3-D array, 1
/// for a 2-D one) without reading the data.
pub fn npy_frame_count(path: &Path) -> Option<usize> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    file.read_exact(&mut head).ok()?;
    if &head[..6] != b"\x93NUMPY" {
        return None;
    }
    let (header_len, offset) = match head[6] {
        1 => (u16::from_le_bytes([head[8], head[9]]) as usize, 10),
        _ => (u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize, 12),
    };
    let mut header = vec![0u8; header_len];
    let already = 12 - offset;
    header[..already].copy_from_slice(&head[offset..]);
    file.read_exact(&mut header[already..]).ok()?;
    let header = String::from_utf8_lossy(&header);
    let shape = header.split("'shape':").nth(1)?;
    let inner = shape.split('(').nth(1)?.split(')').next()?;
    let dims: Vec<usize> = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().ok())
        .collect::<Option<Vec<usize>>>()?;
    match dims.len() {
        2 => Some(1),
        3 => Some(dims[0]),
        _ => None,
    }
}

/// Load one input — a folder of TIFF images or a `.npy` stack file — and
/// compute the projections. `detector` decides how TIFF frames are oriented
/// (a `.npy` stack is loaded as-is). `on_progress` is called from worker
/// threads as files finish (or every few megabytes of a `.npy` stack), so a
/// caller can drive a progress bar.
pub fn load_input_with_progress<F>(
    path: &Path,
    detector: Selection,
    on_progress: F,
) -> Result<FolderData>
where
    F: Fn(&LoadProgress) + Sync,
{
    if path.is_dir() {
        return load_folder_with_progress(path, detector, on_progress);
    }
    let mut progress = LoadProgress {
        files_total: 1,
        frames_total: npy_frame_count(path).unwrap_or(1),
        bytes_total: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        stack_file: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ..LoadProgress::default()
    };
    on_progress(&progress);
    let frames = load_npy(path, &mut |read| {
        progress.bytes_done = read;
        on_progress(&progress);
    })?;
    progress.files_done = 1;
    progress.frames_total = frames.len();
    progress.bytes_done = progress.bytes_total;
    on_progress(&progress);
    build_folder_data(path, frames, detector, Orientation::Identity)
}

/// Load every TIFF of `dir` (in parallel) and compute the projections.
/// `on_progress` is called from worker threads as files finish, so a caller
/// can drive a progress bar.
pub fn load_folder_with_progress<F>(
    dir: &Path,
    detector: Selection,
    on_progress: F,
) -> Result<FolderData>
where
    F: Fn(&LoadProgress) + Sync,
{
    let orientation = detector.orientation();
    let paths = list_tiff_in_dir(dir)?;
    let total = paths.len();
    let done = AtomicUsize::new(0);
    let report = |files_done: usize| {
        on_progress(&LoadProgress {
            files_done,
            files_total: total,
            frames_total: total,
            ..LoadProgress::default()
        })
    };
    report(0);

    // `par_iter().map().collect()` preserves the sorted file order.
    let per_file: Vec<Result<Vec<Array2<f32>>>> = paths
        .par_iter()
        .map(|p| {
            let r = load_tiff(p, orientation);
            report(done.fetch_add(1, Ordering::Relaxed) + 1);
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
    build_folder_data(dir, frames, detector, orientation)
}

/// Compute the projections and per-frame totals of `frames` (all of the same
/// size, already oriented) and assemble the [`FolderData`].
fn build_folder_data(
    path: &Path,
    frames: Vec<Array2<f32>>,
    detector: Selection,
    orientation: Orientation,
) -> Result<FolderData> {
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

    let frame_totals: Vec<(f64, usize)> = frames
        .par_iter()
        .map(|f| {
            let (mut sum, mut n) = (0.0f64, 0usize);
            for &v in f.iter() {
                if v.is_finite() {
                    sum += v as f64;
                    n += 1;
                }
            }
            (sum, n)
        })
        .collect();

    Ok(FolderData {
        path: path.to_path_buf(),
        detector,
        orientation,
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
/// `on_read(bytes_so_far)` is called every few megabytes — the stack handed
/// over by NeCTAR is one multi-gigabyte file on a network filesystem.
fn load_npy(path: &Path, on_read: &mut dyn FnMut(u64)) -> Result<Vec<Array2<f32>>> {
    use ndarray_npy::ReadNpyExt;
    use std::io::{Cursor, Read};

    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let size = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
    let mut bytes: Vec<u8> = Vec::with_capacity(size);
    const CHUNK: usize = 8 << 20;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        on_read(bytes.len() as u64);
    }
    drop(buf);

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

/// One frame's region, straightened when tilted, and put back in the
/// on-disk frame of the input files: for an untilted box the same pixels as
/// slicing the file on disk with the box's [`CropRect::to_disk`] counterpart.
pub(crate) fn extract_to_disk(
    frame: &Array2<f32>,
    region: &OrientedBox,
    orientation: Orientation,
) -> Array2<f32> {
    orientation.undo(extract::extract(frame, region))
}

/// The cropped stack as one contiguous `float32` array of shape
/// `(n_frames, disk_height, disk_width)` — the crop in the on-disk frame of
/// the input files, what `--output-stack` returns to the calling
/// application. A tilted region comes out straightened (resampled on its
/// own grid). `orientation` is how `frames` were oriented on load.
pub fn cropped_stack(
    frames: &[Array2<f32>],
    region: &OrientedBox,
    orientation: Orientation,
) -> Array3<f32> {
    let (out_w, out_h) = region.out_size();
    let (dw, dh) = orientation.dims(out_w, out_h);
    let extracted: Vec<Array2<f32>> = frames
        .par_iter()
        .map(|f| extract_to_disk(f, region, orientation))
        .collect();
    let mut out = Array3::<f32>::zeros((frames.len(), dh, dw));
    for (i, e) in extracted.iter().enumerate() {
        out.slice_mut(s![i, .., ..]).assign(e);
    }
    out
}

/// Write the cropped stack to `path` as a NumPy `.npy` file (on-disk frame,
/// see [`cropped_stack`]).
pub fn write_cropped_stack(
    path: &Path,
    frames: &[Array2<f32>],
    region: &OrientedBox,
    orientation: Orientation,
) -> Result<()> {
    let stack = cropped_stack(frames, region, orientation);
    ndarray_npy::write_npy(path, &stack)
        .with_context(|| format!("write cropped stack {}", path.display()))
}

/// Apply `crop` (drawn on the oriented frames) to every frame and write them
/// as individual float32 TIFF files into `dest` (created if needed), in the
/// on-disk frame of the input files. When the input was a folder whose
/// TIFF files map 1:1 onto the frames, the original file names are kept so
/// downstream tools and the spectra file still line up; otherwise the frames
/// are numbered after the input's stem. Any `*_Spectra.txt` file next to the
/// input is copied along. Returns status notes for the UI.
pub fn export_cropped_images(
    dest: &Path,
    data: &FolderData,
    region: &OrientedBox,
) -> Result<Vec<String>> {
    if data.path.is_dir() && dest == data.path {
        bail!("Export folder must be different from the input folder");
    }
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;

    // Keep the original file names when they map 1:1 onto the frames.
    let from_files: Option<Vec<String>> = if data.path.is_dir() {
        list_tiff_in_dir(&data.path).ok().and_then(|paths| {
            (paths.len() == data.frames.len()).then(|| {
                paths
                    .iter()
                    .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
                    .collect()
            })
        })
    } else {
        None
    };
    let names: Vec<String> = from_files.unwrap_or_else(|| {
        let stem = data
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "cropped".to_owned());
        (0..data.frames.len())
            .map(|i| format!("{stem}_{i:05}.tiff"))
            .collect()
    });

    data.frames
        .par_iter()
        .zip(names.par_iter())
        .try_for_each(|(frame, name)| -> Result<()> {
            let path = dest.join(name);
            let cropped = extract_to_disk(frame, region, data.orientation);
            let (dh, dw) = (cropped.nrows(), cropped.ncols());
            let values: Vec<f32> = cropped.iter().copied().collect();
            let file = std::fs::File::create(&path)
                .with_context(|| format!("create {}", path.display()))?;
            let mut enc = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))
                .with_context(|| format!("encode TIFF {}", path.display()))?;
            enc.write_image::<tiff::encoder::colortype::Gray32Float>(
                dw as u32,
                dh as u32,
                &values,
            )
            .with_context(|| format!("write {}", path.display()))?;
            Ok(())
        })?;

    let (out_w, out_h) = region.out_size();
    let (dw, dh) = data.orientation.dims(out_w, out_h);
    let mut notes = vec![format!(
        "{} cropped images ({dw}×{dh} px on disk, {}) → {}",
        data.frames.len(),
        data.orientation,
        dest.display()
    )];

    // Copy the spectra file(s) of a folder input along with the images.
    if data.path.is_dir() {
        let mut copied = 0usize;
        for entry in std::fs::read_dir(&data.path)? {
            let path = entry?.path();
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            if path.is_file() && name.ends_with("_spectra.txt") {
                let to = dest.join(path.file_name().unwrap_or_default());
                std::fs::copy(&path, &to)
                    .with_context(|| format!("copy {} to {}", path.display(), to.display()))?;
                copied += 1;
                notes.push(format!("spectra file → {}", to.display()));
            }
        }
        if copied == 0 {
            notes.push("no *_Spectra.txt file found to copy".to_owned());
        }
    }

    Ok(notes)
}

/// Read every page of a (possibly multi-page) TIFF file, oriented.
pub(crate) fn load_tiff(path: &Path, orientation: Orientation) -> Result<Vec<Array2<f32>>> {
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

        out.push(orientation.apply(to_frame(values, w, h)?));

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

        let data = load_folder_with_progress(&dir, Selection::default(), |_| {}).unwrap();
        assert_eq!(data.n_frames(), 2);
        assert_eq!(data.orientation, Orientation::Identity);
        assert_eq!((data.width, data.height), (4, 3));
        // Sorted order: img_00000 (5) first.
        assert_eq!(data.frames[0][(0, 0)], 5.0);
        assert_eq!(data.sum[(2, 3)], 12.0);
        assert_eq!(data.mean[(0, 0)], 6.0);
        assert_eq!(data.max[(1, 1)], 7.0);
        assert_eq!(data.min[(1, 1)], 5.0);
        assert!((data.std[(0, 0)] - 1.0).abs() < 1e-6, "std of {{5,7}} is 1");
        assert_eq!(data.frame_totals, vec![(60.0, 12), (84.0, 12)]);
    }

    #[test]
    fn folder_without_tiff_fails() {
        let dir = tmp_dir("empty");
        assert!(load_folder_with_progress(&dir, Selection::default(), |_| {}).is_err());
    }

    #[test]
    fn npy_stack_progress_counts_images_and_bytes() {
        let dir = tmp_dir("npy_progress");
        let path = dir.join("sample_stack.npy");
        let a = Array3::<f32>::zeros((9, 4, 5));
        ndarray_npy::write_npy(&path, &a).unwrap();
        assert_eq!(npy_frame_count(&path), Some(9));
        let flat = dir.join("one.npy");
        ndarray_npy::write_npy(&flat, &Array2::<u16>::zeros((4, 5))).unwrap();
        assert_eq!(npy_frame_count(&flat), Some(1));

        let seen = std::sync::Mutex::new(Vec::<LoadProgress>::new());
        let data = load_input_with_progress(&path, Selection::default(), |p| {
            seen.lock().unwrap().push(p.clone())
        })
        .unwrap();
        assert_eq!(data.n_frames(), 9);
        let seen = seen.into_inner().unwrap();
        let first = seen.first().unwrap();
        assert_eq!((first.frames_total, first.files_total, first.files_done), (9, 1, 0));
        assert_eq!(first.bytes_total, std::fs::metadata(&path).unwrap().len());
        assert_eq!(first.stack_file, "sample_stack.npy");
        let last = seen.last().unwrap();
        assert_eq!(last.files_done, 1);
        assert_eq!(last.bytes_done, last.bytes_total);
        assert_eq!(last.fraction(), 1.0);
        assert!(last.text().starts_with("9 images from sample_stack.npy — "), "{}", last.text());
        assert_eq!(format_bytes(3_310_000_000), "3.31 GB");
        // A folder still counts files.
        let folder = LoadProgress {
            files_done: 240,
            files_total: 1385,
            frames_total: 1385,
            ..LoadProgress::default()
        };
        assert_eq!(folder.text(), "240 / 1385 files");
        let _ = std::fs::remove_dir_all(&dir);
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
        // a .npy stack is never re-oriented, whatever the detector says
        let timepix = Selection { manual: Some(Detector::Timepix), ..Default::default() };
        let data = load_input_with_progress(&path, timepix, |_| {}).unwrap();
        assert_eq!(data.n_frames(), 2);
        assert_eq!(data.orientation, Orientation::Identity);
        assert_eq!((data.width, data.height), (4, 3));
        assert_eq!(data.frames[1][(2, 3)], 123.0);
        assert_eq!(data.sum[(0, 1)], 1.0 + 101.0);
    }

    #[test]
    fn export_writes_cropped_tiffs_and_copies_spectra() {
        let dir = tmp_dir("export_src");
        write_tiff_u16(&dir.join("img_00000.tif"), 6, 5, 5);
        write_tiff_u16(&dir.join("img_00001.tif"), 6, 5, 7);
        std::fs::write(dir.join("run_Spectra.txt"), "tof,counts\n").unwrap();

        let data = load_folder_with_progress(&dir, Selection::default(), |_| {}).unwrap();
        let crop = CropRect {
            x: 1,
            y: 2,
            width: 3,
            height: 2,
        };
        let dest = tmp_dir("export_dst");
        let notes = export_cropped_images(&dest, &data, &OrientedBox::from_crop(crop)).unwrap();
        assert!(notes.iter().any(|n| n.contains("2 cropped images")));
        assert!(notes.iter().any(|n| n.contains("spectra file")));

        // Original names kept, spectra copied.
        assert!(dest.join("run_Spectra.txt").is_file());
        let frames = load_tiff(&dest.join("img_00001.tif"), Orientation::Identity).unwrap();
        assert_eq!(frames[0].shape(), &[2, 3]);
        assert_eq!(frames[0][(0, 0)], 7.0);

        // Exporting onto the input folder must be refused.
        assert!(export_cropped_images(&dir, &data, &OrientedBox::from_crop(crop)).is_err());
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
        write_cropped_stack(&path, &frames, &OrientedBox::from_crop(crop), Orientation::Identity).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let back = Array3::<f32>::read_npy(file).unwrap();
        assert_eq!(back.shape(), &[2, 2, 3]);
        // frame 1, crop-local (0, 0) is full-image (y=2, x=1).
        assert_eq!(back[(1, 0, 0)], 121.0);
        assert_eq!(back[(0, 1, 2)], 33.0);
    }

    /// A distinct-valued 6 wide × 5 tall TIFF: value = 10*y + x.
    fn write_tiff_pattern(path: &Path) {
        use tiff::encoder::{colortype, TiffEncoder};
        let file = std::fs::File::create(path).unwrap();
        let mut enc = TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
        let data: Vec<u16> = (0..5).flat_map(|y| (0..6).map(move |x| (10 * y + x) as u16)).collect();
        enc.write_image::<colortype::Gray16>(6, 5, &data).unwrap();
    }

    /// Cropping an oriented stack (Timepix transpose, CCD flip) writes exactly
    /// the pixels that slicing the on-disk file with the crop's on-disk
    /// counterpart gives — for both the `.npy` stack and the TIFF export.
    #[test]
    fn oriented_crops_write_disk_pixels() {
        let src = tmp_dir("oriented_src");
        write_tiff_pattern(&src.join("img_00000.tif"));
        let disk = load_folder_with_progress(&src, Selection::default(), |_| {}).unwrap();
        assert_eq!((disk.width, disk.height), (6, 5));

        for det in [Detector::Timepix, Detector::Ccd, Detector::Qhy] {
            let sel = Selection { manual: Some(det), ..Default::default() };
            let data = load_folder_with_progress(&src, sel, |_| {}).unwrap();
            let o = data.orientation;
            assert_eq!(o, det.orientation());
            assert_eq!(data.disk_dims(), (6, 5));
            // a crop drawn on the oriented image
            let crop = CropRect { x: 1, y: 2, width: 2, height: 3 }
                .clamp_to(data.width, data.height)
                .unwrap();
            let disk_crop = crop.to_disk(o, data.width, data.height);
            assert_eq!(disk_crop.from_disk(o, 6, 5), crop, "{det:?} round trip");
            let expected: Vec<f32> = (disk_crop.y..disk_crop.y1())
                .flat_map(|y| (disk_crop.x..disk_crop.x1()).map(move |x| (10 * y + x) as f32))
                .collect();

            let stack = cropped_stack(&data.frames, &OrientedBox::from_crop(crop), o);
            assert_eq!(stack.shape(), &[1, disk_crop.height, disk_crop.width], "{det:?}");
            assert_eq!(stack.iter().copied().collect::<Vec<_>>(), expected, "{det:?} stack");

            let dest = tmp_dir(&format!("oriented_dst_{det:?}"));
            export_cropped_images(&dest, &data, &OrientedBox::from_crop(crop)).unwrap();
            let frames = load_tiff(&dest.join("img_00000.tif"), Orientation::Identity).unwrap();
            assert_eq!(frames[0].shape(), &[disk_crop.height, disk_crop.width], "{det:?}");
            assert_eq!(frames[0].iter().copied().collect::<Vec<_>>(), expected, "{det:?} tiff");
        }
    }

    #[test]
    fn crop_json_is_in_disk_frame() {
        // 6 wide × 5 tall on disk; transposed it is 5 wide × 6 tall.
        let crop = CropRect { x: 1, y: 2, width: 2, height: 3 };
        let json = crop.to_json(5, 6, "/data/images/tpx1/run", Detector::Timepix, Orientation::Transpose);
        let back = CropRect::from_json_text(&json).unwrap();
        assert_eq!(back, CropRect { x: 2, y: 1, width: 3, height: 2 });
        assert!(json.contains("\"image_width\": 6"));
        assert!(json.contains("\"image_height\": 5"));
        assert!(json.contains("\"detector\": \"Timepix\""));
        assert!(json.contains("\"display_x\": 1"));
        // identity: unchanged
        let json = crop.to_json(6, 5, "x", Detector::Unknown, Orientation::Identity);
        assert_eq!(CropRect::from_json_text(&json).unwrap(), crop);
    }
}
