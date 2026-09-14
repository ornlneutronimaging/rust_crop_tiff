//! Batch crop: apply the crop drawn in the main window to many other folders
//! at once, in a separate window.
//!
//! The folders the user picks are usually *sample* folders whose TIFF images
//! sit one or more levels further down (e.g. `<sample>/<Run_NNNN>/…`), and the
//! names of those intermediate folders differ from sample to sample. So the
//! user picks the data sub-folder on the tree of the **first** folder, and
//! the same relative path is resolved in every other folder level by level:
//! a sub-folder with the same name is taken when there is one, otherwise the
//! *only* sub-folder at that level. The resolved data folder is shown for
//! every input before anything runs.
//!
//! Every resolved folder is then loaded file by file (in parallel), cropped
//! with the crop as drawn (on the detector-oriented frames — the frames of
//! every folder are oriented the same way, by the detector recognized from
//! its path), and written as float32 TIFF files keeping the original names,
//! under `<output>/cropped_x0…_<folder name>/<relative path>` — the same
//! folder naming as the single export, and the same sub-structure as the
//! input — with the `*_Spectra.txt` file copied along and a
//! `crop_region.json` sidecar. An optional preview shows a thumbnail of every
//! folder with the crop drawn on it before running.

use crate::colormap::Colormap;
use crate::crop::CropRect;
use crate::loader::{self, Detector, Orientation, Selection};
use anyhow::{bail, Context, Result};
use egui::{Color32, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Files sampled for a preview thumbnail (evenly spaced through the folder).
const PREVIEW_FILES: usize = 8;
/// Longest side of a preview thumbnail texture, in pixels.
const THUMB_MAX: usize = 256;
/// Thumbnail size on screen.
const THUMB_VIEW: f32 = 200.0;
/// Enlarged preview size on screen.
const ENLARGED_VIEW: f32 = 560.0;
/// How deep the sub-folder tree goes.
const TREE_MAX_DEPTH: usize = 8;

const CROP_COLOR: Color32 = Color32::from_rgb(0, 220, 160);
const BAD_COLOR: Color32 = Color32::from_rgb(255, 90, 80);

// ---------------------------------------------------------------------------
// What the main window hands over
// ---------------------------------------------------------------------------

/// The crop and the stack it was drawn on, as the main window has them.
#[derive(Clone)]
pub struct BatchRef {
    /// The crop as drawn, on the oriented frames.
    pub crop: CropRect,
    /// Oriented size of the reference stack.
    pub width: usize,
    pub height: usize,
    pub orientation: Orientation,
    /// The input the crop was drawn on.
    pub source: PathBuf,
    /// The toolbar's detector override (`None` = recognize from the path).
    pub detector_override: Option<Detector>,
    pub colormap: Colormap,
}

impl BatchRef {
    fn detector_for(&self, path: &Path) -> Selection {
        let mut sel = Selection::from_path(path);
        sel.manual = self.detector_override;
        sel
    }

    /// The crop in the on-disk frame of the reference stack (what the output
    /// folder names carry, like the single export).
    fn disk_crop(&self) -> CropRect {
        self.crop.to_disk(self.orientation, self.width, self.height)
    }
}

/// Name of the folder an export creates: the crop bounds in the on-disk
/// frame (exclusive stops) and the input's name.
pub fn export_folder_name(disk: CropRect, stem: &str) -> String {
    format!(
        "cropped_x0{}_y0{}_x1{}_y1{}_{stem}",
        disk.x,
        disk.y,
        disk.x1(),
        disk.y1()
    )
}

// ---------------------------------------------------------------------------
// Folder tree and sub-folder resolution
// ---------------------------------------------------------------------------

/// One sub-folder of a directory, with what it holds.
#[derive(Clone, Debug)]
pub struct DirInfo {
    pub name: String,
    pub n_tiff: usize,
    pub n_subdirs: usize,
}

fn is_tiff(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
}

/// Number of TIFF files and of sub-folders directly inside `dir`.
pub fn count_dir(dir: &Path) -> (usize, usize) {
    let (mut tiffs, mut subdirs) = (0, 0);
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                subdirs += 1;
            } else if is_tiff(&path) {
                tiffs += 1;
            }
        }
    }
    (tiffs, subdirs)
}

/// The sub-folders of `dir`, sorted by name, each with its own counts.
pub fn list_subdirs(dir: &Path) -> Vec<DirInfo> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let (n_tiff, n_subdirs) = count_dir(&path);
                out.push(DirInfo {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    n_tiff,
                    n_subdirs,
                });
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// A data folder found under one of the picked folders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub path: PathBuf,
    /// The path relative to the picked folder, as actually resolved.
    pub rel: Vec<String>,
    pub n_tiff: usize,
    /// Levels where the name did not exist and the only sub-folder was taken.
    pub substitutions: Vec<String>,
}

impl Resolved {
    pub fn rel_label(&self) -> String {
        if self.rel.is_empty() {
            ".".to_owned()
        } else {
            self.rel.join("/")
        }
    }
}

/// Follow `rel` down from `top`, level by level: a sub-folder with the same
/// name when there is one, otherwise the only sub-folder at that level.
pub fn resolve(top: &Path, rel: &[String]) -> Result<Resolved, String> {
    if !top.is_dir() {
        return Err(format!("{} is not a folder", top.display()));
    }
    let mut path = top.to_path_buf();
    let mut actual = Vec::new();
    let mut substitutions = Vec::new();
    for name in rel {
        let candidate = path.join(name);
        if candidate.is_dir() {
            path = candidate;
            actual.push(name.clone());
            continue;
        }
        let subdirs = list_subdirs(&path);
        match subdirs.len() {
            1 => {
                let only = &subdirs[0].name;
                substitutions.push(format!("{name} → {only}"));
                path = path.join(only);
                actual.push(only.clone());
            }
            0 => {
                return Err(format!(
                    "no sub-folder '{name}' in {} (it has no sub-folders)",
                    path.display()
                ))
            }
            n => {
                return Err(format!(
                    "no sub-folder '{name}' in {} and {n} sub-folders to choose from",
                    path.display()
                ))
            }
        }
    }
    let (n_tiff, _) = count_dir(&path);
    if n_tiff == 0 {
        return Err(format!("no TIFF files in {}", path.display()));
    }
    Ok(Resolved {
        path,
        rel: actual,
        n_tiff,
        substitutions,
    })
}

/// Guess the data sub-folder of `top`: go down while the folder holds no
/// TIFF file and has exactly one sub-folder.
pub fn auto_find(top: &Path) -> Vec<String> {
    let mut path = top.to_path_buf();
    let mut rel = Vec::new();
    for _ in 0..TREE_MAX_DEPTH {
        let (n_tiff, _) = count_dir(&path);
        if n_tiff > 0 {
            break;
        }
        let subdirs = list_subdirs(&path);
        if subdirs.len() != 1 {
            break;
        }
        rel.push(subdirs[0].name.clone());
        path = path.join(&subdirs[0].name);
    }
    rel
}

/// `a/b/c` → `["a", "b", "c"]`; `.` and empty components are dropped.
pub fn parse_rel(text: &str) -> Vec<String> {
    text.split(['/', '\\'])
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != ".")
        .map(str::to_owned)
        .collect()
}

fn rel_text(rel: &[String]) -> String {
    if rel.is_empty() {
        ".".to_owned()
    } else {
        rel.join("/")
    }
}

fn folder_label(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

// ---------------------------------------------------------------------------
// Preview thumbnails
// ---------------------------------------------------------------------------

/// A folder's thumbnail, ready for the GPU.
pub struct PreviewImage {
    /// Oriented size of the folder's frames.
    pub width: usize,
    pub height: usize,
    pub n_files: usize,
    pub sampled: usize,
    pub detector: Selection,
    pub image: egui::ColorImage,
}

/// Mean of up to [`PREVIEW_FILES`] evenly spaced files of `dir`, oriented,
/// reduced to a thumbnail and colored with `colormap` at the auto contrast.
pub fn preview_image(dir: &Path, detector: Selection, colormap: Colormap) -> Result<PreviewImage> {
    let paths = loader::list_tiff_in_dir(dir)?;
    let n_files = paths.len();
    let take = PREVIEW_FILES.min(n_files);
    let picked: Vec<&PathBuf> = (0..take)
        .map(|i| &paths[i * n_files / take])
        .collect();
    let orientation = detector.orientation();
    let frames: Vec<Array2<f32>> = picked
        .par_iter()
        .map(|p| loader::load_tiff(p, orientation))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    let Some(first) = frames.first() else {
        bail!("no frame in {}", dir.display());
    };
    let (h, w) = (first.nrows(), first.ncols());
    if frames.iter().any(|f| f.nrows() != h || f.ncols() != w) {
        bail!("frame sizes differ within {}", dir.display());
    }
    let mut mean = Array2::<f32>::zeros((h, w));
    for f in &frames {
        mean += f;
    }
    mean /= frames.len() as f32;

    let full = crate::app::normalize_range(crate::app::raw_range(mean.iter().copied()));
    let (vmin, vmax) = crate::app::robust_range(std::slice::from_ref(&mean), full);
    let image = thumbnail(&mean, vmin, vmax, colormap);
    Ok(PreviewImage {
        width: w,
        height: h,
        n_files,
        sampled: frames.len(),
        detector,
        image,
    })
}

/// Block-average `img` down to at most [`THUMB_MAX`] pixels on its longest
/// side and color it.
fn thumbnail(img: &Array2<f32>, vmin: f32, vmax: f32, colormap: Colormap) -> egui::ColorImage {
    let (h, w) = (img.nrows(), img.ncols());
    let k = w.max(h).div_ceil(THUMB_MAX).max(1);
    let (tw, th) = ((w / k).max(1), (h / k).max(1));
    let span = (vmax - vmin).max(1e-12);
    let lut = colormap.lut();
    let mut buf = vec![0u8; tw * th * 4];
    for ty in 0..th {
        for tx in 0..tw {
            let (mut sum, mut n) = (0.0f64, 0usize);
            for y in ty * k..((ty + 1) * k).min(h) {
                for x in tx * k..((tx + 1) * k).min(w) {
                    let v = img[(y, x)];
                    if v.is_finite() {
                        sum += v as f64;
                        n += 1;
                    }
                }
            }
            let t = if n > 0 {
                (((sum / n as f64) as f32 - vmin) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let [r, g, b] = lut[((t * 255.0).round() as usize).min(255)];
            let i = (ty * tw + tx) * 4;
            buf[i] = r;
            buf[i + 1] = g;
            buf[i + 2] = b;
            buf[i + 3] = 255;
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([tw, th], &buf)
}

// ---------------------------------------------------------------------------
// Export of one folder
// ---------------------------------------------------------------------------

/// What the export of one folder produced.
#[derive(Clone, Debug)]
pub struct FolderReport {
    pub n_files: usize,
    pub dest: PathBuf,
    pub notes: Vec<String>,
}

/// Crop every TIFF file of `src` with `crop` (drawn on the frames as oriented
/// by `detector`) and write it under the same name into `dest`, as float32
/// TIFF in the on-disk frame, file by file in parallel. The spectra file is
/// copied along and a `crop_region.json` dropped in `dest`. `cancel` stops
/// the export between files; `on_progress(files_done, files_total)` is
/// called from worker threads.
pub fn export_folder<F>(
    src: &Path,
    dest: &Path,
    detector: Selection,
    crop: CropRect,
    cancel: &AtomicBool,
    on_progress: F,
) -> Result<FolderReport>
where
    F: Fn(usize, usize) + Sync,
{
    if dest == src {
        bail!("the output folder must be different from the input folder");
    }
    let orientation = detector.orientation();
    let paths = loader::list_tiff_in_dir(src)?;
    let total = paths.len();
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    let done = AtomicUsize::new(0);

    // Oriented frame size of every file, to check the crop fits and to
    // record the size in the crop JSON.
    let dims: Vec<(usize, usize)> = paths
        .par_iter()
        .map(|p| -> Result<(usize, usize)> {
            if cancel.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            let frames = loader::load_tiff(p, orientation)?;
            let mut dims = (0, 0);
            let name = p.file_name().unwrap_or_default();
            let out_path = dest.join(name);
            let file = std::fs::File::create(&out_path)
                .with_context(|| format!("create {}", out_path.display()))?;
            let mut enc = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))
                .with_context(|| format!("encode TIFF {}", out_path.display()))?;
            for frame in &frames {
                let (h, w) = (frame.nrows(), frame.ncols());
                dims = (w, h);
                if crop.x1() > w || crop.y1() > h {
                    bail!(
                        "the crop ({}×{} at {}, {}) does not fit the {w}×{h} images of {}",
                        crop.width,
                        crop.height,
                        crop.x,
                        crop.y,
                        src.display()
                    );
                }
                let cropped = loader::crop_to_disk(frame, crop, orientation);
                let (dh, dw) = (cropped.nrows(), cropped.ncols());
                let values: Vec<f32> = cropped.iter().copied().collect();
                enc.write_image::<tiff::encoder::colortype::Gray32Float>(
                    dw as u32,
                    dh as u32,
                    &values,
                )
                .with_context(|| format!("write {}", out_path.display()))?;
            }
            on_progress(done.fetch_add(1, Ordering::Relaxed) + 1, total);
            Ok(dims)
        })
        .collect::<Result<Vec<_>>>()?;

    let (w, h) = dims.first().copied().unwrap_or((0, 0));
    let mut notes = Vec::new();
    if let Some(&(ow, oh)) = dims.iter().find(|&&d| d != (w, h)) {
        notes.push(format!("image sizes differ within the folder ({w}×{h} and {ow}×{oh})"));
    }

    let json = crop.to_json(w, h, &src.display().to_string(), detector.detector(), orientation);
    let json_path = dest.join("crop_region.json");
    std::fs::write(&json_path, &json)
        .with_context(|| format!("write {}", json_path.display()))?;

    let mut copied = 0usize;
    for entry in std::fs::read_dir(src)?.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if path.is_file() && name.ends_with("_spectra.txt") {
            let to = dest.join(entry.file_name());
            std::fs::copy(&path, &to)
                .with_context(|| format!("copy {} to {}", path.display(), to.display()))?;
            copied += 1;
        }
    }
    if copied == 0 {
        notes.push("no *_Spectra.txt file to copy".to_owned());
    }

    Ok(FolderReport {
        n_files: total,
        dest: dest.to_path_buf(),
        notes,
    })
}

// ---------------------------------------------------------------------------
// The batch window
// ---------------------------------------------------------------------------

/// Progress of one folder of a running (or finished) batch.
#[derive(Clone, Debug)]
enum RowState {
    Pending,
    Running { done: usize, total: usize },
    Done(FolderReport),
    Failed(String),
    Skipped(String),
}

enum JobMsg {
    Progress { idx: usize, done: usize, total: usize },
    Done { idx: usize, result: Result<FolderReport, String> },
    Finished,
}

struct Job {
    rx: Receiver<JobMsg>,
    cancel: Arc<AtomicBool>,
    finished: bool,
}

/// One loaded preview.
struct Preview {
    tex: TextureHandle,
    width: usize,
    height: usize,
    n_files: usize,
    sampled: usize,
    detector: Selection,
}

/// A preview load in flight; `Err` is the failure message.
type PreviewMsg = (usize, Result<PreviewImage, String>);

pub struct BatchWindow {
    pub open: bool,
    /// The crop and stack of the main window, refreshed while no batch runs.
    reference: Option<BatchRef>,
    folders: Vec<PathBuf>,
    /// The data sub-folder chosen on the first folder's tree.
    rel: Vec<String>,
    rel_text: String,
    tree_cache: HashMap<PathBuf, Vec<DirInfo>>,
    show_tree: bool,
    resolved: Vec<Result<Resolved, String>>,
    output: Option<PathBuf>,

    show_previews: bool,
    previews: Vec<Option<Result<Preview, String>>>,
    preview_rx: Option<Receiver<PreviewMsg>>,
    enlarged: Option<usize>,

    job: Option<Job>,
    rows: Vec<RowState>,
    /// The crop the running/finished batch used (for the summary line).
    job_ref: Option<BatchRef>,
    status: String,
}

impl Default for BatchWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchWindow {
    pub fn new() -> Self {
        Self {
            open: false,
            reference: None,
            folders: Vec::new(),
            rel: Vec::new(),
            rel_text: ".".to_owned(),
            tree_cache: HashMap::new(),
            show_tree: true,
            resolved: Vec::new(),
            output: None,
            show_previews: false,
            previews: Vec::new(),
            preview_rx: None,
            enlarged: None,
            job: None,
            rows: Vec::new(),
            job_ref: None,
            status: String::new(),
        }
    }

    pub fn is_running(&self) -> bool {
        self.job.as_ref().is_some_and(|j| !j.finished)
    }

    /// Open the window; `reference` is the main window's crop.
    pub fn show(&mut self, reference: Option<BatchRef>) {
        self.open = true;
        self.set_reference(reference);
    }

    /// The main window's current crop, taken while no batch runs.
    pub fn set_reference(&mut self, reference: Option<BatchRef>) {
        if self.is_running() {
            return;
        }
        if let Some(r) = reference {
            self.reference = Some(r);
        }
    }

    // ----- background jobs ----------------------------------------------

    /// Poll the running batch and preview loads; call every frame, whether
    /// the window is shown or not, so a batch closed out of sight finishes.
    pub fn poll(&mut self, ctx: &egui::Context) {
        if let Some(job) = &mut self.job {
            loop {
                match job.rx.try_recv() {
                    Ok(JobMsg::Progress { idx, done, total }) => {
                        if let Some(row) = self.rows.get_mut(idx) {
                            *row = RowState::Running { done, total };
                        }
                    }
                    Ok(JobMsg::Done { idx, result }) => {
                        if let Some(row) = self.rows.get_mut(idx) {
                            *row = match result {
                                Ok(r) => RowState::Done(r),
                                Err(e) => RowState::Failed(e),
                            };
                        }
                    }
                    Ok(JobMsg::Finished) => job.finished = true,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        job.finished = true;
                        break;
                    }
                }
            }
            if job.finished {
                let n_ok = self.rows.iter().filter(|r| matches!(r, RowState::Done(_))).count();
                let n_fail = self.rows.iter().filter(|r| matches!(r, RowState::Failed(_))).count();
                let n_skip = self.rows.iter().filter(|r| matches!(r, RowState::Skipped(_))).count();
                let cancelled = job.cancel.load(Ordering::Relaxed);
                let mut parts = vec![format!("{n_ok} folder(s) exported")];
                if n_fail > 0 {
                    parts.push(format!("{n_fail} failed"));
                }
                if n_skip > 0 {
                    parts.push(format!("{n_skip} skipped"));
                }
                let out = self
                    .output
                    .as_ref()
                    .map(|p| format!(" → {}", p.display()))
                    .unwrap_or_default();
                self.status = format!(
                    "{}: {}{out}",
                    if cancelled { "Batch cancelled" } else { "Batch done" },
                    parts.join(", ")
                );
            } else {
                ctx.request_repaint_after(Duration::from_millis(100));
            }
        }

        if let Some(rx) = &self.preview_rx {
            loop {
                match rx.try_recv() {
                    Ok((idx, result)) => {
                        let entry = match result {
                            Ok(img) => {
                                let tex = ctx.load_texture(
                                    format!("batch_preview_{idx}"),
                                    img.image,
                                    TextureOptions::LINEAR,
                                );
                                Ok(Preview {
                                    tex,
                                    width: img.width,
                                    height: img.height,
                                    n_files: img.n_files,
                                    sampled: img.sampled,
                                    detector: img.detector,
                                })
                            }
                            Err(e) => Err(e),
                        };
                        if let Some(slot) = self.previews.get_mut(idx) {
                            *slot = Some(entry);
                        }
                    }
                    Err(TryRecvError::Empty) => {
                        ctx.request_repaint_after(Duration::from_millis(100));
                        break;
                    }
                    Err(TryRecvError::Disconnected) => {
                        self.preview_rx = None;
                        break;
                    }
                }
            }
        }
    }

    fn start_previews(&mut self, ctx: &egui::Context) {
        let Some(r) = self.reference.clone() else { return };
        self.previews = (0..self.folders.len()).map(|_| None).collect();
        self.enlarged = None;
        let targets: Vec<(usize, PathBuf)> = self
            .resolved
            .iter()
            .enumerate()
            .filter_map(|(i, res)| res.as_ref().ok().map(|d| (i, d.path.clone())))
            .collect();
        for (i, res) in self.resolved.iter().enumerate() {
            if let Err(e) = res {
                self.previews[i] = Some(Err(e.clone()));
            }
        }
        let (tx, rx): (Sender<PreviewMsg>, Receiver<PreviewMsg>) = std::sync::mpsc::channel();
        self.preview_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for (idx, path) in targets {
                let detector = r.detector_for(&path);
                let result = preview_image(&path, detector, r.colormap).map_err(|e| format!("{e:#}"));
                // The receiver is dropped when the previews are restarted or
                // the window forgets them: stop loading.
                if tx.send((idx, result)).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
        });
    }

    fn start_job(&mut self, ctx: &egui::Context) {
        let (Some(r), Some(output)) = (self.reference.clone(), self.output.clone()) else {
            return;
        };
        if self.is_running() {
            return;
        }
        let disk = r.disk_crop();
        let mut plan: Vec<(usize, PathBuf, PathBuf)> = Vec::new();
        self.rows = self
            .resolved
            .iter()
            .enumerate()
            .map(|(i, res)| match res {
                Ok(d) => {
                    let stem = folder_label(&self.folders[i]);
                    let mut dest = output.join(export_folder_name(disk, &stem));
                    for c in &d.rel {
                        dest = dest.join(c);
                    }
                    plan.push((i, d.path.clone(), dest));
                    RowState::Pending
                }
                Err(e) => RowState::Skipped(e.clone()),
            })
            .collect();

        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.job = Some(Job {
            rx,
            cancel: Arc::clone(&cancel),
            finished: false,
        });
        self.job_ref = Some(r.clone());
        self.status = format!("Cropping {} folder(s)…", plan.len());
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for (idx, src, dest) in plan {
                if cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(JobMsg::Done {
                        idx,
                        result: Err("cancelled".to_owned()),
                    });
                    continue;
                }
                let detector = r.detector_for(&src);
                let progress_tx = Mutex::new(tx.clone());
                let _ = tx.send(JobMsg::Progress {
                    idx,
                    done: 0,
                    total: 0,
                });
                let result = export_folder(&src, &dest, detector, r.crop, &cancel, |done, total| {
                    if let Ok(tx) = progress_tx.lock() {
                        let _ = tx.send(JobMsg::Progress { idx, done, total });
                    }
                    ctx.request_repaint();
                })
                .map_err(|e| format!("{e:#}"));
                let _ = tx.send(JobMsg::Done { idx, result });
                ctx.request_repaint();
            }
            let _ = tx.send(JobMsg::Finished);
            ctx.request_repaint();
        });
    }

    // ----- folder list --------------------------------------------------

    fn add_folders(&mut self, dirs: Vec<PathBuf>) {
        let first_before = self.folders.first().cloned();
        for d in dirs {
            if !self.folders.contains(&d) {
                self.folders.push(d);
            }
        }
        // A fresh list: guess the data sub-folder — from where the
        // reference stack lives when it sits under one of the folders,
        // otherwise by walking down the first folder.
        if first_before.is_none() && !self.folders.is_empty() {
            let from_reference = self.reference.as_ref().and_then(|r| {
                let src = if r.source.is_dir() {
                    r.source.clone()
                } else {
                    r.source.parent()?.to_path_buf()
                };
                self.folders.iter().find_map(|top| {
                    src.strip_prefix(top).ok().map(|rest| {
                        rest.components()
                            .map(|c| c.as_os_str().to_string_lossy().into_owned())
                            .collect::<Vec<_>>()
                    })
                })
            });
            self.rel = from_reference.unwrap_or_else(|| auto_find(&self.folders[0]));
            self.rel_text = rel_text(&self.rel);
        }
        self.tree_cache.clear();
        self.resolve_all();
    }

    /// A folder dialog starting where the loaded data set is (like the main
    /// window's dialogs), else next to the folders already listed.
    fn folder_dialog(&self) -> rfd::FileDialog {
        let start = self
            .reference
            .as_ref()
            .and_then(|r| {
                if r.source.is_dir() {
                    Some(r.source.clone())
                } else {
                    r.source.parent().map(Path::to_path_buf)
                }
            })
            .or_else(|| self.folders.last().and_then(|f| f.parent().map(Path::to_path_buf)));
        let mut dialog = rfd::FileDialog::new();
        if let Some(dir) = start {
            dialog = dialog.set_directory(dir);
        }
        dialog
    }

    fn set_rel(&mut self, rel: Vec<String>) {
        self.rel_text = rel_text(&rel);
        self.rel = rel;
        self.resolve_all();
    }

    fn resolve_all(&mut self) {
        self.resolved = self.folders.iter().map(|f| resolve(f, &self.rel)).collect();
        self.rows.clear();
        self.job = None;
        self.previews.clear();
        self.preview_rx = None;
        self.enlarged = None;
    }

    fn n_ready(&self) -> usize {
        self.resolved.iter().filter(|r| r.is_ok()).count()
    }

    // ----- UI -----------------------------------------------------------

    /// The contents of the batch window.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::Panel::top("batch_top").show(ui, |ui| self.top_panel(ui));
        egui::Panel::bottom("batch_bottom").show(ui, |ui| self.bottom_panel(ui, &ctx));
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::both()
                .id_salt("batch_center")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    self.folder_table(ui);
                    if self.show_previews {
                        ui.add_space(8.0);
                        ui.separator();
                        self.preview_grid(ui);
                    }
                });
        });
    }

    fn top_panel(&mut self, ui: &mut egui::Ui) {
        let running = self.is_running();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Batch crop");
            ui.label(
                egui::RichText::new(
                    "apply the crop drawn in the main window to other folders",
                )
                .weak(),
            );
        });
        match &self.reference {
            Some(r) => {
                let disk = r.disk_crop();
                ui.label(format!(
                    "Crop: {}×{} px at ({}, {}) on disk — drawn on {} ({}×{} px, {}); \
                     adjust it in the main window",
                    disk.width,
                    disk.height,
                    disk.x,
                    disk.y,
                    folder_label(&r.source),
                    r.width,
                    r.height,
                    r.orientation
                ));
            }
            None => {
                ui.colored_label(BAD_COLOR, "Draw a crop region in the main window first.");
            }
        }
        ui.add_space(4.0);

        // Folders.
        ui.horizontal_wrapped(|ui| {
            ui.strong("1. Folders to crop:");
            if ui
                .add_enabled(!running, egui::Button::new("📁 Add folders…"))
                .on_hover_text(
                    "Pick the folders to crop (several at once). The TIFF images may sit \
                     in a sub-folder — choose which one below.",
                )
                .clicked()
                && let Some(dirs) = self
                    .folder_dialog()
                    .set_title("Pick the folders to crop (select several)")
                    .pick_folders()
            {
                self.add_folders(dirs);
            }
            if ui
                .add_enabled(!running && !self.folders.is_empty(), egui::Button::new("🗑 Clear"))
                .clicked()
            {
                self.folders.clear();
                self.tree_cache.clear();
                self.resolve_all();
            }
            ui.label(format!(
                "{} folder(s), {} with data found",
                self.folders.len(),
                self.n_ready()
            ));
        });

        // Data sub-folder.
        ui.horizontal_wrapped(|ui| {
            ui.strong("2. Data sub-folder:");
            ui.label(
                egui::RichText::new("(where the TIFF images sit inside each folder)").weak(),
            );
            let edit = ui.add_enabled(
                !running,
                egui::TextEdit::singleline(&mut self.rel_text).desired_width(320.0),
            );
            if edit.lost_focus() || (edit.changed() && !edit.has_focus()) {
                let rel = parse_rel(&self.rel_text);
                if rel != self.rel {
                    self.set_rel(rel);
                } else {
                    self.rel_text = rel_text(&self.rel);
                }
            }
            edit.on_hover_text(
                "Relative path, '.' for the folder itself. In every folder each level takes \
                 the sub-folder of that name when it exists, otherwise its only sub-folder — \
                 so run folders whose names differ still resolve.",
            );
            if ui
                .add_enabled(!running && !self.folders.is_empty(), egui::Button::new("🔍 Auto"))
                .on_hover_text(
                    "Walk down the first folder while it holds no TIFF file and has a single \
                     sub-folder",
                )
                .clicked()
            {
                let rel = auto_find(&self.folders[0]);
                self.set_rel(rel);
            }
            ui.checkbox(&mut self.show_tree, "Show the tree of the first folder");
        });
        if self.show_tree
            && !running
            && let Some(first) = self.folders.first().cloned()
        {
            egui::ScrollArea::vertical()
                .id_salt("batch_tree")
                .max_height(180.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let mut picked = None;
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        picked = dir_tree(ui, &mut self.tree_cache, &first, &[], &self.rel, 0);
                    });
                    if let Some(rel) = picked {
                        self.set_rel(rel);
                    }
                });
        }
        ui.add_space(4.0);
    }

    fn bottom_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let running = self.is_running();
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.strong("3. Output folder:");
            if ui
                .add_enabled(!running, egui::Button::new("📂 Choose…"))
                .on_hover_text(
                    "Every folder is written under it as \
                     cropped_x0…_<folder name>/<data sub-folder>",
                )
                .clicked()
            {
                let mut dialog = rfd::FileDialog::new().set_title("Pick the top output folder");
                if let Some(dir) = self.output.as_ref().or(self.folders.first()) {
                    dialog = dialog.set_directory(dir.parent().unwrap_or(dir));
                }
                if let Some(dir) = dialog.pick_folder() {
                    self.output = Some(dir);
                }
            }
            match &self.output {
                Some(p) => {
                    ui.add(egui::Label::new(p.display().to_string()).truncate())
                        .on_hover_text(p.display().to_string());
                }
                None => {
                    ui.label(egui::RichText::new("— not chosen —").weak());
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.strong("4. Run:");
            let preview_label = if self.show_previews {
                "👁 Hide previews"
            } else {
                "👁 Preview crops"
            };
            if ui
                .add_enabled(
                    self.reference.is_some() && self.n_ready() > 0,
                    egui::Button::new(preview_label),
                )
                .on_hover_text(
                    "Show a thumbnail of every folder (mean of a few images) with the crop \
                     drawn on it",
                )
                .clicked()
            {
                self.show_previews = !self.show_previews;
                if self.show_previews && self.previews.is_empty() {
                    self.start_previews(ctx);
                }
            }
            if self.show_previews
                && ui
                    .add_enabled(self.n_ready() > 0, egui::Button::new("⟳ Reload previews"))
                    .clicked()
            {
                self.start_previews(ctx);
            }
            ui.separator();
            let can_run = self.reference.is_some()
                && self.output.is_some()
                && self.n_ready() > 0
                && !running;
            if ui
                .add_enabled(can_run, egui::Button::new("▶ Crop all folders"))
                .on_hover_text("Crop every folder with data found and write it under the output folder")
                .clicked()
            {
                self.start_job(ctx);
            }
            if running {
                if ui.button("⏹ Cancel").clicked()
                    && let Some(job) = &self.job
                {
                    job.cancel.store(true, Ordering::Relaxed);
                }
                let n_done = self
                    .rows
                    .iter()
                    .filter(|r| !matches!(r, RowState::Pending | RowState::Running { .. }))
                    .count();
                let total = self.rows.len().max(1);
                ui.add(
                    egui::ProgressBar::new(n_done as f32 / total as f32)
                        .desired_width(220.0)
                        .text(format!("{n_done} / {total} folders")),
                );
                ui.spinner();
            }
        });
        if !self.status.is_empty() {
            ui.add(egui::Label::new(&self.status).truncate())
                .on_hover_text(&self.status);
        }
        ui.add_space(4.0);
    }

    fn folder_table(&mut self, ui: &mut egui::Ui) {
        if self.folders.is_empty() {
            ui.add_space(20.0);
            ui.vertical_centered(|ui| {
                ui.label("Add the folders to crop with “📁 Add folders…” above.");
            });
            return;
        }
        let running = self.is_running();
        let mut remove = None;
        egui::Grid::new("batch_rows")
            .striped(true)
            .num_columns(5)
            .min_col_width(40.0)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                ui.strong("");
                ui.strong("Folder");
                ui.strong("Data sub-folder");
                ui.strong("TIFF files");
                ui.strong("Status");
                ui.end_row();
                for i in 0..self.folders.len() {
                    if ui
                        .add_enabled(!running, egui::Button::new("✖").small())
                        .on_hover_text("Remove from the list")
                        .clicked()
                    {
                        remove = Some(i);
                    }
                    let folder = &self.folders[i];
                    ui.label(folder_label(folder))
                        .on_hover_text(folder.display().to_string());
                    match &self.resolved[i] {
                        Ok(d) => {
                            let label = ui.label(d.rel_label());
                            if !d.substitutions.is_empty() {
                                label.on_hover_text(format!(
                                    "Name not found, only sub-folder taken: {}",
                                    d.substitutions.join(", ")
                                ));
                            } else {
                                label.on_hover_text(d.path.display().to_string());
                            }
                            ui.label(d.n_tiff.to_string());
                        }
                        Err(e) => {
                            ui.colored_label(BAD_COLOR, "not found").on_hover_text(e);
                            ui.label("—");
                        }
                    }
                    match self.rows.get(i) {
                        None => match &self.resolved[i] {
                            Ok(_) => {
                                ui.label(egui::RichText::new("ready").weak());
                            }
                            Err(e) => {
                                ui.colored_label(BAD_COLOR, e);
                            }
                        },
                        Some(RowState::Pending) => {
                            ui.label(egui::RichText::new("waiting…").weak());
                        }
                        Some(RowState::Running { done, total }) => {
                            let frac = if *total > 0 {
                                *done as f32 / *total as f32
                            } else {
                                0.0
                            };
                            ui.add(
                                egui::ProgressBar::new(frac)
                                    .desired_width(180.0)
                                    .text(format!("{done} / {total} files")),
                            );
                        }
                        Some(RowState::Done(r)) => {
                            let mut tip = format!("→ {}", r.dest.display());
                            if !r.notes.is_empty() {
                                tip.push_str(&format!("\n{}", r.notes.join("\n")));
                            }
                            let mark = if r.notes.is_empty() { "" } else { " ⚠" };
                            ui.label(format!("✅ {} files{mark}", r.n_files)).on_hover_text(tip);
                        }
                        Some(RowState::Failed(e)) => {
                            ui.colored_label(BAD_COLOR, format!("❌ {e}"));
                        }
                        Some(RowState::Skipped(e)) => {
                            ui.label(egui::RichText::new(format!("skipped: {e}")).weak());
                        }
                    }
                    ui.end_row();
                }
            });
        if let Some(i) = remove {
            self.folders.remove(i);
            self.tree_cache.clear();
            self.resolve_all();
        }
    }

    fn preview_grid(&mut self, ui: &mut egui::Ui) {
        let Some(r) = self.reference.clone() else { return };
        ui.horizontal(|ui| {
            ui.heading("Preview");
            let loaded = self.previews.iter().filter(|p| p.is_some()).count();
            if loaded < self.previews.len() {
                ui.spinner();
                ui.label(format!("loading {loaded} / {}…", self.previews.len()));
            } else {
                ui.label(
                    egui::RichText::new(
                        "mean of a few images of each folder, crop in green; click to enlarge",
                    )
                    .weak(),
                );
            }
        });

        if let Some(i) = self.enlarged {
            if let Some(Some(Ok(p))) = self.previews.get(i) {
                ui.horizontal(|ui| {
                    ui.strong(folder_label(&self.folders[i]));
                    if ui.small_button("✖ close").clicked() {
                        self.enlarged = None;
                    }
                });
                draw_preview(ui, p, &r, ENLARGED_VIEW);
                ui.separator();
            } else {
                self.enlarged = None;
            }
        }

        let mut clicked = None;
        ui.horizontal_wrapped(|ui| {
            for i in 0..self.folders.len() {
                let name = folder_label(&self.folders[i]);
                ui.allocate_ui(egui::vec2(THUMB_VIEW + 8.0, THUMB_VIEW + 48.0), |ui| {
                    ui.vertical(|ui| {
                        match self.previews.get(i) {
                            Some(Some(Ok(p))) => {
                                if draw_preview(ui, p, &r, THUMB_VIEW).clicked() {
                                    clicked = Some(i);
                                }
                            }
                            Some(Some(Err(e))) => {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(THUMB_VIEW, THUMB_VIEW * 0.6),
                                    Sense::hover(),
                                );
                                ui.painter().rect_stroke(
                                    rect,
                                    2.0,
                                    Stroke::new(1.0, BAD_COLOR),
                                    egui::StrokeKind::Inside,
                                );
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "✖",
                                    egui::FontId::proportional(24.0),
                                    BAD_COLOR,
                                );
                                ui.add(egui::Label::new(egui::RichText::new(e).color(BAD_COLOR).small()).truncate())
                                    .on_hover_text(e);
                            }
                            _ => {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(THUMB_VIEW, THUMB_VIEW * 0.6),
                                    Sense::hover(),
                                );
                                ui.painter().rect_stroke(
                                    rect,
                                    2.0,
                                    Stroke::new(1.0, ui.visuals().weak_text_color()),
                                    egui::StrokeKind::Inside,
                                );
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "⏳",
                                    egui::FontId::proportional(24.0),
                                    ui.visuals().weak_text_color(),
                                );
                            }
                        }
                        ui.add(egui::Label::new(egui::RichText::new(&name).small()).truncate())
                            .on_hover_text(self.folders[i].display().to_string());
                    });
                });
            }
        });
        if let Some(i) = clicked {
            self.enlarged = if self.enlarged == Some(i) { None } else { Some(i) };
        }
    }
}

/// Draw one preview thumbnail with the crop on it (the area outside the crop
/// dimmed, the crop outline in green — red when it does not fit the image).
/// Returns the image's response, so a click can enlarge it.
fn draw_preview(ui: &mut egui::Ui, p: &Preview, r: &BatchRef, view: f32) -> egui::Response {
    let (w, h) = (p.width as f32, p.height as f32);
    let scale = (view / w).min(view / h);
    let size = egui::vec2(w * scale, h * scale);
    let response = ui.add(egui::Image::new((p.tex.id(), size)).sense(Sense::click()));
    let rect = response.rect;
    let painter = ui.painter_at(rect);
    let crop = r.crop;
    let fits = crop.x1() <= p.width && crop.y1() <= p.height;
    let same = (p.width, p.height) == (r.width, r.height);
    let to_screen = |x: usize, y: usize| {
        Pos2::new(
            rect.left() + (x as f32).min(w) * scale,
            rect.top() + (y as f32).min(h) * scale,
        )
    };
    let c = Rect::from_min_max(to_screen(crop.x, crop.y), to_screen(crop.x1(), crop.y1()));
    let dim = Color32::from_black_alpha(150);
    painter.rect_filled(Rect::from_min_max(rect.min, Pos2::new(rect.right(), c.top())), 0.0, dim);
    painter.rect_filled(Rect::from_min_max(Pos2::new(rect.left(), c.bottom()), rect.max), 0.0, dim);
    painter.rect_filled(Rect::from_min_max(Pos2::new(rect.left(), c.top()), Pos2::new(c.left(), c.bottom())), 0.0, dim);
    painter.rect_filled(Rect::from_min_max(Pos2::new(c.right(), c.top()), Pos2::new(rect.right(), c.bottom())), 0.0, dim);
    painter.rect_stroke(
        c,
        0.0,
        Stroke::new(1.5, if fits { CROP_COLOR } else { BAD_COLOR }),
        egui::StrokeKind::Inside,
    );
    let mut tip = format!(
        "{} files ({} averaged), {}×{} px, {}: {}",
        p.n_files,
        p.sampled,
        p.width,
        p.height,
        p.detector.summary(),
        p.detector.orientation()
    );
    if !fits {
        tip.push_str("\n⚠ the crop does not fit these images — this folder will fail");
    } else if !same {
        tip.push_str("\n⚠ image size differs from the reference stack; the crop still fits");
    }
    if !fits || !same {
        painter.text(
            rect.left_top() + egui::vec2(4.0, 2.0),
            egui::Align2::LEFT_TOP,
            "⚠",
            egui::FontId::proportional(18.0),
            BAD_COLOR,
        );
    }
    response.on_hover_text(tip)
}

/// The sub-folder tree of `dir` (`rel` is its path relative to the picked
/// folder), each folder selectable; returns the folder clicked, if any.
/// Directory listings are read lazily, when a node is first shown, and kept
/// in `cache`.
fn dir_tree(
    ui: &mut egui::Ui,
    cache: &mut HashMap<PathBuf, Vec<DirInfo>>,
    dir: &Path,
    rel: &[String],
    current: &[String],
    depth: usize,
) -> Option<Vec<String>> {
    let mut picked = None;
    if depth == 0 {
        let (n_tiff, _) = count_dir(dir);
        let text = format!("📂 {}  ({n_tiff} TIFF) — the folder itself", folder_label(dir));
        if ui.selectable_label(current.is_empty(), text).clicked() {
            picked = Some(Vec::new());
        }
    }
    if depth >= TREE_MAX_DEPTH {
        return picked;
    }
    let entries = cache
        .entry(dir.to_path_buf())
        .or_insert_with(|| list_subdirs(dir))
        .clone();
    ui.indent(("batch_tree_level", depth, rel), |ui| {
        for info in &entries {
            let mut child_rel = rel.to_vec();
            child_rel.push(info.name.clone());
            let selected = current == child_rel.as_slice();
            let text = format!("{}  ({} TIFF)", info.name, info.n_tiff);
            if info.n_subdirs == 0 {
                ui.horizontal(|ui| {
                    ui.add_space(18.0);
                    if ui.selectable_label(selected, text).clicked() {
                        picked = Some(child_rel.clone());
                    }
                });
            } else {
                let id = ui.make_persistent_id(("batch_tree_node", &child_rel));
                let child_dir = dir.join(&info.name);
                let open_default = current.starts_with(&child_rel);
                egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, open_default)
                    .show_header(ui, |ui| {
                        if ui.selectable_label(selected, text).clicked() {
                            picked = Some(child_rel.clone());
                        }
                    })
                    .body(|ui| {
                        if let Some(p) = dir_tree(ui, cache, &child_dir, &child_rel, current, depth + 1) {
                            picked = Some(p);
                        }
                    });
            }
        }
    });
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "crop_tiff_batch_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tiff(path: &Path, w: usize, h: usize) {
        let values: Vec<u16> = (0..w * h).map(|i| (i % 65535) as u16).collect();
        let file = std::fs::File::create(path).unwrap();
        let mut enc = tiff::encoder::TiffEncoder::new(file).unwrap();
        enc.write_image::<tiff::encoder::colortype::Gray16>(w as u32, h as u32, &values)
            .unwrap();
    }

    #[test]
    fn resolve_takes_name_or_only_subfolder() {
        let root = tmp_dir("resolve");
        // sample_a/Run_1/img/*.tif and sample_b/Run_2/img/*.tif: the run
        // folder names differ.
        for (sample, run) in [("sample_a", "Run_1"), ("sample_b", "Run_2")] {
            let img = root.join(sample).join(run).join("img");
            std::fs::create_dir_all(&img).unwrap();
            write_tiff(&img.join("f0.tif"), 8, 6);
            write_tiff(&img.join("f1.tif"), 8, 6);
        }
        let rel = auto_find(&root.join("sample_a"));
        assert_eq!(rel, vec!["Run_1".to_owned(), "img".to_owned()]);

        let a = resolve(&root.join("sample_a"), &rel).unwrap();
        assert_eq!(a.n_tiff, 2);
        assert!(a.substitutions.is_empty());
        let b = resolve(&root.join("sample_b"), &rel).unwrap();
        assert_eq!(b.path, root.join("sample_b").join("Run_2").join("img"));
        assert_eq!(b.rel, vec!["Run_2".to_owned(), "img".to_owned()]);
        assert_eq!(b.substitutions, vec!["Run_1 → Run_2".to_owned()]);

        // Ambiguous: two sub-folders and none with the name.
        std::fs::create_dir_all(root.join("sample_b").join("Run_3")).unwrap();
        assert!(resolve(&root.join("sample_b"), &rel).is_err());
        // The folder itself, without TIFFs.
        assert!(resolve(&root.join("sample_a"), &[]).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn parse_rel_drops_dots_and_empties() {
        assert_eq!(parse_rel("."), Vec::<String>::new());
        assert_eq!(parse_rel(" a//b/./c/ "), vec!["a", "b", "c"]);
        assert_eq!(rel_text(&[]), ".");
    }

    #[test]
    fn export_folder_crops_every_file() {
        let root = tmp_dir("export");
        let src = root.join("src");
        std::fs::create_dir_all(&src).unwrap();
        for i in 0..3 {
            write_tiff(&src.join(format!("img_{i}.tif")), 10, 8);
        }
        std::fs::write(src.join("run_Spectra.txt"), "spectra").unwrap();
        let dest = root.join("out");
        let crop = CropRect {
            x: 2,
            y: 1,
            width: 5,
            height: 4,
        };
        let cancel = AtomicBool::new(false);
        let progress = AtomicUsize::new(0);
        let report = export_folder(
            &src,
            &dest,
            Selection::from_path(&src),
            crop,
            &cancel,
            |done, _total| {
                progress.fetch_max(done, Ordering::Relaxed);
            },
        )
        .unwrap();
        assert_eq!(report.n_files, 3);
        assert_eq!(progress.load(Ordering::Relaxed), 3);
        assert!(dest.join("run_Spectra.txt").is_file());
        let json = std::fs::read_to_string(dest.join("crop_region.json")).unwrap();
        assert_eq!(CropRect::from_json_text(&json).unwrap(), crop);
        let data = loader::load_folder_with_progress(&dest, Selection::from_path(&dest), |_, _| {}).unwrap();
        assert_eq!(data.n_frames(), 3);
        assert_eq!((data.width, data.height), (5, 4));
        // Top-left cropped pixel is the source pixel (x=2, y=1) = 1*10+2.
        assert_eq!(data.frames[0][(0, 0)], 12.0);

        // A crop that does not fit fails the folder.
        let big = CropRect {
            x: 8,
            y: 0,
            width: 5,
            height: 4,
        };
        assert!(export_folder(&src, &root.join("out2"), Selection::from_path(&src), big, &cancel, |_, _| {}).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn preview_averages_a_few_files() {
        let root = tmp_dir("preview");
        for i in 0..20 {
            write_tiff(&root.join(format!("img_{i:02}.tif")), 600, 300);
        }
        let p = preview_image(&root, Selection::from_path(&root), Colormap::Viridis).unwrap();
        assert_eq!((p.width, p.height), (600, 300));
        assert_eq!(p.n_files, 20);
        assert_eq!(p.sampled, PREVIEW_FILES);
        assert!(p.image.width() <= THUMB_MAX && p.image.height() <= THUMB_MAX);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
