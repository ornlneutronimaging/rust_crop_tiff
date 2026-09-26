//! The egui/eframe application: folder combobox, image viewer with several
//! display modes (integrated sum, mean, max, min, std-dev, single image with
//! a slider and play button), a single rectangular crop region drawn/edited
//! on the image, and a verification panel plotting per-frame crop statistics
//! so no image of the stack loses important content.

use crate::batch::{self, BatchRef, BatchWindow};
use crate::colormap::Colormap;
use crate::crop::{OrientedBox, RectF};
use crate::loader::{self, Detector, FolderData, Selection};
use crate::stats::{self, CropStats};

use egui::{Color32, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, VLine};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const UNDO_DEPTH: usize = 24;
const HANDLE_HIT: f32 = 10.0;
const HANDLE_SIZE: f32 = 9.0;
/// Screen distance between the crop's top edge and its rotation handle.
const ROT_HANDLE_OFFSET: f32 = 26.0;
/// Rotating with Shift held snaps the tilt to this many degrees.
const ROT_SNAP_DEG: f32 = 5.0;
const PLAY_FRAME_MS: u64 = 80;

const CROP_COLOR: Color32 = Color32::from_rgb(0, 220, 160); // green
const INITIAL_CROP_COLOR: Color32 = Color32::from_rgb(255, 170, 40); // orange
const EDGE_CURVE_COLOR: Color32 = Color32::from_rgb(255, 90, 80); // red
const INSIDE_CURVE_COLOR: Color32 = Color32::from_rgb(0, 220, 160); // green
const OUTSIDE_CURVE_COLOR: Color32 = Color32::from_gray(150);

/// What the viewer shows: one of the whole-stack projections, or one image at
/// a time picked with the slider.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    Integrated,
    Mean,
    Max,
    Min,
    Std,
    Single,
}

impl DisplayMode {
    pub const ALL: [DisplayMode; 6] = [
        DisplayMode::Integrated,
        DisplayMode::Mean,
        DisplayMode::Max,
        DisplayMode::Min,
        DisplayMode::Std,
        DisplayMode::Single,
    ];

    fn label(self) -> &'static str {
        match self {
            DisplayMode::Integrated => "Integrated",
            DisplayMode::Mean => "Mean",
            DisplayMode::Max => "Max",
            DisplayMode::Min => "Min",
            DisplayMode::Std => "Std dev",
            DisplayMode::Single => "Single image",
        }
    }

    fn hover(self) -> &'static str {
        match self {
            DisplayMode::Integrated => "Pixel-wise sum of every image",
            DisplayMode::Mean => "Pixel-wise average of every image",
            DisplayMode::Max => {
                "Brightest value each pixel ever takes — bright features from \
                 any image of the stack all show at once"
            }
            DisplayMode::Min => {
                "Darkest value each pixel ever takes — for transmission images \
                 this is the union of the sample silhouettes over all images \
                 (e.g. a full CT rotation): if the dark envelope fits inside \
                 the crop, no image is cut"
            }
            DisplayMode::Std => {
                "Standard deviation of each pixel across the images — moving \
                 edges and changing regions light up"
            }
            DisplayMode::Single => {
                "One image at a time, picked with the slider below the image \
                 (▶ plays through the stack)"
            }
        }
    }

    fn index(self) -> usize {
        match self {
            DisplayMode::Integrated => 0,
            DisplayMode::Mean => 1,
            DisplayMode::Max => 2,
            DisplayMode::Min => 3,
            DisplayMode::Std => 4,
            DisplayMode::Single => 5,
        }
    }
}

/// Messages sent from the background loading thread to the UI.
enum LoadMsg {
    Progress(loader::LoadProgress),
    Done(anyhow::Result<FolderData>),
}

/// A folder load in flight on a background thread.
struct LoadJob {
    rx: Receiver<LoadMsg>,
    progress: loader::LoadProgress,
}

/// A resize grab-point on the crop rectangle: -1/0/1 for left/middle/right and
/// top/middle/bottom.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Handle {
    hx: i8,
    vy: i8,
}

/// Where the crop JSON goes when the save/return button is pressed. The crop
/// region is always returned alongside a cropped stack, so the calling
/// application can re-apply the same crop to another data set.
struct JsonDest {
    file: Option<PathBuf>,
    /// Also print the JSON on stdout for a calling application to capture
    /// (the rust_tof_profile_viewer convention).
    stdout: bool,
}

pub struct CropApp {
    /// Folders of TIFF images and/or `.npy` stack files.
    inputs: Vec<PathBuf>,
    selected_input: Option<usize>,

    data: Option<Arc<FolderData>>,
    loading: Option<LoadJob>,

    // Display.
    mode: DisplayMode,
    /// Image shown by the slider in single-image mode.
    frame_idx: usize,
    /// Full value range of each display mode, indexed by `DisplayMode::index`;
    /// the single-image entry spans all frames so the contrast stays put while
    /// sliding through them.
    ranges: [(f32, f32); 6],
    /// Sigma-clipped range of each display mode, used as the auto contrast:
    /// the dense bulk of the values. Normalized transmission stacks have
    /// outlier ratios that would otherwise stretch the contrast to the camera
    /// range and make the image look flat.
    auto_ranges: [(f32, f32); 6],
    data_min: f32,
    data_max: f32,
    vmin: f32,
    vmax: f32,
    colormap: Colormap,
    img_tex: Option<TextureHandle>,
    img_dirty: bool,
    scale: f32,
    fit_requested: bool,
    /// True while the image is shown at the fit-to-view scale (after a load or
    /// the Fit button), so it follows the viewport when the divider or the
    /// window is resized. A manual zoom (− / +) clears it.
    fitted: bool,
    /// The viewport size the fitted scale was computed for.
    fitted_viewport: egui::Vec2,
    /// Scroll offset to apply to the image viewport on the next frame, set by
    /// a Ctrl+wheel zoom so the image point under the cursor stays put.
    viewer_scroll: Option<egui::Vec2>,
    cursor: Option<(usize, usize, f32)>,
    playing: bool,
    last_advance: Option<Instant>,
    dim_outside: bool,

    // Crop model (all in the oriented/display frame of the loaded stack): a
    // rectangle, possibly tilted about its center.
    crop: Option<OrientedBox>,
    /// Crop passed on the command line (a previous session's crop). Untilted
    /// it is in the on-disk frame of the files; a tilted one is on the
    /// oriented frames, like the JSON's center / size / angle. Shown as
    /// [`CropApp::initial_crop`].
    initial_crop_disk: Option<OrientedBox>,
    /// [`CropApp::initial_crop_disk`] on the oriented frames of the loaded
    /// stack, kept as a dashed reference outline.
    initial_crop: Option<OrientedBox>,
    /// Detector chosen by the user (toolbar combobox / `--detector`), which
    /// decides how the frames are oriented on load; `None` = guess it from
    /// the folder layout.
    detector_override: Option<Detector>,
    undo: Vec<Option<OrientedBox>>,

    // Interaction transients.
    drawing: bool,
    moving: bool,
    resizing: Option<Handle>,
    /// A rotation-handle drag in progress: the constant offset between the
    /// crop angle and the pointer's angle about the center, taken at grab
    /// time so the handle doesn't jump under the pointer.
    rotating: Option<f32>,
    drag_changed: bool, // an undo snapshot was taken for the current drag
    move_last: Option<(f32, f32)>,
    drag_start: Option<(f32, f32)>,

    // Verification statistics.
    edge_band: usize,
    stats: Option<CropStats>,
    stats_rx: Option<Receiver<CropStats>>,
    stats_dirty: bool,
    plot_refit: bool,

    // Saving.
    output_path: Option<PathBuf>,
    /// Where the cropped 3-D stack (`.npy`, float32) is written on save/return.
    output_stack: Option<PathBuf>,
    called_from_app: bool,
    /// A save running on a background thread (the cropped stack can be GBs).
    saving_rx: Option<Receiver<Result<String, String>>>,
    close_after_save: bool,
    instructions: Option<String>,
    show_instructions: bool,

    /// The batch-crop window (a separate OS window) and its running job.
    batch: BatchWindow,

    status: String,
}

impl CropApp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inputs: Vec<PathBuf>,
        initial_crop: Option<OrientedBox>,
        output_path: Option<PathBuf>,
        output_stack: Option<PathBuf>,
        called_from_app: bool,
        instructions: Option<String>,
    ) -> Self {
        let status = if inputs.is_empty() {
            "Add a folder of TIFF images or a .npy stack to begin.".to_owned()
        } else {
            format!("{} input(s) on the command line.", inputs.len())
        };
        Self {
            inputs,
            selected_input: None,
            data: None,
            loading: None,
            mode: DisplayMode::Integrated,
            frame_idx: 0,
            ranges: [(0.0, 1.0); 6],
            auto_ranges: [(0.0, 1.0); 6],
            data_min: 0.0,
            data_max: 1.0,
            vmin: 0.0,
            vmax: 1.0,
            colormap: Colormap::Viridis,
            img_tex: None,
            img_dirty: false,
            scale: 1.0,
            fit_requested: false,
            fitted: false,
            fitted_viewport: egui::Vec2::ZERO,
            viewer_scroll: None,
            cursor: None,
            playing: false,
            last_advance: None,
            dim_outside: true,
            crop: None,
            initial_crop_disk: initial_crop,
            initial_crop: None,
            detector_override: None,
            undo: Vec::new(),
            drawing: false,
            moving: false,
            resizing: None,
            rotating: None,
            drag_changed: false,
            move_last: None,
            drag_start: None,
            edge_band: 5,
            stats: None,
            stats_rx: None,
            stats_dirty: false,
            plot_refit: false,
            output_path,
            output_stack,
            called_from_app,
            saving_rx: None,
            close_after_save: false,
            show_instructions: instructions.is_some(),
            instructions,
            batch: BatchWindow::new(),
            status,
        }
    }

    /// Force the detector (hence the orientation) of every input, `None` to
    /// go back to the automatic guess (the `--detector` command-line option).
    pub fn set_detector_override(&mut self, detector: Option<Detector>) {
        self.detector_override = detector;
    }

    /// Detector to load `path` with: the user's override, else the folder
    /// layout (`images/tpx1`, `images/ikonxl`, …).
    fn detector_for(&self, path: &Path) -> Selection {
        let mut sel = Selection::from_path(path);
        sel.manual = self.detector_override;
        sel
    }

    /// Toolbar combobox choosing the detector (hence the orientation on
    /// load): "auto" follows the folder layout, the other entries force one.
    /// Changing it reloads the displayed input.
    fn detector_combo(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Detector:").on_hover_text(
            "How the frames are oriented on load: Timepix → transposed, CCD → flipped \
             vertically, QHY → rotated 90° counterclockwise. 'auto' recognizes the detector from \
             the folder layout (images/tpx1, images/ikonxl, …). The crop is drawn on the \
             oriented image; the saved crop and every cropped file are in the on-disk frame \
             of the input files.",
        );
        let auto_text = match self.data.as_ref() {
            Some(d) if d.detector.is_auto() => format!("auto: {}", d.detector.summary()),
            _ => "auto".to_owned(),
        };
        let current = match self.detector_override {
            None => auto_text.clone(),
            Some(d) => d.label().to_owned(),
        };
        let mut changed = false;
        egui::ComboBox::from_id_salt("detector")
            .selected_text(current)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(self.detector_override.is_none(), auto_text)
                    .on_hover_text("Guess the detector from the folder layout")
                    .clicked()
                    && self.detector_override.is_some()
                {
                    self.detector_override = None;
                    changed = true;
                }
                for d in Detector::ALL {
                    if ui
                        .selectable_label(self.detector_override == Some(d), d.label())
                        .on_hover_text(d.description())
                        .clicked()
                        && self.detector_override != Some(d)
                    {
                        self.detector_override = Some(d);
                        changed = true;
                    }
                }
            });
        if let Some(d) = self.data.as_ref() {
            ui.label(egui::RichText::new(orientation_note(d)).weak())
                .on_hover_text(if d.path.is_dir() {
                    d.detector.detector().description().to_owned()
                } else {
                    "A .npy stack is shown exactly as it is: the application that wrote it \
                     (e.g. NeCTAR) already oriented the frames for this detector."
                        .to_owned()
                });
        }
        if changed && let Some(idx) = self.selected_input {
            self.select_input(idx, ctx);
        }
    }

    /// Start loading the first command-line input, if any.
    pub fn select_first(&mut self, ctx: &egui::Context) {
        if !self.inputs.is_empty() {
            self.select_input(0, ctx);
        }
    }

    /// The crop as it would be saved (size snapped to whole pixels, angle
    /// canonical; untilted crops on the pixel grid inside the image), if a
    /// valid one is on the image.
    fn crop_box(&self) -> Option<OrientedBox> {
        let data = self.data.as_ref()?;
        self.crop?.snapped(data.width, data.height)
    }

    // ----- input loading -----------------------------------------------------

    fn select_input(&mut self, idx: usize, ctx: &egui::Context) {
        let Some(dir) = self.inputs.get(idx).cloned() else {
            return;
        };
        self.selected_input = Some(idx);
        self.status = format!("Loading {}…", folder_label(&dir));
        let detector = self.detector_for(&dir);
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            // The parallel loader reports progress from worker threads, so the
            // sender goes behind a mutex.
            let progress_tx = Mutex::new(tx.clone());
            let result = loader::load_input_with_progress(&dir, detector, |progress| {
                if let Ok(tx) = progress_tx.lock() {
                    let _ = tx.send(LoadMsg::Progress(progress.clone()));
                }
                ctx.request_repaint();
            });
            let _ = tx.send(LoadMsg::Done(result));
            ctx.request_repaint();
        });

        // Replacing an in-flight job drops its receiver; the stale thread's
        // sends fail silently and its result is discarded.
        self.loading = Some(LoadJob {
            rx,
            progress: loader::LoadProgress::default(),
        });
    }

    fn poll_load(&mut self) {
        let mut result = None;
        if let Some(job) = &mut self.loading {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    LoadMsg::Progress(progress) => {
                        // A .npy stack: say how many images the file holds.
                        if !progress.stack_file.is_empty() && self.status.starts_with("Loading ") {
                            self.status = format!(
                                "Loading {} image{} from {}…",
                                progress.frames_total,
                                if progress.frames_total == 1 { "" } else { "s" },
                                progress.stack_file
                            );
                        }
                        job.progress = progress;
                    }
                    LoadMsg::Done(res) => result = Some(res),
                }
            }
        }
        if let Some(res) = result {
            self.loading = None;
            match res {
                Ok(data) => self.apply_data(data),
                Err(e) => self.status = format!("Load failed: {e:#}"),
            }
        }
    }

    fn apply_data(&mut self, data: FolderData) {
        let (w, h) = (data.width, data.height);

        // Keep the user's crop when the new folder has the same image size
        // and orientation.
        let same_dims = self
            .data
            .as_ref()
            .map(|d| (d.width, d.height, d.orientation))
            == Some((w, h, data.orientation));
        if !same_dims {
            self.crop = None;
            self.undo.clear();
        }
        // The command-line crop is in the on-disk frame: show it on the
        // oriented frames of this stack.
        // (A tilted one is already on the oriented frames.)
        let (disk_w, disk_h) = data.disk_dims();
        self.initial_crop = self.initial_crop_disk.map(|b| match b.as_crop(disk_w, disk_h) {
            Some(c) if b.is_axis_aligned() => {
                OrientedBox::from_crop(c.from_disk(data.orientation, disk_w, disk_h))
            }
            _ => b,
        });
        self.drawing = false;
        self.moving = false;
        self.resizing = None;
        self.rotating = None;
        self.playing = false;

        // Show the initial (previous-session) crop as the starting region.
        let mut crop_note = String::new();
        if self.crop.is_none() {
            if let Some(init) = self.initial_crop {
                match init.snapped(w, h) {
                    Some(c) => {
                        self.crop = Some(c);
                        if c != init {
                            crop_note = " — initial crop clipped to the image".to_owned();
                        }
                    }
                    None => crop_note = " — initial crop is outside the image, ignored".to_owned(),
                }
            }
        }

        let frames_range = data
            .frames
            .iter()
            .map(|f| raw_range(f.iter().copied()))
            .fold((f32::INFINITY, f32::NEG_INFINITY), |a, b| {
                (a.0.min(b.0), a.1.max(b.1))
            });
        self.ranges = [
            normalize_range(raw_range(data.sum.iter().copied())),
            normalize_range(raw_range(data.mean.iter().copied())),
            normalize_range(raw_range(data.max.iter().copied())),
            normalize_range(raw_range(data.min.iter().copied())),
            normalize_range(raw_range(data.std.iter().copied())),
            normalize_range(frames_range),
        ];
        self.auto_ranges = [
            robust_range(std::slice::from_ref(&data.sum), self.ranges[0]),
            robust_range(std::slice::from_ref(&data.mean), self.ranges[1]),
            robust_range(std::slice::from_ref(&data.max), self.ranges[2]),
            robust_range(std::slice::from_ref(&data.min), self.ranges[3]),
            robust_range(std::slice::from_ref(&data.std), self.ranges[4]),
            robust_range(&data.frames, self.ranges[5]),
        ];
        self.frame_idx = self.frame_idx.min(data.n_frames() - 1);
        self.apply_active_range();

        self.status = format!(
            "Loaded {} images ({w}×{h} px, {}: {}) from {}{crop_note}",
            data.n_frames(),
            data.detector.summary(),
            orientation_note(&data),
            folder_label(&data.path),
        );

        self.data = Some(Arc::new(data));
        self.img_dirty = true;
        self.fit_requested = true;
        self.plot_refit = true;
        self.stats = None;
        self.stats_dirty = true;
    }

    // ----- crop statistics ----------------------------------------------------

    fn poll_stats(&mut self) {
        if let Some(rx) = &self.stats_rx {
            match rx.try_recv() {
                Ok(s) => {
                    self.stats = Some(s);
                    self.stats_rx = None;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.stats_rx = None,
            }
        }
    }

    /// At most one statistics computation runs at a time; if the crop changes
    /// while one is in flight, `stats_dirty` stays set and the next call
    /// starts a fresh one, so the plot converges to the latest crop.
    fn maybe_spawn_stats(&mut self, ctx: &egui::Context) {
        if !self.stats_dirty || self.stats_rx.is_some() || self.loading.is_some() {
            return;
        }
        let Some(data) = &self.data else { return };
        self.stats_dirty = false;
        let Some(crop) = self.crop_box() else {
            self.stats = None;
            return;
        };

        let (tx, rx) = std::sync::mpsc::channel();
        let data = Arc::clone(data);
        let band = self.edge_band.max(1);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let out = stats::compute_oriented(&data.frames, &data.frame_totals, &crop, band);
            let _ = tx.send(out);
            ctx.request_repaint();
        });
        self.stats_rx = Some(rx);
    }

    // ----- crop model ---------------------------------------------------------

    fn push_undo(&mut self) {
        self.undo.push(self.crop);
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
    }

    fn undo(&mut self) {
        let Some(prev) = self.undo.pop() else { return };
        self.crop = prev;
        self.stats_dirty = true;
    }

    fn set_crop(&mut self, crop: Option<OrientedBox>) {
        if self.crop != crop {
            self.push_undo();
            self.crop = crop;
            self.stats_dirty = true;
        }
    }

    /// Snap the live crop to whole-pixel sizes (as saved; untilted ones onto
    /// the pixel grid); drop it when it is degenerate or entirely off the
    /// image, restoring the pre-drag state.
    fn snap_crop(&mut self) {
        let Some(data) = &self.data else { return };
        if let Some(b) = self.crop {
            match b.snapped(data.width, data.height) {
                Some(c) => self.crop = Some(c),
                None => self.undo(),
            }
        }
    }

    // ----- display -----------------------------------------------------------

    /// The image currently on screen.
    fn display_image<'a>(&self, data: &'a FolderData) -> &'a Array2<f32> {
        match self.mode {
            DisplayMode::Integrated => &data.sum,
            DisplayMode::Mean => &data.mean,
            DisplayMode::Max => &data.max,
            DisplayMode::Min => &data.min,
            DisplayMode::Std => &data.std,
            DisplayMode::Single => &data.frames[self.frame_idx.min(data.n_frames() - 1)],
        }
    }

    fn set_mode(&mut self, mode: DisplayMode) {
        if self.mode != mode {
            self.mode = mode;
            if mode != DisplayMode::Single {
                self.playing = false;
            }
            self.apply_active_range();
        }
    }

    /// Reset the contrast limits to the auto (sigma-clipped) range of the
    /// displayed mode, and the manual edit bounds to its full range.
    fn apply_active_range(&mut self) {
        let (lo, hi) = self.ranges[self.mode.index()];
        self.data_min = lo;
        self.data_max = hi;
        let (alo, ahi) = self.auto_ranges[self.mode.index()];
        self.vmin = alo;
        self.vmax = ahi;
        self.img_dirty = true;
    }

    fn ensure_img_texture(&mut self, ctx: &egui::Context) {
        if !self.img_dirty {
            return;
        }
        let Some(data) = &self.data else { return };
        let img = self.display_image(data);
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let span = (self.vmax - self.vmin).max(1e-12);
        let lut = self.colormap.lut();
        let mut buf = vec![0u8; w * h * 4];
        for (i, &v) in img.iter().enumerate() {
            let t = ((v - self.vmin) / span).clamp(0.0, 1.0);
            let idx = ((t * 255.0).round() as usize).min(255);
            let [r, g, b] = lut[idx];
            buf[i * 4] = r;
            buf[i * 4 + 1] = g;
            buf[i * 4 + 2] = b;
            buf[i * 4 + 3] = 255;
        }
        let color = egui::ColorImage::from_rgba_unmultiplied([w, h], &buf);
        self.img_tex = Some(ctx.load_texture("display", color, TextureOptions::NEAREST));
        self.img_dirty = false;
    }

    /// Advance the single-image slider while playing.
    fn animate(&mut self, ctx: &egui::Context) {
        if !self.playing || self.mode != DisplayMode::Single {
            return;
        }
        let Some(data) = &self.data else { return };
        let now = Instant::now();
        let due = self
            .last_advance
            .is_none_or(|t| now.duration_since(t) >= Duration::from_millis(PLAY_FRAME_MS));
        if due {
            self.frame_idx = (self.frame_idx + 1) % data.n_frames();
            self.last_advance = Some(now);
            self.img_dirty = true;
        }
        ctx.request_repaint_after(Duration::from_millis(PLAY_FRAME_MS / 2));
    }

    /// Jump the viewer to one image (e.g. from a click on the plot).
    fn show_frame(&mut self, idx: usize) {
        let Some(n_frames) = self.data.as_ref().map(|d| d.n_frames()) else {
            return;
        };
        self.set_mode(DisplayMode::Single);
        self.frame_idx = idx.min(n_frames - 1);
        self.playing = false;
        self.img_dirty = true;
    }

    // ----- saving -------------------------------------------------------------

    /// Write the crop JSON (to a file or stdout) and, when asked, the cropped
    /// 3-D stack, on a background thread — the stack can be GBs. When
    /// `close_after` is set the window closes once the save succeeded (the
    /// save-and-quit / return-to-caller workflows).
    fn start_save(&mut self, json_dest: JsonDest, stack_dest: Option<PathBuf>, close_after: bool) {
        if self.saving_rx.is_some() {
            return;
        }
        let (Some(data), Some(crop)) = (self.data.clone(), self.crop_box()) else {
            self.status = "Nothing to save — draw a crop region first.".to_owned();
            return;
        };

        let (tx, rx) = std::sync::mpsc::channel();
        self.saving_rx = Some(rx);
        self.close_after_save = close_after;
        self.status = match &stack_dest {
            Some(p) => format!("Writing the cropped stack to {}…", p.display()),
            None => "Saving the crop…".to_owned(),
        };

        std::thread::spawn(move || {
            let result = (|| -> Result<String, String> {
                let mut notes: Vec<String> = Vec::new();
                if let Some(stack_path) = &stack_dest {
                    loader::write_cropped_stack(stack_path, &data.frames, &crop, data.orientation)
                        .map_err(|e| format!("Failed to write {}: {e:#}", stack_path.display()))?;
                    notes.push(format!("cropped stack → {}", stack_path.display()));
                }
                let json = crop.to_json(
                    data.width,
                    data.height,
                    &data.path.display().to_string(),
                    data.detector.detector(),
                    data.orientation,
                );
                if let Some(path) = &json_dest.file {
                    std::fs::write(path, &json)
                        .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
                    notes.push(format!("crop → {}", path.display()));
                }
                if json_dest.stdout {
                    println!("{json}");
                    notes.push("crop → stdout".to_owned());
                }
                Ok(format!("Saved: {}", notes.join(", ")))
            })();
            let _ = tx.send(result);
        });
    }

    fn poll_saving(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.saving_rx else { return };
        match rx.try_recv() {
            Ok(result) => {
                self.saving_rx = None;
                match result {
                    Ok(msg) => {
                        self.status = msg;
                        if self.close_after_save {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                    Err(e) => self.status = e,
                }
                self.close_after_save = false;
            }
            Err(TryRecvError::Empty) => {
                // Keep polling while the background save runs.
                ctx.request_repaint_after(Duration::from_millis(100));
            }
            Err(TryRecvError::Disconnected) => {
                self.saving_rx = None;
                self.close_after_save = false;
            }
        }
    }

    /// The crop-JSON destination of the save-and-quit / return-to-caller
    /// button: `--output` when given, otherwise a `<stem>_crop.json` sidecar
    /// next to the cropped stack — the crop region is always returned so the
    /// caller can re-apply it to another data set. Driven by another
    /// application, the JSON is additionally printed on stdout.
    fn quit_json_dest(&self) -> JsonDest {
        let file = self
            .output_path
            .clone()
            .or_else(|| self.output_stack.as_deref().map(stack_sidecar_json));
        JsonDest {
            file,
            stdout: self.called_from_app,
        }
    }

    /// The save-and-quit / return-to-caller button: crop JSON (and cropped
    /// stack when `--output-stack` was given), then close.
    fn save_and_quit(&mut self) {
        self.start_save(self.quit_json_dest(), self.output_stack.clone(), true);
    }

    /// The folder the currently displayed input was loaded from, so save
    /// dialogs open next to the data instead of in the working directory.
    fn loaded_dir(&self) -> Option<PathBuf> {
        let path = &self.data.as_ref()?.path;
        if path.is_dir() {
            Some(path.clone())
        } else {
            path.parent().map(Path::to_path_buf)
        }
    }

    /// A file dialog that starts in the loaded input's folder when one is known.
    fn file_dialog(&self) -> rfd::FileDialog {
        let mut dialog = rfd::FileDialog::new();
        if let Some(dir) = self.loaded_dir() {
            dialog = dialog.set_directory(dir);
        }
        dialog
    }

    fn save_crop_dialog(&mut self) {
        let Some(path) = self
            .file_dialog()
            .add_filter("JSON", &["json"])
            .set_file_name("crop_region.json")
            .set_title("Save the crop region as JSON")
            .save_file()
        else {
            return;
        };
        self.start_save(
            JsonDest {
                file: Some(path),
                stdout: false,
            },
            None,
            false,
        );
    }

    fn save_stack_dialog(&mut self) {
        let Some(path) = self
            .file_dialog()
            .add_filter("NumPy", &["npy"])
            .set_file_name("cropped_stack.npy")
            .set_title("Save the cropped 3-D stack as .npy (float32)")
            .save_file()
        else {
            return;
        };
        // The applied crop region always travels with the stack, as a
        // `<stem>_crop.json` sidecar.
        self.start_save(
            JsonDest {
                file: Some(stack_sidecar_json(&path)),
                stdout: false,
            },
            Some(path),
            false,
        );
    }

    fn export_folder_dialog(&mut self) {
        let Some(dest) = self
            .file_dialog()
            .set_title("Export the cropped images — pick where the new folder is created")
            .pick_folder()
        else {
            return;
        };
        self.start_export(dest);
    }

    /// Apply the crop to every image and write them as individual TIFF files
    /// into a new folder inside `parent`, named after the crop bounds and the
    /// input (`cropped_x0<x0>_y0<y0>_x1<x1>_y1<y1>_<input>`, exclusive stops),
    /// with the spectra file copied along and a `crop_region.json` sidecar —
    /// on a background thread.
    fn start_export(&mut self, parent: PathBuf) {
        if self.saving_rx.is_some() {
            return;
        }
        let (Some(data), Some(crop)) = (self.data.clone(), self.crop_box()) else {
            self.status = "Nothing to export — draw a crop region first.".to_owned();
            return;
        };
        let stem = if data.path.is_dir() {
            data.path.file_name()
        } else {
            data.path.file_stem()
        }
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "cropped".to_owned());
        // Folder name from the bounds in the on-disk frame, like the crop JSON.
        let disk = crop.disk_bounds(data.orientation, data.width, data.height);
        let dest = parent.join(batch::export_folder_name(disk, &stem));

        let (tx, rx) = std::sync::mpsc::channel();
        self.saving_rx = Some(rx);
        self.close_after_save = false;
        self.status = format!("Exporting the cropped images to {}…", dest.display());

        std::thread::spawn(move || {
            let result = (|| -> Result<String, String> {
                let mut notes = loader::export_cropped_images(&dest, &data, &crop)
                    .map_err(|e| format!("Export failed: {e:#}"))?;
                let json = crop.to_json(
                    data.width,
                    data.height,
                    &data.path.display().to_string(),
                    data.detector.detector(),
                    data.orientation,
                );
                let json_path = dest.join("crop_region.json");
                std::fs::write(&json_path, &json)
                    .map_err(|e| format!("Failed to write {}: {e}", json_path.display()))?;
                notes.push(format!("crop → {}", json_path.display()));
                Ok(format!("Exported: {}", notes.join(", ")))
            })();
            let _ = tx.send(result);
        });
    }

    // ----- batch crop window ----------------------------------------------

    /// The crop and stack of the main window, as the batch window applies
    /// them to other folders; `None` without a crop.
    fn batch_ref(&self) -> Option<BatchRef> {
        let data = self.data.as_ref()?;
        let crop = self.crop_box()?;
        Some(BatchRef {
            crop,
            width: data.width,
            height: data.height,
            orientation: data.orientation,
            source: data.path.clone(),
            detector_override: self.detector_override,
            colormap: self.colormap,
        })
    }

    /// The batch-crop window, a separate OS window (an embedded egui window
    /// on backends without multiple viewports). Its job keeps running, and
    /// is polled, while the window is closed.
    fn batch_window(&mut self, ctx: &egui::Context) {
        self.batch.poll(ctx);
        if !self.batch.open {
            return;
        }
        let reference = self.batch_ref();
        self.batch.set_reference(reference);
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("batch_crop"),
            egui::ViewportBuilder::default()
                .with_title("VENUS Crop TIFF — Batch crop")
                .with_inner_size([1040.0, 780.0])
                .with_min_inner_size([640.0, 480.0]),
            |ui, _class| {
                self.batch.ui(ui);
                if ui.ctx().input(|i| i.viewport().close_requested()) {
                    self.batch.open = false;
                }
            },
        );
    }

    // ----- instructions modal ----------------------------------------------

    /// Modal shown on top of the application when the caller passed
    /// `--instructions`: an info icon, the caller's text, and a dismiss
    /// button. Clicking outside the dialog or pressing Escape also closes it.
    fn instructions_modal(&mut self, ctx: &egui::Context) {
        if !self.show_instructions {
            return;
        }
        let Some(text) = self.instructions.clone() else {
            self.show_instructions = false;
            return;
        };

        let modal = egui::Modal::new(egui::Id::new("instructions_modal")).show(ctx, |ui| {
            ui.set_max_width(520.0);

            ui.horizontal(|ui| {
                // Info icon: white ℹ on a filled blue disc.
                let (rect, _) = ui.allocate_exact_size(egui::vec2(40.0, 40.0), Sense::hover());
                ui.painter()
                    .circle_filled(rect.center(), 19.0, Color32::from_rgb(37, 99, 235));
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "ℹ",
                    egui::FontId::proportional(26.0),
                    Color32::WHITE,
                );
                ui.add_space(6.0);
                ui.heading("Instructions");
            });
            ui.separator();
            ui.add_space(4.0);

            egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                ui.label(egui::RichText::new(text).size(15.0));
            });

            ui.add_space(10.0);
            ui.vertical_centered(|ui| {
                if ui
                    .button(egui::RichText::new("  Got it  ").size(15.0))
                    .clicked()
                {
                    self.show_instructions = false;
                }
            });
        });
        if modal.should_close() {
            self.show_instructions = false;
        }
    }

    // ----- UI -------------------------------------------------------------------

    fn add_folder_dialog(&mut self, ctx: &egui::Context) {
        if let Some(dir) = rfd::FileDialog::new()
            .set_title("Add a folder of TIFF images")
            .pick_folder()
        {
            self.inputs.push(dir);
            self.select_input(self.inputs.len() - 1, ctx);
        }
    }

    fn add_npy_dialog(&mut self, ctx: &egui::Context) {
        if let Some(file) = rfd::FileDialog::new()
            .add_filter("NumPy", &["npy"])
            .set_title("Add a .npy stack (2-D or 3-D array)")
            .pick_file()
        {
            self.inputs.push(file);
            self.select_input(self.inputs.len() - 1, ctx);
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let ctx = ui.ctx().clone();
            if ui.button("📁 Add folder…").clicked() {
                self.add_folder_dialog(&ctx);
            }
            if ui
                .button("🗋 Add .npy…")
                .on_hover_text("Add a 2-D or 3-D .npy stack file (e.g. exported by another application)")
                .clicked()
            {
                self.add_npy_dialog(&ctx);
            }

            ui.separator();

            ui.label("Input:");
            let current = self
                .selected_input
                .and_then(|i| self.inputs.get(i))
                .map(|p| folder_label(p))
                .unwrap_or_else(|| "— select an input —".to_owned());
            let mut clicked = None;
            egui::ComboBox::from_id_salt("input")
                .selected_text(current)
                .width(320.0)
                .show_ui(ui, |ui| {
                    for (i, f) in self.inputs.iter().enumerate() {
                        if ui
                            .selectable_label(self.selected_input == Some(i), folder_label(f))
                            .on_hover_text(f.display().to_string())
                            .clicked()
                        {
                            clicked = Some(i);
                        }
                    }
                });
            if let Some(i) = clicked {
                if self.selected_input != Some(i) {
                    self.select_input(i, &ctx);
                }
            }

            ui.separator();

            self.detector_combo(ui, &ctx);

            ui.separator();

            ui.label("Display:");
            for m in DisplayMode::ALL {
                if ui
                    .selectable_label(self.mode == m, m.label())
                    .on_hover_text(m.hover())
                    .clicked()
                {
                    self.set_mode(m);
                }
            }

            ui.separator();

            ui.label("Contrast:");
            let range = self.data_min..=self.data_max;
            let speed = (self.data_max - self.data_min).max(1.0) / 200.0;
            let r1 = ui.add(
                egui::DragValue::new(&mut self.vmin)
                    .speed(speed)
                    .range(range.clone()),
            );
            let r2 = ui.add(egui::DragValue::new(&mut self.vmax).speed(speed).range(range));
            if ui
                .button("Auto")
                .on_hover_text("Range of the bulk of the data, ignoring outlier pixels")
                .clicked()
            {
                self.apply_active_range();
            }
            if ui
                .button("Full")
                .on_hover_text("Full min–max range, outliers included")
                .clicked()
            {
                self.vmin = self.data_min;
                self.vmax = self.data_max;
                self.img_dirty = true;
            }
            if r1.changed() || r2.changed() {
                self.img_dirty = true;
            }

            ui.separator();

            let mut cmap_changed = false;
            egui::ComboBox::from_id_salt("colormap")
                .selected_text(format!("Colormap: {}", self.colormap.label()))
                .show_ui(ui, |ui| {
                    for c in Colormap::ALL {
                        cmap_changed |= ui
                            .selectable_value(&mut self.colormap, c, c.label())
                            .changed();
                    }
                });
            if cmap_changed {
                self.img_dirty = true;
            }

            if self.instructions.is_some() {
                ui.separator();
                if ui.button("ℹ Instructions").clicked() {
                    self.show_instructions = true;
                }
            }

            ui.separator();
            crate::theme::toggle_button(ui);
            crate::zoom::toggle_button(ui);
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Crop:");
            if ui
                .add_enabled(!self.undo.is_empty(), egui::Button::new("↩ Undo"))
                .clicked()
            {
                self.undo();
            }
            let has_data = self.data.is_some();
            if ui
                .add_enabled(has_data, egui::Button::new("⛶ Full image"))
                .on_hover_text("Set the crop to the whole image")
                .clicked()
            {
                if let Some(data) = &self.data {
                    let full = OrientedBox::full(data.width, data.height);
                    self.set_crop(Some(full));
                }
            }
            if let Some(init) = self.initial_crop {
                if ui
                    .add_enabled(has_data, egui::Button::new("↧ Use initial crop"))
                    .on_hover_text("Reset the crop to the region passed on the command line")
                    .clicked()
                {
                    if let Some(data) = &self.data {
                        match init.snapped(data.width, data.height) {
                            Some(c) => self.set_crop(Some(c)),
                            None => {
                                self.status =
                                    "Initial crop is outside this image — not applied.".to_owned()
                            }
                        }
                    }
                }
            }
            if let Some(b) = self.crop
                && b.angle_deg != 0.0
                && ui
                    .button("⟲ Straighten")
                    .on_hover_text("Reset the tilt of the crop region to 0° (keeping its center and size)")
                    .clicked()
            {
                self.set_crop(Some(OrientedBox { angle_deg: 0.0, ..b }));
                self.snap_crop();
            }
            if ui
                .add_enabled(self.crop.is_some(), egui::Button::new("🗑 Clear"))
                .clicked()
            {
                self.set_crop(None);
            }

            ui.separator();
            ui.checkbox(&mut self.dim_outside, "Dim outside")
                .on_hover_text("Darken everything the crop throws away");

            ui.separator();
            ui.label("Zoom:")
                .on_hover_text("Ctrl + mouse wheel over the image zooms around the cursor");
            if ui.button("−").clicked() {
                self.scale = (self.scale / 1.25).max(0.02);
                self.fitted = false;
            }
            if ui.button("+").clicked() {
                self.scale = (self.scale * 1.25).min(64.0);
                self.fitted = false;
            }
            if ui.button("Fit").clicked() {
                self.fit_requested = true;
            }
            ui.label(format!("{:.0}%", self.scale * 100.0));
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Save buttons sit at the far right; lay them out first so the
            // status text on the left can take the remaining width.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let can_save = self.crop_box().is_some() && self.saving_rx.is_none();

                // Save-and-quit / return-to-caller button, when there is
                // somewhere for the result to go.
                if self.called_from_app || self.output_path.is_some() || self.output_stack.is_some()
                {
                    let dest = self.quit_json_dest();
                    let mut sends = Vec::new();
                    if let Some(p) = &dest.file {
                        sends.push(format!("crop → {}", p.display()));
                    }
                    if dest.stdout {
                        sends.push("crop → stdout".to_owned());
                    }
                    if let Some(p) = &self.output_stack {
                        sends.push(format!("cropped 3-D stack (.npy) → {}", p.display()));
                    }
                    let (label, hover) = if self.called_from_app {
                        (
                            "↩ Return to main application",
                            format!(
                                "Return the result to the calling application ({}) and close",
                                sends.join(", ")
                            ),
                        )
                    } else {
                        (
                            "✅ Save crop & quit",
                            format!("Write the result ({}) and close", sends.join(", ")),
                        )
                    };
                    if ui
                        .add_enabled(can_save, egui::Button::new(label))
                        .on_hover_text(hover)
                        .clicked()
                    {
                        self.save_and_quit();
                    }
                }
                if ui
                    .add_enabled(can_save, egui::Button::new("💾 Save as JSON…"))
                    .on_hover_text("Write the crop region (x, y, width, height — plus center and angle when tilted) to a JSON file")
                    .clicked()
                {
                    self.save_crop_dialog();
                }
                if ui
                    .add_enabled(can_save, egui::Button::new("🗋 Save as NumPy array…"))
                    .on_hover_text(
                        "Write the cropped 3-D stack as a NumPy .npy file \
                         (float32, shape (n_images, height, width))",
                    )
                    .clicked()
                {
                    self.save_stack_dialog();
                }
                if ui
                    .add_enabled(
                        self.crop_box().is_some() || self.batch.is_running(),
                        egui::Button::new("🗄 Batch crop…"),
                    )
                    .on_hover_text(
                        "Apply this crop to many other folders at once, in a separate \
                         window: pick the folders, the sub-folder holding their images, \
                         and an output folder",
                    )
                    .clicked()
                {
                    let reference = self.batch_ref();
                    self.batch.show(reference);
                }
                if ui
                    .add_enabled(can_save, egui::Button::new("🗀 Export cropped stack…"))
                    .on_hover_text(
                        "Apply the crop to every image and write them as individual \
                         TIFF files (float32) into a new folder named after the \
                         input and the crop bounds, copying the spectra file along",
                    )
                    .clicked()
                {
                    self.export_folder_dialog();
                }
                if self.saving_rx.is_some() {
                    ui.spinner();
                }
                ui.separator();
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    // Truncated: labels are click-selectable, so a long status
                    // (e.g. an export path) spilling under the buttons would
                    // not just look wrong — it would steal their clicks.
                    ui.add(egui::Label::new(&self.status).truncate())
                        .on_hover_text(&self.status);
                    if let Some((x, y, v)) = self.cursor {
                        ui.separator();
                        ui.add(egui::Label::new(format!("({x}, {y}) = {v:.4}")).truncate());
                    }
                });
            });
        });
    }

    /// The right-hand panel: crop coordinates, size summary, and the per-frame
    /// verification plot.
    fn crop_panel(&mut self, ui: &mut egui::Ui) {
        // The plot adapts to the window, but the crop fields, the
        // verification text and the plot's minimum height can outgrow a short
        // window (small displays, the large-text mode). The panel height is
        // measured before entering the scroll area — inside it the available
        // height is unbounded — and the plot's minimum height below is what
        // makes the scroll bar appear.
        let panel_h = ui.available_height();
        egui::ScrollArea::vertical()
            .id_salt("crop_panel_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| self.crop_panel_content(ui, panel_h));
    }

    fn crop_panel_content(&mut self, ui: &mut egui::Ui, panel_h: f32) {
        ui.set_min_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.heading("Crop region");
            if self.stats_rx.is_some() {
                ui.spinner();
            }
        });
        ui.add_space(4.0);

        let Some(data) = self.data.clone() else {
            ui.label("Load a folder to define the crop region.");
            return;
        };
        let (w, h) = (data.width, data.height);

        match self.crop_box() {
            None => {
                ui.label(
                    "Drag a rectangle on the image to define the crop; the round handle \
                     above it tilts the region (Shift snaps to 5°).",
                );
            }
            Some(crop) => {
                if self.edit_crop_fields(ui, crop, w, h) {
                    self.stats_dirty = true;
                }
                ui.add_space(4.0);
                let (ow, oh) = crop.out_size();
                let kept = 100.0 * crop.area() / (w * h) as f64;
                ui.label(format!(
                    "Keeps {kept:.1}% of the pixels — {w}×{h} → {ow}×{oh}{}",
                    if crop.is_axis_aligned() {
                        String::new()
                    } else {
                        format!(" (tilted {:+.1}°, exported straightened)", crop.angle_deg)
                    }
                ));
                let n = data.n_frames();
                ui.label(format!(
                    "Stack in memory (f32): {} → {}",
                    human_bytes((n * w * h * 4) as f64),
                    human_bytes((n * ow * oh * 4) as f64),
                ));
                if !crop.is_axis_aligned() {
                    let bb = crop.bounding();
                    let out = bb.x0 < 0.0 || bb.y0 < 0.0 || bb.x1 > w as f32 || bb.y1 > h as f32;
                    ui.label(
                        egui::RichText::new(if out {
                            "⚠ The tilted region sticks out of the image: the pixels sampled \
                             outside come out as NaN (no data)."
                        } else {
                            "The region is resampled bilinearly on its own grid: every \
                             exported image is box width × box height pixels, straightened."
                        })
                        .small(),
                    );
                }
                if !data.orientation.is_identity() {
                    if let Some(c) = crop.as_crop(w, h) {
                        let disk = c.to_disk(data.orientation, w, h);
                        ui.label(
                            egui::RichText::new(format!(
                                "Saved in the on-disk frame ({}: {}): x={} y={} {}×{}",
                                data.detector.detector().label(),
                                data.orientation,
                                disk.x,
                                disk.y,
                                disk.width,
                                disk.height
                            ))
                            .small(),
                        )
                        .on_hover_text(
                            "The image is shown re-oriented for this detector; the crop JSON, the \
                             cropped stack and the exported images use the coordinates of the files \
                             as they are on disk.",
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!(
                                "Tilted region: center / size / angle are saved as drawn ({}: {}); \
                                 x, y, width, height carry its bounding box in the on-disk frame.",
                                data.detector.detector().label(),
                                data.orientation,
                            ))
                            .small(),
                        );
                    }
                }
            }
        }
        if let Some(init) = self.initial_crop {
            let (ix0, iy0) = (init.cx - init.width * 0.5, init.cy - init.height * 0.5);
            ui.label(
                egui::RichText::new(format!(
                    "Initial crop (dashed orange): x={:.0} y={:.0} {:.0}×{:.0}{}",
                    ix0,
                    iy0,
                    init.width,
                    init.height,
                    if init.angle_deg == 0.0 {
                        String::new()
                    } else {
                        format!(" @ {:+.1}°", init.angle_deg)
                    }
                ))
                .small()
                .color(INITIAL_CROP_COLOR),
            );
        }

        ui.separator();
        ui.heading("Verification");
        ui.label(
            egui::RichText::new(
                "Mean counts per image in a thin band just inside the crop edge. \
                 If the crop never cuts the sample, the edge-band curve stays flat; \
                 a dip (dark sample crossing the edge) or a spike in some images \
                 means the crop is too tight there. Click a point to inspect that \
                 image; also check the Min / Max projections.",
            )
            .small(),
        );
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.label("Edge band width:");
            let r = ui.add(
                egui::DragValue::new(&mut self.edge_band)
                    .speed(0.2)
                    .range(1..=200)
                    .suffix(" px"),
            );
            if r.changed() {
                self.stats_dirty = true;
            }
            let worst = self.stats.as_ref().and_then(|s| s.most_suspicious_frame());
            if let Some(idx) = worst {
                if ui
                    .button(format!("⚠ Most suspicious image ({idx})"))
                    .on_hover_text(
                        "Show the image whose edge-band value deviates the most \
                         from the median — the most likely place where the crop \
                         cuts something",
                    )
                    .clicked()
                {
                    self.show_frame(idx);
                }
            }
        });

        let Some(stats) = &self.stats else {
            if self.crop_box().is_none() {
                ui.label("Draw a crop region to see its per-image statistics.");
            }
            return;
        };

        let series: Vec<(&str, Color32, Vec<[f64; 2]>)> = [
            ("Crop edge band", EDGE_CURVE_COLOR, &stats.edge_mean),
            ("Inside crop", INSIDE_CURVE_COLOR, &stats.inside_mean),
            ("Outside crop", OUTSIDE_CURVE_COLOR, &stats.outside_mean),
        ]
        .map(|(name, color, values)| {
            let pts: Vec<[f64; 2]> = values
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, &v)| [i as f64, v])
                .collect();
            (name, color, pts)
        })
        .into_iter()
        .collect();

        let marker_x =
            (self.mode == DisplayMode::Single).then_some(self.frame_idx as f64);

        let plot = Plot::new("crop_stats_plot")
            .x_axis_label("file index")
            .y_axis_label("mean counts / pixel")
            .legend(Legend::default())
            .height(
                (panel_h - ui.min_rect().height() - ui.spacing().item_spacing.y).max(200.0),
            );

        let mut clicked_x: Option<f64> = None;
        plot.show(ui, |plot_ui| {
            if self.plot_refit {
                plot_ui.set_auto_bounds(true);
                self.plot_refit = false;
            }
            for (name, color, pts) in series {
                plot_ui.line(Line::new(name, PlotPoints::from(pts)).color(color));
            }
            if let Some(x) = marker_x {
                plot_ui.vline(
                    VLine::new("Displayed image", x)
                        .stroke(Stroke::new(1.5, Color32::from_gray(150)))
                        .style(LineStyle::dashed_loose()),
                );
            }
            if plot_ui.response().clicked() {
                clicked_x = plot_ui.pointer_coordinate().map(|p| p.x);
            }
        });
        if let Some(x) = clicked_x {
            let idx = x.round().clamp(0.0, (data.n_frames() - 1) as f64) as usize;
            self.show_frame(idx);
        }
    }

    /// Numeric editor for the crop: the top-left corner and size of the
    /// untilted rectangle (integer pixels) plus its tilt about the center.
    /// Returns whether anything changed.
    fn edit_crop_fields(&mut self, ui: &mut egui::Ui, crop: OrientedBox, w: usize, h: usize) -> bool {
        let (mut x, mut y) = (
            (crop.cx - crop.width * 0.5).round() as i64,
            (crop.cy - crop.height * 0.5).round() as i64,
        );
        let (mut cw, mut ch) = (crop.width.round() as i64, crop.height.round() as i64);
        let mut angle = crop.angle_deg;
        let mut changed = false;
        let tilted = crop.angle_deg != 0.0;
        let (max_w, max_h) = if tilted { ((w.max(h) * 2) as i64, (w.max(h) * 2) as i64) } else { (w as i64, h as i64) };
        let (min_xy, max_x, max_y) = if tilted {
            (-(w.max(h) as i64), (2 * w) as i64, (2 * h) as i64)
        } else {
            (0, w as i64 - 1, h as i64 - 1)
        };
        egui::Grid::new("crop_grid")
            .num_columns(4)
            .spacing([6.0, 2.0])
            .show(ui, |ui| {
                changed |= int_field(ui, "X", &mut x, min_xy, max_x);
                changed |= int_field(ui, "Width", &mut cw, 1, max_w);
                ui.end_row();
                changed |= int_field(ui, "Y", &mut y, min_xy, max_y);
                changed |= int_field(ui, "Height", &mut ch, 1, max_h);
                ui.end_row();
                ui.label("Angle");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut angle)
                            .speed(0.2)
                            .range(-180.0..=180.0)
                            .suffix("°")
                            .fixed_decimals(1),
                    )
                    .on_hover_text(
                        "Tilt of the crop region about its center, positive counter-clockwise; \
                         drag the round handle above the region to rotate it (Shift snaps to 5°)",
                    )
                    .changed();
                ui.end_row();
            });
        if changed {
            let (cw, ch) = (cw.max(1) as f32, ch.max(1) as f32);
            let b = OrientedBox {
                cx: x as f32 + cw * 0.5,
                cy: y as f32 + ch * 0.5,
                width: cw,
                height: ch,
                angle_deg: angle,
            };
            if let Some(b) = b.snapped(w, h) {
                // A direct numeric edit is undoable like a drag.
                self.push_undo();
                self.crop = Some(b);
            }
        }
        changed
    }

    fn viewer(&mut self, ui: &mut egui::Ui) {
        if let Some(job) = &self.loading {
            // A folder counts files; a .npy stack handed over by NeCTAR is
            // one file, so the text names its images and the bytes read.
            ui.centered_and_justified(|ui| {
                ui.add_sized(
                    [420.0, 24.0],
                    egui::ProgressBar::new(job.progress.fraction())
                        .show_percentage()
                        .text(format!("⏳ Loading {}", job.progress.text())),
                );
            });
            return;
        }

        let Some(data) = self.data.clone() else {
            ui.centered_and_justified(|ui| {
                ui.label("No folder loaded.");
            });
            return;
        };
        // The image viewport adapts to the window, but its minimum height
        // plus the image slider can outgrow a short window (small displays,
        // the large-text mode). The panel height is measured before entering
        // the scroll area — inside it the available height is unbounded —
        // and the viewport's minimum height below is what makes the scroll
        // bar appear.
        let panel_h = ui.available_height();
        egui::ScrollArea::vertical()
            .id_salt("viewer_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| self.viewer_content(ui, &data, panel_h));
    }

    fn viewer_content(&mut self, ui: &mut egui::Ui, data: &FolderData, panel_h: f32) {
        let (w, h) = (data.width, data.height);

        // In single-image mode a slider at the bottom picks the image on screen.
        let show_slider = self.mode == DisplayMode::Single && data.n_frames() > 1;
        let slider_height = if show_slider {
            ui.spacing().interact_size.y + ui.spacing().item_spacing.y * 2.0
        } else {
            0.0
        };

        // The viewport the image has to fit in: the panel minus the slider.
        // Dragging the divider (or resizing the window) changes it; while the
        // image is in the fitted state it follows, so the left side never
        // keeps a stale scale after the panel grows or shrinks.
        let viewport = egui::vec2(ui.available_width(), (panel_h - slider_height).max(1.0));
        if self.fitted && viewport != self.fitted_viewport {
            self.fit_requested = true;
        }
        if self.fit_requested && w > 0 && h > 0 {
            let s = (viewport.x / w as f32).min(viewport.y / h as f32);
            self.scale = s.clamp(0.02, 64.0);
            self.fit_requested = false;
            self.fitted = true;
            self.fitted_viewport = viewport;
        }

        let mut scroll = egui::ScrollArea::both()
            .max_height((panel_h - slider_height).max(120.0))
            .auto_shrink([false, false]);
        if let Some(offset) = self.viewer_scroll.take() {
            scroll = scroll.scroll_offset(offset);
        }
        // A Ctrl+wheel zoom over the image: (image x, image y under the
        // cursor, zoom factor). Applied after the scroll area reports its
        // current offset, so the next frame's scale and offset match.
        let mut wheel_zoom: Option<(f32, f32, f32)> = None;
        let out = scroll.show(ui, |ui| {
            let size = egui::vec2(w as f32 * self.scale, h as f32 * self.scale);
            let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());

            let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            let painter = ui.painter_at(rect);
            if let Some(t) = &self.img_tex {
                painter.image(t.id(), rect, full_uv, Color32::WHITE);
            }

            self.handle_interaction(&painter, rect, &response, w, h, &data);

            // egui turns Ctrl (Cmd on macOS) + wheel into a zoom factor
            // instead of a scroll; the pinch gesture arrives the same way.
            if response.contains_pointer() {
                let factor = ui.input(|i| i.zoom_delta());
                if factor != 1.0
                    && let Some(p) = response.hover_pos()
                {
                    let ix = (p.x - rect.left()) / self.scale;
                    let iy = (p.y - rect.top()) / self.scale;
                    wheel_zoom = Some((ix, iy, factor));
                }
            }
        });
        if let Some((ix, iy, factor)) = wheel_zoom {
            let old = self.scale;
            let new = (old * factor).clamp(0.02, 64.0);
            if new != old {
                // Keep the image point (ix, iy) under the cursor: the content
                // shifts by its distance to the origin times the scale change.
                let offset = out.state.offset + egui::vec2(ix, iy) * (new - old);
                self.viewer_scroll = Some(offset.max(egui::Vec2::ZERO));
                self.scale = new;
                self.fitted = false;
                ui.ctx().request_repaint();
            }
        }

        if show_slider {
            ui.horizontal(|ui| {
                let play_label = if self.playing { "⏸" } else { "▶" };
                if ui
                    .button(play_label)
                    .on_hover_text("Play through the stack to watch the crop against every image")
                    .clicked()
                {
                    self.playing = !self.playing;
                    self.last_advance = None;
                }
                ui.label("Image:");
                let max = data.n_frames() - 1;
                ui.style_mut().spacing.slider_width = (ui.available_width() - 160.0).max(100.0);
                if ui
                    .add(egui::Slider::new(&mut self.frame_idx, 0..=max).show_value(false))
                    .changed()
                {
                    self.playing = false;
                    self.img_dirty = true;
                }
                ui.label(format!("{} / {}", self.frame_idx + 1, data.n_frames()));
            });
        }
    }

    fn handle_interaction(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        response: &egui::Response,
        w: usize,
        h: usize,
        data: &FolderData,
    ) {
        // Set while the pointer is on (or dragging) the rotation handle: the
        // OS arrow is replaced by a rotation glyph painted at this position.
        let mut rotate_cursor: Option<Pos2> = None;
        let scale = self.scale;
        let to_img =
            |p: Pos2| -> (f32, f32) { ((p.x - rect.left()) / scale, (p.y - rect.top()) / scale) };
        let to_screen =
            |ix: f32, iy: f32| -> Pos2 { Pos2::new(rect.left() + ix * scale, rect.top() + iy * scale) };

        // Cursor read-out.
        self.cursor = None;
        if let Some(p) = response.hover_pos() {
            let (ix, iy) = to_img(p);
            let (xi, yi) = (ix.floor() as i64, iy.floor() as i64);
            if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                let v = self.display_image(data)[(yi as usize, xi as usize)];
                self.cursor = Some((xi as usize, yi as usize, v));
            }
        }

        // Image-space offset of the rotation handle above the crop's top edge
        // (constant on screen, whatever the zoom).
        let rot_offset = ROT_HANDLE_OFFSET / scale;
        let rot_handle_pos =
            |b: &OrientedBox| -> (f32, f32) { b.from_local(0.0, -b.height * 0.5 - rot_offset) };
        // Crop angle that puts the rotation handle at the given image point:
        // the handle sits along the box's -v axis, which is (-sin, -cos) of
        // the angle (positive angles are counter-clockwise on screen).
        let angle_to = |b: &OrientedBox, px: f32, py: f32| -> f32 {
            (-(px - b.cx)).atan2(-(py - b.cy)).to_degrees()
        };

        // Rotate, resize a handle, move the crop, or draw a new one
        // (replacing the old).
        if response.drag_started() {
            self.drag_changed = false;
            let press = painter
                .ctx()
                .input(|i| i.pointer.press_origin())
                .or_else(|| response.interact_pointer_pos());
            let cur = response.interact_pointer_pos().map(to_img);
            if let Some(sp) = press {
                let start = to_img(sp);
                let mut acted = false;

                if let Some(b) = self.crop {
                    let (rx, ry) = rot_handle_pos(&b);
                    // 1) The rotation handle.
                    if to_screen(rx, ry).distance(sp) <= HANDLE_HIT {
                        self.rotating = Some(b.angle_deg - angle_to(&b, start.0, start.1));
                        acted = true;
                    } else if let Some(hd) =
                        // 2) A resize handle of the crop.
                        crop_handles(&b).into_iter().find_map(|(hd, (ix, iy))| {
                            (to_screen(ix, iy).distance(sp) <= HANDLE_HIT).then_some(hd)
                        })
                    {
                        self.resizing = Some(hd);
                        acted = true;
                    } else if b.contains(start.0, start.1) {
                        // 3) Grab the crop to move it.
                        self.moving = true;
                        self.move_last = cur.or(Some(start));
                        acted = true;
                    }
                }

                // 4) Otherwise start drawing a new crop (snapshot now, so undo
                //    restores the previous one).
                if !acted {
                    self.push_undo();
                    self.drag_changed = true;
                    self.crop = Some(OrientedBox::from_rectf(RectF {
                        x0: start.0,
                        y0: start.1,
                        x1: start.0,
                        y1: start.1,
                    }));
                    self.drawing = true;
                    self.drag_start = Some(start);
                }
            }
        }

        if response.dragged() {
            let cur = response.interact_pointer_pos().map(to_img);
            // Snapshot once, on the first real movement of a move/resize/rotate.
            if (self.moving || self.resizing.is_some() || self.rotating.is_some())
                && !self.drag_changed
            {
                self.push_undo();
                self.drag_changed = true;
            }
            if let Some(offset) = self.rotating {
                rotate_cursor = response.interact_pointer_pos();
                if let Some(c) = cur {
                    let snap = painter.ctx().input(|i| i.modifiers.shift);
                    if let Some(b) = self.crop.as_mut() {
                        let mut a = angle_to(b, c.0, c.1) + offset;
                        if snap {
                            a = (a / ROT_SNAP_DEG).round() * ROT_SNAP_DEG;
                        }
                        b.angle_deg = a;
                        self.stats_dirty = true;
                    }
                }
            } else if let Some(hd) = self.resizing {
                if let (Some(c), Some(b)) = (cur, self.crop.as_mut()) {
                    resize_oriented(b, hd, c);
                    self.stats_dirty = true;
                }
            } else if self.moving {
                if let (Some(last), Some(c)) = (self.move_last, cur) {
                    if let Some(b) = self.crop.as_mut() {
                        b.translate(c.0 - last.0, c.1 - last.1);
                        self.stats_dirty = true;
                    }
                    self.move_last = cur;
                }
            } else if self.drawing {
                if let (Some(a), Some(c)) = (self.drag_start, cur) {
                    self.crop = Some(OrientedBox::from_rectf(RectF {
                        x0: a.0,
                        y0: a.1,
                        x1: c.0,
                        y1: c.1,
                    }));
                    self.stats_dirty = true;
                }
            }
        }

        if response.drag_stopped() {
            if self.drawing {
                // Drop a crop that never grew beyond a click.
                if let Some(a) = self.drag_start {
                    let c = response.interact_pointer_pos().map(to_img).unwrap_or(a);
                    let dist = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
                    if dist < 1.0 {
                        self.undo(); // restore the pre-drag crop
                    }
                }
            }
            self.drawing = false;
            self.moving = false;
            self.resizing = None;
            self.rotating = None;
            self.drag_changed = false;
            self.move_last = None;
            self.drag_start = None;
            self.snap_crop();
            self.stats_dirty = true;
        }

        // Hover cursor: rotate over the rotation handle, resize over a
        // handle, grab inside the crop.
        if !response.dragged() {
            if let (Some(hp), Some(b)) = (response.hover_pos(), self.crop) {
                let (rx, ry) = rot_handle_pos(&b);
                let over_handle = crop_handles(&b).into_iter().find_map(|(hd, (ix, iy))| {
                    (to_screen(ix, iy).distance(hp) <= HANDLE_HIT).then_some(hd)
                });
                if to_screen(rx, ry).distance(hp) <= HANDLE_HIT {
                    rotate_cursor = Some(hp);
                } else if let Some(hd) = over_handle {
                    painter.ctx().set_cursor_icon(cursor_for_handle(&b, hd));
                } else {
                    let (ix, iy) = to_img(hp);
                    if b.contains(ix, iy) {
                        painter.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }
                }
            }
        }

        // Delete/Backspace clears the crop (unless typing).
        if self.crop.is_some()
            && !painter.ctx().egui_wants_keyboard_input()
            && painter.ctx().input(|i| {
                i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)
            })
        {
            self.set_crop(None);
        }

        // Dim the region the crop throws away: darken the whole image, then
        // repaint the inside of the (possibly tilted) crop undimmed as a
        // textured quad — the corners' image coordinates are the UVs.
        if self.dim_outside
            && let (Some(b), Some(tex)) = (self.crop, &self.img_tex)
        {
            painter.rect_filled(rect, egui::CornerRadius::ZERO, Color32::from_black_alpha(140));
            let mut mesh = egui::Mesh::with_texture(tex.id());
            for (ix, iy) in b.corners() {
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: to_screen(ix, iy),
                    uv: Pos2::new(ix / w as f32, iy / h as f32),
                    color: Color32::WHITE,
                });
            }
            mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
            painter.add(mesh);
        }

        // Initial (previous-session) crop as a dashed reference outline.
        if let Some(init) = self.initial_crop
            && self.crop_box() != Some(init)
        {
            let corners = init.corners().map(|(ix, iy)| to_screen(ix, iy));
            draw_dashed_polygon(painter, &corners, INITIAL_CROP_COLOR);
        }

        // The crop itself, with the resize handles and the rotation handle.
        if let Some(b) = self.crop {
            let corners = b.corners().map(|(ix, iy)| to_screen(ix, iy));
            let stroke = Stroke::new(2.0, CROP_COLOR);
            for k in 0..4 {
                painter.line_segment([corners[k], corners[(k + 1) % 4]], stroke);
            }
            // Rotation handle: a stem from the top edge with a round knob.
            let top_mid = {
                let (ix, iy) = b.from_local(0.0, -b.height * 0.5);
                to_screen(ix, iy)
            };
            let knob = {
                let (ix, iy) = rot_handle_pos(&b);
                to_screen(ix, iy)
            };
            painter.line_segment([top_mid, knob], Stroke::new(1.5, CROP_COLOR));
            painter.circle_filled(knob, HANDLE_SIZE * 0.55, Color32::WHITE);
            painter.circle_stroke(knob, HANDLE_SIZE * 0.55, Stroke::new(1.5, Color32::BLACK));
            for (_hd, (ix, iy)) in crop_handles(&b) {
                let hr = Rect::from_center_size(to_screen(ix, iy), egui::vec2(HANDLE_SIZE, HANDLE_SIZE));
                painter.rect_filled(hr, egui::CornerRadius::ZERO, Color32::WHITE);
                painter.rect_stroke(
                    hr,
                    egui::CornerRadius::ZERO,
                    Stroke::new(1.0, Color32::BLACK),
                    egui::StrokeKind::Middle,
                );
            }
        }

        // No native rotation cursor exists, so hide the arrow and paint a
        // rotation glyph at the pointer instead (painted last, over the crop).
        if let Some(p) = rotate_cursor {
            painter.ctx().set_cursor_icon(egui::CursorIcon::None);
            let font = egui::FontId::proportional(20.0);
            for d in [
                egui::vec2(-1.0, 0.0),
                egui::vec2(1.0, 0.0),
                egui::vec2(0.0, -1.0),
                egui::vec2(0.0, 1.0),
            ] {
                painter.text(p + d, egui::Align2::CENTER_CENTER, "↻", font.clone(), Color32::BLACK);
            }
            painter.text(p, egui::Align2::CENTER_CENTER, "↻", font, Color32::WHITE);
        }
    }
}

// ----- helpers ----------------------------------------------------------------

/// The `<stem>_crop.json` sidecar written next to a cropped stack, recording
/// the crop region that was applied.
fn stack_sidecar_json(stack: &Path) -> PathBuf {
    let stem = stack
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("cropped_stack");
    stack.with_file_name(format!("{stem}_crop.json"))
}

fn folder_label(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

/// Min/max of the finite values; `(inf, -inf)` when there are none.
pub(crate) fn raw_range<I: Iterator<Item = f32>>(vals: I) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for v in vals {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo, hi)
}

pub(crate) fn normalize_range((lo, hi): (f32, f32)) -> (f32, f32) {
    if lo.is_finite() && hi.is_finite() {
        (lo, hi)
    } else {
        (0.0, 1.0)
    }
}

/// Number of values `robust_range` samples across the frames.
const AUTO_SAMPLES: usize = 1 << 20;
/// Sigma-clip width: the auto range is the clipped mean ± this many standard
/// deviations.
const AUTO_K: f64 = 3.0;
/// Never clip the auto range below this fraction of the sampled values.
const AUTO_MIN_COVER: f64 = 0.5;

/// Auto-contrast range by iterative sigma clipping: repeatedly narrow the
/// range to mean ± `AUTO_K`·σ of the values inside it until it stops moving.
/// A dense bulk plus a sparse tail — e.g. normalized 0–1 data whose failed
/// ratios reach 65535 — converges to the bulk, while well-behaved data
/// (Gaussian, uniform, bimodal) is left essentially untouched. Works on a
/// strided sample of the finite values; falls back to `full` when the sample
/// is too small or the range collapses.
pub(crate) fn robust_range(frames: &[Array2<f32>], full: (f32, f32)) -> (f32, f32) {
    use rayon::prelude::*;

    let per_frame = (AUTO_SAMPLES / frames.len().max(1)).max(1);
    let sample: Vec<f32> = frames
        .par_iter()
        .flat_map_iter(|f| match f.as_slice() {
            Some(s) => {
                let stride = (s.len() / per_frame).max(1);
                s.iter()
                    .copied()
                    .step_by(stride)
                    .filter(|v| v.is_finite())
                    .collect::<Vec<_>>()
            }
            None => f.iter().copied().filter(|v| v.is_finite()).collect(),
        })
        .collect();
    if sample.len() < 100 {
        return full;
    }
    let n = sample.len() as f64;
    let (mut lo, mut hi) = full;
    for _ in 0..40 {
        let (mut cnt, mut sum, mut sum2) = (0u64, 0f64, 0f64);
        for &v in &sample {
            if v >= lo && v <= hi {
                let v = v as f64;
                cnt += 1;
                sum += v;
                sum2 += v * v;
            }
        }
        if cnt < 100 {
            break;
        }
        let mean = sum / cnt as f64;
        let sigma = (sum2 / cnt as f64 - mean * mean).max(0.0).sqrt();
        let nlo = ((mean - AUTO_K * sigma) as f32).max(full.0);
        let nhi = ((mean + AUTO_K * sigma) as f32).min(full.1);
        if nlo >= nhi {
            break;
        }
        let inside = sample.iter().filter(|v| (nlo..=nhi).contains(v)).count();
        if (inside as f64) < AUTO_MIN_COVER * n {
            break;
        }
        if nlo == lo && nhi == hi {
            break;
        }
        lo = nlo;
        hi = nhi;
    }
    if lo < hi {
        (lo, hi)
    } else {
        full
    }
}

fn human_bytes(b: f64) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut v = b;
    let mut u = 0;
    while v >= 1000.0 && u < UNITS.len() - 1 {
        v /= 1000.0;
        u += 1;
    }
    format!("{v:.1} {}", UNITS[u])
}

fn int_field(ui: &mut egui::Ui, label: &str, v: &mut i64, min: i64, max: i64) -> bool {
    ui.label(label);
    ui.add(egui::DragValue::new(v).speed(0.5).range(min..=max))
        .changed()
}

/// Image-space positions of the 8 resize handles of the crop rectangle.
fn crop_handles(b: &OrientedBox) -> Vec<(Handle, (f32, f32))> {
    let (hw, hh) = (b.width * 0.5, b.height * 0.5);
    let mut out = Vec::with_capacity(8);
    for vy in [-1i8, 0, 1] {
        for hx in [-1i8, 0, 1] {
            if hx == 0 && vy == 0 {
                continue;
            }
            let p = b.from_local(hx as f32 * hw, vy as f32 * hh);
            out.push((Handle { hx, vy }, p));
        }
    }
    out
}

/// Drag a resize handle to the image point `p`: the grabbed edge(s) follow
/// the pointer in the crop's own frame, the opposite edge stays fixed in
/// image space, and the tilt is unchanged.
fn resize_oriented(b: &mut OrientedBox, handle: Handle, p: (f32, f32)) {
    let (lu, lv) = b.to_local(p.0, p.1);
    let (mut u0, mut u1) = (-b.width * 0.5, b.width * 0.5);
    let (mut v0, mut v1) = (-b.height * 0.5, b.height * 0.5);
    match handle.hx {
        -1 => u0 = lu,
        1 => u1 = lu,
        _ => {}
    }
    match handle.vy {
        -1 => v0 = lv,
        1 => v1 = lv,
        _ => {}
    }
    let (nu0, nu1) = (u0.min(u1), u0.max(u1));
    let (nv0, nv1) = (v0.min(v1), v0.max(v1));
    let (ncx, ncy) = b.from_local((nu0 + nu1) * 0.5, (nv0 + nv1) * 0.5);
    b.cx = ncx;
    b.cy = ncy;
    b.width = nu1 - nu0;
    b.height = nv1 - nv0;
}

/// The resize cursor matching the handle's outward direction on screen,
/// whatever the crop's tilt.
fn cursor_for_handle(b: &OrientedBox, handle: Handle) -> egui::CursorIcon {
    let (u, v) = b.axes();
    let dx = u.0 * handle.hx as f32 + v.0 * handle.vy as f32;
    let dy = u.1 * handle.hx as f32 + v.1 * handle.vy as f32;
    // 8 sectors of 45°, centered on the compass directions (0 = W).
    let sector = ((dy.atan2(dx).to_degrees() + 202.5).rem_euclid(360.0) / 45.0) as i32 % 8;
    match sector {
        0 | 4 => egui::CursorIcon::ResizeHorizontal, // W, E
        1 | 5 => egui::CursorIcon::ResizeNwSe,       // NW, SE
        2 | 6 => egui::CursorIcon::ResizeVertical,   // N, S
        _ => egui::CursorIcon::ResizeNeSw,           // NE, SW
    }
}

/// Dashed outline through the given screen corners.
fn draw_dashed_polygon(painter: &egui::Painter, corners: &[Pos2; 4], color: Color32) {
    let stroke = Stroke::new(2.0, color);
    for i in 0..4 {
        painter.add(egui::Shape::dashed_line(
            &[corners[i], corners[(i + 1) % 4]],
            stroke,
            6.0,
            4.0,
        ));
    }
}

impl eframe::App for CropApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_load();
        self.poll_stats();
        self.poll_saving(&ctx);
        self.maybe_spawn_stats(&ctx);
        self.animate(&ctx);
        if self.loading.is_some() {
            ctx.request_repaint();
        }

        self.ensure_img_texture(&ctx);
        self.instructions_modal(&ctx);
        self.batch_window(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            self.toolbar(ui);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            self.status_bar(ui);
        });
        // The crop panel opens at 38% of the window width; it stays resizable
        // from the divider, but never past half the window, so the image on
        // the left always keeps at least 50% of the view. (A stored panel
        // width from a wider window, or a divider dragged all the way left,
        // used to leave the viewer squeezed to nothing after a load.)
        let content_w = ctx.content_rect().width();
        egui::Panel::right("crop_panel")
            .resizable(true)
            .default_size(content_w * 0.38)
            .max_size(content_w * 0.5)
            .show(ui, |ui| {
                self.crop_panel(ui);
            });
        egui::CentralPanel::default().show(ui, |ui| {
            self.viewer(ui);
        });
    }
}

/// How the loaded frames relate to the files: the orientation applied to a
/// TIFF folder, or a note that a `.npy` stack is taken as it is (its frames
/// were oriented by whoever wrote it).
fn orientation_note(data: &loader::FolderData) -> String {
    if data.path.is_dir() {
        data.orientation.label().to_owned()
    } else {
        "frames already oriented (.npy stack)".to_owned()
    }
}
