//! The egui/eframe application: folder combobox, image viewer with several
//! display modes (integrated sum, mean, max, min, std-dev, single image with
//! a slider and play button), a single rectangular crop region drawn/edited
//! on the image, and a verification panel plotting per-frame crop statistics
//! so no image of the stack loses important content.

use crate::colormap::Colormap;
use crate::crop::{CropRect, RectF};
use crate::loader::{self, FolderData};
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
    Progress { done: usize, total: usize },
    Done(anyhow::Result<FolderData>),
}

/// A folder load in flight on a background thread.
struct LoadJob {
    rx: Receiver<LoadMsg>,
    done: usize,
    total: usize,
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
    data_min: f32,
    data_max: f32,
    vmin: f32,
    vmax: f32,
    colormap: Colormap,
    img_tex: Option<TextureHandle>,
    img_dirty: bool,
    scale: f32,
    fit_requested: bool,
    cursor: Option<(usize, usize, f32)>,
    playing: bool,
    last_advance: Option<Instant>,
    dim_outside: bool,

    // Crop model.
    crop: Option<RectF>,
    /// Crop passed on the command line (a previous session's crop), kept as a
    /// dashed reference outline.
    initial_crop: Option<CropRect>,
    undo: Vec<Option<RectF>>,

    // Interaction transients.
    drawing: bool,
    moving: bool,
    resizing: Option<Handle>,
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

    status: String,
}

impl CropApp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inputs: Vec<PathBuf>,
        initial_crop: Option<CropRect>,
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
            data_min: 0.0,
            data_max: 1.0,
            vmin: 0.0,
            vmax: 1.0,
            colormap: Colormap::Viridis,
            img_tex: None,
            img_dirty: false,
            scale: 1.0,
            fit_requested: false,
            cursor: None,
            playing: false,
            last_advance: None,
            dim_outside: true,
            crop: None,
            initial_crop,
            undo: Vec::new(),
            drawing: false,
            moving: false,
            resizing: None,
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
            status,
        }
    }

    /// Start loading the first command-line input, if any.
    pub fn select_first(&mut self, ctx: &egui::Context) {
        if !self.inputs.is_empty() {
            self.select_input(0, ctx);
        }
    }

    /// The integer crop currently on the image, if any.
    fn crop_px(&self) -> Option<CropRect> {
        let data = self.data.as_ref()?;
        self.crop?.to_crop(data.width, data.height)
    }

    // ----- input loading -----------------------------------------------------

    fn select_input(&mut self, idx: usize, ctx: &egui::Context) {
        let Some(dir) = self.inputs.get(idx).cloned() else {
            return;
        };
        self.selected_input = Some(idx);
        self.status = format!("Loading {}…", folder_label(&dir));
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            // The parallel loader reports progress from worker threads, so the
            // sender goes behind a mutex.
            let progress_tx = Mutex::new(tx.clone());
            let result = loader::load_input_with_progress(&dir, |done, total| {
                if let Ok(tx) = progress_tx.lock() {
                    let _ = tx.send(LoadMsg::Progress { done, total });
                }
                ctx.request_repaint();
            });
            let _ = tx.send(LoadMsg::Done(result));
            ctx.request_repaint();
        });

        // Replacing an in-flight job drops its receiver; the stale thread's
        // sends fail silently and its result is discarded.
        self.loading = Some(LoadJob { rx, done: 0, total: 0 });
    }

    fn poll_load(&mut self) {
        let mut result = None;
        if let Some(job) = &mut self.loading {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    LoadMsg::Progress { done, total } => {
                        job.done = done;
                        job.total = total;
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

        // Keep the user's crop when the new folder has the same image size.
        let same_dims = self.data.as_ref().map(|d| (d.width, d.height)) == Some((w, h));
        if !same_dims {
            self.crop = None;
            self.undo.clear();
        }
        self.drawing = false;
        self.moving = false;
        self.resizing = None;
        self.playing = false;

        // Show the initial (previous-session) crop as the starting region.
        let mut crop_note = String::new();
        if self.crop.is_none() {
            if let Some(init) = self.initial_crop {
                match init.clamp_to(w, h) {
                    Some(c) => {
                        self.crop = Some(c.to_rectf());
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
        self.frame_idx = self.frame_idx.min(data.n_frames() - 1);
        self.apply_active_range();

        self.status = format!(
            "Loaded {} images ({w}×{h} px) from {}{crop_note}",
            data.n_frames(),
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
        let Some(crop) = self.crop_px() else {
            self.stats = None;
            return;
        };

        let (tx, rx) = std::sync::mpsc::channel();
        let data = Arc::clone(data);
        let band = self.edge_band.max(1);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let out = stats::compute(&data.frames, &data.frame_totals, crop, band);
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

    fn set_crop(&mut self, crop: Option<RectF>) {
        if self.crop != crop {
            self.push_undo();
            self.crop = crop;
            self.stats_dirty = true;
        }
    }

    /// Snap the live rectangle to whole pixels (as saved); drop it when it is
    /// degenerate, restoring the pre-drag state.
    fn snap_crop(&mut self) {
        let Some(data) = &self.data else { return };
        if let Some(r) = self.crop {
            match r.to_crop(data.width, data.height) {
                Some(c) => self.crop = Some(c.to_rectf()),
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

    /// Reset the contrast limits to the full range of the displayed mode.
    fn apply_active_range(&mut self) {
        let (lo, hi) = self.ranges[self.mode.index()];
        self.data_min = lo;
        self.data_max = hi;
        self.vmin = lo;
        self.vmax = hi;
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
        let (Some(data), Some(crop)) = (self.data.clone(), self.crop_px()) else {
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
                    loader::write_cropped_stack(stack_path, &data.frames, crop)
                        .map_err(|e| format!("Failed to write {}: {e:#}", stack_path.display()))?;
                    notes.push(format!("cropped stack → {}", stack_path.display()));
                }
                let json = crop.to_json(data.width, data.height, &data.path.display().to_string());
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

    fn save_crop_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
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
        let Some(path) = rfd::FileDialog::new()
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
            if ui.button("Auto").clicked() {
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
                    let full = CropRect::full(data.width, data.height).to_rectf();
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
                        match init.clamp_to(data.width, data.height) {
                            Some(c) => self.set_crop(Some(c.to_rectf())),
                            None => {
                                self.status =
                                    "Initial crop is outside this image — not applied.".to_owned()
                            }
                        }
                    }
                }
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
            ui.label("Zoom:");
            if ui.button("−").clicked() {
                self.scale = (self.scale / 1.25).max(0.02);
            }
            if ui.button("+").clicked() {
                self.scale = (self.scale * 1.25).min(64.0);
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
                let can_save = self.crop_px().is_some() && self.saving_rx.is_none();

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
                    .add_enabled(can_save, egui::Button::new("💾 Save crop as…"))
                    .on_hover_text("Write the crop region (x, y, width, height) to a JSON file")
                    .clicked()
                {
                    self.save_crop_dialog();
                }
                if ui
                    .add_enabled(can_save, egui::Button::new("🗋 Save cropped stack as…"))
                    .on_hover_text(
                        "Write the cropped 3-D stack as a NumPy .npy file \
                         (float32, shape (n_images, height, width))",
                    )
                    .clicked()
                {
                    self.save_stack_dialog();
                }
                if self.saving_rx.is_some() {
                    ui.spinner();
                }
                ui.separator();
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.label(&self.status);
                    if let Some((x, y, v)) = self.cursor {
                        ui.separator();
                        ui.label(format!("({x}, {y}) = {v:.4}"));
                    }
                });
            });
        });
    }

    /// The right-hand panel: crop coordinates, size summary, and the per-frame
    /// verification plot.
    fn crop_panel(&mut self, ui: &mut egui::Ui) {
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

        match self.crop_px() {
            None => {
                ui.label("Drag a rectangle on the image to define the crop.");
            }
            Some(crop) => {
                if self.edit_crop_fields(ui, crop, w, h) {
                    self.stats_dirty = true;
                }
                ui.add_space(4.0);
                let kept = 100.0 * crop.area() as f64 / (w * h) as f64;
                ui.label(format!(
                    "Keeps {kept:.1}% of the pixels — {w}×{h} → {}×{}",
                    crop.width, crop.height
                ));
                let n = data.n_frames();
                ui.label(format!(
                    "Stack in memory (f32): {} → {}",
                    human_bytes((n * w * h * 4) as f64),
                    human_bytes((n * crop.area() * 4) as f64),
                ));
            }
        }
        if let Some(init) = self.initial_crop {
            ui.label(
                egui::RichText::new(format!(
                    "Initial crop (dashed orange): x={} y={} {}×{}",
                    init.x, init.y, init.width, init.height
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
            if self.crop_px().is_none() {
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
            .height(ui.available_height().max(200.0));

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

    /// Numeric editor for the crop (integer pixels). Returns whether anything
    /// changed.
    fn edit_crop_fields(&mut self, ui: &mut egui::Ui, crop: CropRect, w: usize, h: usize) -> bool {
        let (mut x, mut y) = (crop.x as i64, crop.y as i64);
        let (mut cw, mut ch) = (crop.width as i64, crop.height as i64);
        let mut changed = false;
        egui::Grid::new("crop_grid")
            .num_columns(4)
            .spacing([6.0, 2.0])
            .show(ui, |ui| {
                changed |= int_field(ui, "X", &mut x, 0, w as i64 - 1);
                changed |= int_field(ui, "Width", &mut cw, 1, w as i64);
                ui.end_row();
                changed |= int_field(ui, "Y", &mut y, 0, h as i64 - 1);
                changed |= int_field(ui, "Height", &mut ch, 1, h as i64);
                ui.end_row();
            });
        if changed {
            let c = CropRect {
                x: x as usize,
                y: y as usize,
                width: cw.max(1) as usize,
                height: ch.max(1) as usize,
            };
            if let Some(c) = c.clamp_to(w, h) {
                // A direct numeric edit is undoable like a drag.
                self.push_undo();
                self.crop = Some(c.to_rectf());
            }
        }
        changed
    }

    fn viewer(&mut self, ui: &mut egui::Ui) {
        if let Some(job) = &self.loading {
            let frac = if job.total > 0 {
                job.done as f32 / job.total as f32
            } else {
                0.0
            };
            ui.centered_and_justified(|ui| {
                ui.add_sized(
                    [320.0, 24.0],
                    egui::ProgressBar::new(frac)
                        .show_percentage()
                        .text(format!("⏳ Loading {} / {} files", job.done, job.total)),
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
        let (w, h) = (data.width, data.height);

        // In single-image mode a slider at the bottom picks the image on screen.
        let show_slider = self.mode == DisplayMode::Single && data.n_frames() > 1;
        let slider_height = if show_slider {
            ui.spacing().interact_size.y + ui.spacing().item_spacing.y * 2.0
        } else {
            0.0
        };

        if self.fit_requested && w > 0 && h > 0 {
            let avail = ui.available_size();
            let s = (avail.x / w as f32).min((avail.y - slider_height).max(1.0) / h as f32);
            self.scale = s.clamp(0.02, 64.0);
            self.fit_requested = false;
        }

        egui::ScrollArea::both()
            .max_height((ui.available_height() - slider_height).max(0.0))
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let size = egui::vec2(w as f32 * self.scale, h as f32 * self.scale);
                let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());

                let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                let painter = ui.painter_at(rect);
                if let Some(t) = &self.img_tex {
                    painter.image(t.id(), rect, full_uv, Color32::WHITE);
                }

                self.handle_interaction(&painter, rect, &response, w, h, &data);
            });

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

        // Resize a handle, move the crop, or draw a new one (replacing the old).
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

                if let Some(r) = self.crop {
                    // 1) A resize handle of the crop.
                    if let Some(hd) = crop_handles(&r).into_iter().find_map(|(hd, (ix, iy))| {
                        (to_screen(ix, iy).distance(sp) <= HANDLE_HIT).then_some(hd)
                    }) {
                        self.resizing = Some(hd);
                        acted = true;
                    } else if r.contains(start.0, start.1) {
                        // 2) Grab the crop to move it.
                        self.moving = true;
                        self.move_last = cur.or(Some(start));
                        acted = true;
                    }
                }

                // 3) Otherwise start drawing a new crop (snapshot now, so undo
                //    restores the previous one).
                if !acted {
                    self.push_undo();
                    self.drag_changed = true;
                    self.crop = Some(RectF {
                        x0: start.0,
                        y0: start.1,
                        x1: start.0,
                        y1: start.1,
                    });
                    self.drawing = true;
                    self.drag_start = Some(start);
                }
            }
        }

        if response.dragged() {
            let cur = response.interact_pointer_pos().map(to_img);
            // Snapshot once, on the first real movement of a move/resize.
            if (self.moving || self.resizing.is_some()) && !self.drag_changed {
                self.push_undo();
                self.drag_changed = true;
            }
            if let Some(hd) = self.resizing {
                if let (Some(c), Some(r)) = (cur, self.crop.as_mut()) {
                    resize_rect(r, hd, c);
                    self.stats_dirty = true;
                }
            } else if self.moving {
                if let (Some(last), Some(c)) = (self.move_last, cur) {
                    if let Some(r) = self.crop.as_mut() {
                        r.translate(c.0 - last.0, c.1 - last.1);
                        self.stats_dirty = true;
                    }
                    self.move_last = cur;
                }
            } else if self.drawing {
                if let (Some(a), Some(c)) = (self.drag_start, cur) {
                    self.crop = Some(RectF {
                        x0: a.0,
                        y0: a.1,
                        x1: c.0,
                        y1: c.1,
                    });
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
            self.drag_changed = false;
            self.move_last = None;
            self.drag_start = None;
            self.snap_crop();
            self.stats_dirty = true;
        }

        // Hover cursor: resize over a handle, grab inside the crop.
        if !response.dragged() {
            if let (Some(hp), Some(r)) = (response.hover_pos(), self.crop) {
                let over = crop_handles(&r).into_iter().find_map(|(hd, (ix, iy))| {
                    (to_screen(ix, iy).distance(hp) <= HANDLE_HIT).then_some(hd)
                });
                if let Some(hd) = over {
                    painter.ctx().set_cursor_icon(cursor_for_handle(hd));
                } else {
                    let (ix, iy) = to_img(hp);
                    if r.contains(ix, iy) {
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

        // Dim the region the crop throws away.
        if self.dim_outside {
            if let Some(r) = self.crop.map(RectF::normalized) {
                let c = Rect::from_min_max(to_screen(r.x0, r.y0), to_screen(r.x1, r.y1))
                    .intersect(rect);
                let dim = Color32::from_black_alpha(140);
                if c.is_positive() {
                    let zero = egui::CornerRadius::ZERO;
                    painter.rect_filled(
                        Rect::from_min_max(rect.min, Pos2::new(rect.right(), c.top())),
                        zero,
                        dim,
                    );
                    painter.rect_filled(
                        Rect::from_min_max(Pos2::new(rect.left(), c.bottom()), rect.max),
                        zero,
                        dim,
                    );
                    painter.rect_filled(
                        Rect::from_min_max(
                            Pos2::new(rect.left(), c.top()),
                            Pos2::new(c.left(), c.bottom()),
                        ),
                        zero,
                        dim,
                    );
                    painter.rect_filled(
                        Rect::from_min_max(
                            Pos2::new(c.right(), c.top()),
                            Pos2::new(rect.right(), c.bottom()),
                        ),
                        zero,
                        dim,
                    );
                }
            }
        }

        // Initial (previous-session) crop as a dashed reference outline.
        if let Some(init) = self.initial_crop {
            let differs = self.crop_px() != Some(init);
            if differs {
                draw_dashed_rect(
                    painter,
                    to_screen(init.x as f32, init.y as f32),
                    to_screen(init.x1() as f32, init.y1() as f32),
                    INITIAL_CROP_COLOR,
                );
            }
        }

        // The crop itself, with handles.
        if let Some(r) = self.crop.map(RectF::normalized) {
            let screen = Rect::from_min_max(to_screen(r.x0, r.y0), to_screen(r.x1, r.y1));
            painter.rect_stroke(
                screen,
                egui::CornerRadius::ZERO,
                Stroke::new(2.0, CROP_COLOR),
                egui::StrokeKind::Middle,
            );
            for (_hd, (ix, iy)) in crop_handles(&r) {
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
fn raw_range<I: Iterator<Item = f32>>(vals: I) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for v in vals {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo, hi)
}

fn normalize_range((lo, hi): (f32, f32)) -> (f32, f32) {
    if lo.is_finite() && hi.is_finite() {
        (lo, hi)
    } else {
        (0.0, 1.0)
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
fn crop_handles(r: &RectF) -> Vec<(Handle, (f32, f32))> {
    let n = r.normalized();
    let (midx, midy) = ((n.x0 + n.x1) * 0.5, (n.y0 + n.y1) * 0.5);
    let mut out = Vec::with_capacity(8);
    for vy in [-1i8, 0, 1] {
        for hx in [-1i8, 0, 1] {
            if hx == 0 && vy == 0 {
                continue;
            }
            let x = match hx {
                -1 => n.x0,
                1 => n.x1,
                _ => midx,
            };
            let y = match vy {
                -1 => n.y0,
                1 => n.y1,
                _ => midy,
            };
            out.push((Handle { hx, vy }, (x, y)));
        }
    }
    out
}

fn resize_rect(r: &mut RectF, handle: Handle, p: (f32, f32)) {
    let mut n = r.normalized();
    match handle.hx {
        -1 => n.x0 = p.0,
        1 => n.x1 = p.0,
        _ => {}
    }
    match handle.vy {
        -1 => n.y0 = p.1,
        1 => n.y1 = p.1,
        _ => {}
    }
    *r = n;
}

fn cursor_for_handle(handle: Handle) -> egui::CursorIcon {
    match (handle.hx, handle.vy) {
        (0, _) => egui::CursorIcon::ResizeVertical,
        (_, 0) => egui::CursorIcon::ResizeHorizontal,
        (hx, vy) if hx == vy => egui::CursorIcon::ResizeNwSe,
        _ => egui::CursorIcon::ResizeNeSw,
    }
}

/// Dashed rectangle outline between two screen corners.
fn draw_dashed_rect(painter: &egui::Painter, a: Pos2, b: Pos2, color: Color32) {
    let stroke = Stroke::new(2.0, color);
    let r = Rect::from_two_pos(a, b);
    let corners = [
        [r.left_top(), r.right_top()],
        [r.right_top(), r.right_bottom()],
        [r.right_bottom(), r.left_bottom()],
        [r.left_bottom(), r.left_top()],
    ];
    for edge in corners {
        painter.add(egui::Shape::dashed_line(&edge, stroke, 6.0, 4.0));
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

        egui::Panel::top("toolbar").show(ui, |ui| {
            self.toolbar(ui);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            self.status_bar(ui);
        });
        // The crop panel opens at 38% of the window width; it stays resizable
        // from the divider.
        egui::Panel::right("crop_panel")
            .resizable(true)
            .default_size(ctx.content_rect().width() * 0.38)
            .show(ui, |ui| {
                self.crop_panel(ui);
            });
        egui::CentralPanel::default().show(ui, |ui| {
            self.viewer(ui);
        });
    }
}
