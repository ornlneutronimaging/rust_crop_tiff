# Crop TIFF

Native GUI (Rust, [egui](https://github.com/emilk/egui)) to pick a **single
rectangular crop region** on a stack of images (a folder of TIFFs or a `.npy`
stack) and export it as a small JSON file — and optionally the **cropped 3-D
stack itself** as `.npy` — so the calling application (e.g.
rust_ct_reconstruction or a marimo notebook) can work on the reduced stack and
drastically cut memory / processing time.

Started from [rust_tiff_viewer](../rust_tiff_viewer); companion tool for
rust_ct_reconstruction and the VENUS marimo notebooks, and usable standalone.

## Build

```bash
cargo build --release
# binary: target/release/crop_tiff
```

## Usage

```bash
# Standalone: open a folder or .npy stack from within the app
crop_tiff

# Display a folder of TIFF images
crop_tiff /SNS/VENUS/IPTS-XXXX/.../Run_YYYY

# Show a previous crop for review/editing
crop_tiff --crop 100,200,512,512 /path/to/run
crop_tiff --crop-file previous_crop.json /path/to/run

# Driven by another application (e.g. rust_ct_reconstruction) that hands its
# loaded stack over as a 3-D .npy and waits for the cropped stack back
crop_tiff stack.npy --called-from-app \
    --output crop.json --output-stack cropped_stack.npy \
    --instructions "Adjust the crop so the sample stays inside at every angle."
```

- **INPUT** — folder(s) of TIFF images (multi-page supported) and/or `.npy`
  stack files: a 2-D array is one image, a 3-D array one image per plane along
  axis 0 (the form used when another application hands its loaded stack over).
  More inputs can be added from within the application (**📁 Add folder…**,
  **🗋 Add .npy…**); a crop drawn on one input is kept when switching to
  another input with the same image size.
- **`-o, --output <PATH>`** — enables the **✅ Save crop & quit** button, which
  writes the crop JSON to `PATH` and closes the app.
- **`--output-stack <PATH.npy>`** — also write the **cropped 3-D stack** when
  saving/returning: NumPy `.npy`, `float32`, shape
  `(n_images, height, width)`. Written on a background thread (the stack can
  be GBs); the window closes when it is done. The **applied crop region always
  travels with the stack** — to `--output` when given, otherwise to a
  `<stem>_crop.json` sidecar next to the `.npy` — so the calling application
  can re-apply the same crop to another data set (e.g. via `--crop-file`).
- **`-c, --crop <X,Y,W,H>`** — initial crop region (e.g. the crop used last
  time), shown on the image at startup and kept as a dashed orange reference
  outline while you adjust the live (green) crop. **`--crop-file <PATH>`**
  reads the same thing from a JSON file written by a previous session.
- **`--called-from-app`** — the app is driven by another application that is
  blocked waiting for the crop: the save button reads **↩ Return to main
  application** instead (the rust_tof_profile_viewer convention). It writes
  the crop JSON to `--output` (and always also prints it on **stdout**, for a
  caller that captures the child's output), writes the cropped stack to
  `--output-stack` when given, and closes the window so the caller resumes.
  `--called-from-python` and `--called-from-marimo` are accepted as synonyms.
- **`--instructions <TEXT>`** — shows `TEXT` in a modal dialog at startup;
  reopen with the **ℹ Instructions** toolbar button.
- Without `--output`, use **💾 Save crop as…** / **🗋 Save cropped stack as…**
  to pick the destinations; saving a cropped stack interactively also writes
  its `<stem>_crop.json` sidecar.

## Output format

```json
{
  "x": 100,
  "y": 200,
  "width": 512,
  "height": 512,
  "image_width": 2048,
  "image_height": 2048,
  "folder": "/SNS/VENUS/IPTS-XXXX/.../Run_YYYY"
}
```

`x`/`y` is the top-left pixel and the region is `[x, x+width) × [y, y+height)`,
so in Python the crop is simply `frame[y:y+height, x:x+width]`. Only the four
`x/y/width/height` keys are read back by `--crop-file`; the rest records what
the crop was drawn on.

With `--output-stack`, the cropped stack itself is returned as a `.npy` file:
`float32`, shape `(n_images, height, width)` — in Rust readable with
`ndarray-npy`, in Python with `numpy.load`.

## Drawing the crop

Drag on the image to draw the rectangle (replacing the previous one), drag
inside it to move it, drag the 8 white handles to resize it, or type exact
pixel values in the **Crop region** panel. The crop snaps to whole pixels.
`Delete`/`Backspace` clears it, **↩ Undo** steps back, **⛶ Full image** selects
everything, **↧ Use initial crop** returns to the region passed on the command
line.

## Making sure the crop does not cut anything important

The whole point of this tool: a crop that looks fine on the integrated image
can still clip the sample in *some* of the images (e.g. at some rotation angle
of a CT scan). Several tools verify the crop against **every** image:

- **Min projection** — the darkest value each pixel ever takes. For
  transmission images this is the union of all sample silhouettes over the
  whole stack (a full CT rotation): if the dark envelope fits inside the crop,
  no image is cut.
- **Max projection** — the brightest value each pixel ever takes; bright
  features from any image all show at once.
- **Std-dev projection** — pixel-wise standard deviation across the stack;
  moving edges and changing regions light up.
- **Single image + ▶ play** — step or play through the stack with the crop
  overlaid, to eyeball every image.
- **Dim outside** — everything the crop throws away is darkened, so what would
  be lost is obvious at a glance.
- **Crop-edge statistics plot** — for every image, the mean counts in a thin
  band (configurable width) just inside the crop edge, plus the means inside
  and outside the crop. If the crop never cuts the sample, the edge-band curve
  stays flat at the open-beam level; a dip (dark sample crossing the edge) or
  a spike in some images means the crop is too tight there. Click a point to
  jump to that image, or use **⚠ Most suspicious image**, which jumps to the
  image whose edge-band value deviates the most from the median.

## Display

Modes: **Integrated** (pixel-wise sum), **Mean**, **Max**, **Min**, **Std
dev**, **Single image**. Contrast limits (with **Auto** reset per mode),
colormaps (Gray, Viridis, Inferno, Magma, Plasma, Cividis, Turbo, Jet), zoom
−/+/Fit. The status bar shows the cursor position/value; the crop panel shows
the fraction of pixels kept and the resulting stack size in memory.

## Tests

```bash
cargo test
```
