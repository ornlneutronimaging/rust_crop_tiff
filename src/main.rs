//! VENUS Crop TIFF — define a rectangular crop region on a folder of TIFF
//! images (e.g. a VENUS/Timepix measurement) so a calling application can
//! reduce the size of the stack before loading it. The crop is checked
//! against every image of the stack: integrated / mean / max / min / std-dev
//! projections, a single-image slider with play-through, and a per-image
//! edge-band statistics plot.

use rust_crop_tiff::app::CropApp;
use rust_crop_tiff::crop::CropRect;
use std::path::PathBuf;

const USAGE: &str = "\
crop_tiff — pick a rectangular crop region on a stack of images

Draw one rectangle on the image; the region (x, y, width, height) is saved as
JSON — and optionally the cropped 3-D stack itself as .npy — so the calling
application (e.g. rust_ct_reconstruction or a marimo notebook) can work on the
reduced stack. The crop can be checked against every image with the
Min/Max/Std projections, the single-image slider (with play-through), and the
per-image crop-edge statistics plot.

USAGE:
  crop_tiff [OPTIONS] [INPUT ...]

ARGS:
  INPUT   Folder(s) of TIFF images and/or .npy stack files (a 2-D array is
          one image, a 3-D array one image per plane along axis 0 — the form
          used when another application hands its loaded stack over). The
          displayed input is picked from the combobox at the top of the
          window; more inputs can be added from within the application.

OPTIONS:
  -o, --output <PATH>       Crop file written by the 'Save crop & quit'
                            button: JSON with x, y, width, height (top-left
                            corner and size, in pixels)
  --output-stack <PATH>     Also write the cropped 3-D stack to PATH when
                            saving/returning: NumPy .npy, float32, shape
                            (n_images, height, width). The applied crop
                            region always travels with the stack: it goes to
                            --output when given, otherwise to a
                            <stem>_crop.json sidecar next to the stack, so
                            the caller can re-apply the same crop to another
                            data set.
  -c, --crop <X,Y,W,H>      Initial crop region (e.g. a previous crop) shown
                            on the image at startup, e.g. 100,200,512,512
  --crop-file <PATH>        Read the initial crop region from a JSON file
                            written by a previous session
  --called-from-app         The app is driven by another application that is
                            blocked waiting for the crop: the save button
                            reads '↩ Return to main application', which
                            writes the crop to --output (and always also
                            prints it on stdout for the caller to capture),
                            writes the cropped stack to --output-stack when
                            given, and closes the window so the caller
                            resumes. (--called-from-python and
                            --called-from-marimo are accepted as synonyms.)
  --instructions <TEXT>     Instructions shown in a modal window on top of
                            the application at startup. Reopen any time with
                            the 'ℹ Instructions' toolbar button.
  -h, --help                Show this help
";

struct Args {
    inputs: Vec<PathBuf>,
    output: Option<PathBuf>,
    output_stack: Option<PathBuf>,
    initial_crop: Option<CropRect>,
    called_from_app: bool,
    instructions: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut inputs = Vec::new();
    let mut output = None;
    let mut output_stack: Option<PathBuf> = None;
    let mut initial_crop: Option<CropRect> = None;
    let mut called_from_app = false;
    let mut instructions = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "-o" | "--output" => {
                let path = args.next().ok_or("--output requires a path")?;
                output = Some(PathBuf::from(path));
            }
            "--output-stack" | "--output_stack" => {
                let path = args.next().ok_or("--output-stack requires a path")?;
                let path = PathBuf::from(path);
                if !path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("npy"))
                {
                    return Err(format!(
                        "--output-stack must end in .npy (got {})",
                        path.display()
                    ));
                }
                output_stack = Some(path);
            }
            "-c" | "--crop" => {
                let v = args.next().ok_or("--crop requires X,Y,WIDTH,HEIGHT")?;
                if initial_crop.is_some() {
                    return Err("only one of --crop / --crop-file can be given".to_owned());
                }
                initial_crop = Some(CropRect::parse_arg(&v).map_err(|e| format!("--crop: {e}"))?);
            }
            "--crop-file" | "--crop_file" => {
                let path = args.next().ok_or("--crop-file requires a path")?;
                if initial_crop.is_some() {
                    return Err("only one of --crop / --crop-file can be given".to_owned());
                }
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("cannot read --crop-file {path}: {e}"))?;
                initial_crop = Some(
                    CropRect::from_json_text(&text)
                        .map_err(|e| format!("--crop-file {path}: {e}"))?,
                );
            }
            "--called-from-app" | "--called_from_app" | "--called-from-python"
            | "--called_from_python" | "--called-from-marimo" | "--called_from_marimo" => {
                called_from_app = true
            }
            "--instructions" => {
                let text = args.next().ok_or("--instructions requires a text argument")?;
                instructions = Some(text);
            }
            s if s.starts_with('-') => return Err(format!("Unknown option: {s}")),
            _ => inputs.push(PathBuf::from(a)),
        }
    }
    for f in &inputs {
        if !rust_crop_tiff::loader::is_supported_input(f) {
            return Err(format!(
                "not a folder or .npy stack file: {}",
                f.display()
            ));
        }
    }
    if let Some(text) = &instructions {
        if text.trim().is_empty() {
            instructions = None;
        }
    }
    Ok(Args {
        inputs,
        output,
        output_stack,
        initial_crop,
        called_from_app,
        instructions,
    })
}

fn main() -> eframe::Result<()> {
    let args = match parse_args() {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("Error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 900.0])
            .with_title("VENUS Crop TIFF"),
        ..Default::default()
    };

    eframe::run_native(
        "VENUS Crop TIFF",
        native_options,
        Box::new(move |cc| {
            // Always use the dark theme, regardless of the system/desktop theme.
            cc.egui_ctx.set_theme(egui::Theme::Dark);
            let mut app = CropApp::new(
                args.inputs,
                args.initial_crop,
                args.output,
                args.output_stack,
                args.called_from_app,
                args.instructions,
            );
            app.select_first(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
}
