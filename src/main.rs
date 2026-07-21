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
crop_tiff — pick a rectangular crop region on a stack of TIFF images

Draw one rectangle on the image; the region (x, y, width, height) is saved as
JSON so the calling application (e.g. a marimo notebook) can crop the full
stack before loading it. The crop can be checked against every image with the
Min/Max/Std projections, the single-image slider (with play-through), and the
per-image crop-edge statistics plot.

USAGE:
  crop_tiff [OPTIONS] [FOLDER ...]

ARGS:
  FOLDER  Folder(s) containing TIFF images. The displayed folder is picked
          from the combobox at the top of the window; more folders can be
          added from within the application.

OPTIONS:
  -o, --output <PATH>       Crop file written by the 'Save crop & quit'
                            button: JSON with x, y, width, height (top-left
                            corner and size, in pixels)
  -c, --crop <X,Y,W,H>      Initial crop region (e.g. a previous crop) shown
                            on the image at startup, e.g. 100,200,512,512
  --crop-file <PATH>        Read the initial crop region from a JSON file
                            written by a previous session
  --called-from-python      The app is driven by another application that is
                            blocked waiting for the crop: the save button
                            reads '↩ Return to main application', which
                            writes the crop to --output (required) and closes
                            the window so the caller resumes.
  --instructions <TEXT>     Instructions shown in a modal window on top of
                            the application at startup. Reopen any time with
                            the 'ℹ Instructions' toolbar button.
  -h, --help                Show this help
";

struct Args {
    folders: Vec<PathBuf>,
    output: Option<PathBuf>,
    initial_crop: Option<CropRect>,
    called_from_python: bool,
    instructions: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut folders = Vec::new();
    let mut output = None;
    let mut initial_crop: Option<CropRect> = None;
    let mut called_from_python = false;
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
            "--called-from-python" | "--called_from_python" => called_from_python = true,
            "--instructions" => {
                let text = args.next().ok_or("--instructions requires a text argument")?;
                instructions = Some(text);
            }
            s if s.starts_with('-') => return Err(format!("Unknown option: {s}")),
            _ => folders.push(PathBuf::from(a)),
        }
    }
    for f in &folders {
        if !f.is_dir() {
            return Err(format!("not a folder: {}", f.display()));
        }
    }
    if called_from_python && output.is_none() {
        return Err(
            "--called-from-python requires --output <PATH> (where the crop is returned)".to_owned(),
        );
    }
    if let Some(text) = &instructions {
        if text.trim().is_empty() {
            instructions = None;
        }
    }
    Ok(Args {
        folders,
        output,
        initial_crop,
        called_from_python,
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
                args.folders,
                args.initial_crop,
                args.output,
                args.called_from_python,
                args.instructions,
            );
            app.select_first(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
}
