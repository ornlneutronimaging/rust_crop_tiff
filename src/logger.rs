//! Append-only application log in the shared VENUS log directory
//! (`/SNS/VENUS/shared/log/crop_tiff_<user>.log`), following the
//! `<tool>_<user>.log` naming and Python-logging line format
//! (`2026-07-20 09:47:14,967 - INFO - message`) used by the other imaging
//! tools that log there, so the log browser picks it up.
//!
//! Logging is best-effort: if the file cannot be opened (reported once by
//! [`init`]), later calls are silently dropped so the GUI keeps working.

use chrono::Local;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

pub const LOG_DIR: &str = "/SNS/VENUS/shared/log";

static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();

pub fn user_id() -> String {
    std::env::var("USER").unwrap_or_else(|_| "user".to_owned())
}

/// The machine this process runs on (`bl10-analysis1`, ...), so a log line
/// can be matched to the right host when a user reports a problem.
pub fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "unknown host".to_owned())
}

/// `/SNS/VENUS/shared/log/crop_tiff_<user>.log`
pub fn log_path() -> PathBuf {
    PathBuf::from(LOG_DIR).join(format!("crop_tiff_{}.log", user_id()))
}

/// Open (append) the log file and write the session-start line. Call once at
/// startup; the error is the only chance to see why logging is unavailable.
pub fn init() -> Result<PathBuf, String> {
    let path = log_path();
    let mut open_error = None;
    SINK.get_or_init(
        || match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => Some(Mutex::new(file)),
            Err(e) => {
                open_error = Some(format!("cannot open {}: {e}", path.display()));
                None
            }
        },
    );
    if let Some(e) = open_error {
        return Err(e);
    }
    if SINK.get().is_some_and(|s| s.is_none()) {
        return Err(format!("cannot open {}", path.display()));
    }
    log(format!(
        "=== Application started (user: {}, host: {}, pid: {}) ===",
        user_id(),
        hostname(),
        std::process::id()
    ));
    Ok(path)
}

fn write_line(level: &str, msg: &str) {
    if let Some(mutex) = SINK.get().and_then(|s| s.as_ref())
        && let Ok(mut file) = mutex.lock()
    {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S,%3f");
        let _ = writeln!(file, "{ts} - {level} - {msg}");
        let _ = file.flush();
    }
}

pub fn log(msg: impl AsRef<str>) {
    write_line("INFO", msg.as_ref());
}

pub fn warn(msg: impl AsRef<str>) {
    write_line("WARNING", msg.as_ref());
}

pub fn error(msg: impl AsRef<str>) {
    write_line("ERROR", msg.as_ref());
}
