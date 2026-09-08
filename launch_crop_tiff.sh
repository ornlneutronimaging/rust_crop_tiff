#!/usr/bin/env bash
# Launch the VENUS Crop TIFF GUI, rebuilding first if the sources changed.
#
# Usage: ./launch_crop_tiff.sh [crop_tiff arguments...]
#   e.g. ./launch_crop_tiff.sh --crop 100,200,512,512 /SNS/VENUS/IPTS-1234/shared/run_1
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$REPO_DIR/target/release/crop_tiff"

# GUI apps need a display (e.g. a ThinLinc session).
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "Error: no display found (DISPLAY/WAYLAND_DISPLAY unset)." >&2
    echo "Run this from a graphical session such as ThinLinc." >&2
    exit 1
fi

# Rebuild if the binary is missing or any source/manifest file is newer.
needs_build=false
if [[ ! -x "$BINARY" ]]; then
    needs_build=true
elif [[ -n "$(find "$REPO_DIR/src" "$REPO_DIR/Cargo.toml" "$REPO_DIR/../rust_detector_orientation/src" -newer "$BINARY" -print -quit 2>/dev/null)" ]]; then
    needs_build=true
fi

if $needs_build; then
    CARGO="$(command -v cargo || true)"
    [[ -z "$CARGO" && -x "$HOME/.cargo/bin/cargo" ]] && CARGO="$HOME/.cargo/bin/cargo"
    if [[ -n "$CARGO" && -w "$REPO_DIR/target" ]]; then
        echo "Building crop_tiff (release)..."
        (cd "$REPO_DIR" && "$CARGO" build --release)
        # cargo does not relink (or touch) an already up-to-date binary, so
        # advance its mtime explicitly — otherwise the staleness check above
        # keeps firing forever, locking out users who cannot rebuild.
        touch "$BINARY"
    elif [[ -x "$BINARY" ]]; then
        # Regular users have no cargo and no write access to the repo: a
        # possibly stale binary is better than none. The owner's next
        # launch rebuilds it.
        echo "Warning: sources are newer than the binary but it cannot be" \
             "rebuilt here (cargo missing or target/ not writable);" \
             "launching the existing binary." >&2
    else
        echo "Error: no binary found and it cannot be built here" \
             "(cargo missing or target/ not writable)." >&2
        exit 1
    fi
fi

exec "$BINARY" "$@"
