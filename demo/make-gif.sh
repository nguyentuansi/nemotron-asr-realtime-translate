#!/usr/bin/env bash
# Convert a screen recording (mp4/mov) into an optimized GIF for the README.
#
# Usage: ./demo/make-gif.sh [input] [output]
#   defaults: input=demo/raw.mp4  output=demo/demo.gif
#
# Tunables (edit below): WIDTH, FPS. Aim for <5 MB final GIF for README embed.
#
# Prefers gifski if installed (better quality, smaller files). Falls back to
# ffmpeg with palettegen, which is universally available.

set -euo pipefail

INPUT="${1:-demo/raw.mp4}"
OUTPUT="${2:-demo/demo.gif}"
WIDTH=960
FPS=15

if [[ ! -f "$INPUT" ]]; then
  echo "error: input not found: $INPUT" >&2
  echo "record a demo first; see demo/README.md" >&2
  exit 1
fi

echo "==> converting $INPUT -> $OUTPUT  (${WIDTH}px wide, ${FPS} fps)"

if command -v gifski >/dev/null 2>&1; then
  # gifski path: extract frames with ffmpeg, encode with gifski (best quality)
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "$TMPDIR"' EXIT
  ffmpeg -y -i "$INPUT" -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos" \
         "$TMPDIR/frame-%04d.png" -loglevel error
  gifski --fps "$FPS" --width "$WIDTH" --quality 90 \
         -o "$OUTPUT" "$TMPDIR"/frame-*.png
else
  # ffmpeg-only path: two-pass with palettegen
  PALETTE="$(mktemp --suffix=.png)"
  trap 'rm -f "$PALETTE"' EXIT
  ffmpeg -y -i "$INPUT" \
    -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos,palettegen=stats_mode=diff" \
    "$PALETTE" -loglevel error
  ffmpeg -y -i "$INPUT" -i "$PALETTE" \
    -lavfi "fps=$FPS,scale=$WIDTH:-1:flags=lanczos [v]; [v][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
    "$OUTPUT" -loglevel error
fi

SIZE=$(stat -c %s "$OUTPUT" 2>/dev/null || stat -f %z "$OUTPUT")
SIZE_MB=$(awk "BEGIN { printf \"%.2f\", $SIZE / 1048576 }")
echo "==> done: $OUTPUT ($SIZE_MB MB)"
if (( SIZE > 8 * 1048576 )); then
  echo "warning: GIF is over 8 MB. Trim the source, drop FPS to 12, or width to 800." >&2
fi
