# Recording the demo

The README's hero image is `demo/demo.gif`. Until you record it the image link
is broken — that's intentional, it's the reminder to ship the demo before
launching.

## What good looks like

- **25–35 seconds.** Long enough to show a couple of utterances, short enough
  that HN/Reddit users actually finish it.
- **One terminal, full screen, dark theme.** 1280×800 or 1440×900 capture
  window is ideal.
- **Font size 16–18 pt** so it stays readable when the GIF is 800–960 px wide.
- **Speaker audible** if you also produce an MP4 for YouTube. The GIF has no
  audio — make sure the on-screen text alone tells the story.
- **Show the moment Vietnamese turns into English.** That's the whole sale.

## Pre-recording checklist

1. Close noisy apps, mute notifications.
2. `./stream_translate.sh --help` once first to warm the bootstrap; you don't
   want pip install scrolling in the recording.
3. Pick a script from `script.md`. Read it once aloud before recording.
4. Resize your terminal to ~120 cols × 30 rows. Big enough to show partial +
   committed + translation, small enough that text is readable in the GIF.
5. Clear the screen (`clear`) so the recording starts on a clean prompt.

## Recording

**macOS** — QuickTime Player → File → New Screen Recording → choose mic →
record terminal window only → save as `demo/raw.mov`. Convert to mp4:

```bash
ffmpeg -i demo/raw.mov -c:v libx264 -crf 18 -preset slow \
       -c:a aac -b:a 128k demo/raw.mp4
```

**Linux (Wayland)** — `wf-recorder -g "$(slurp)" -f demo/raw.mp4` then read
the slurp prompt to drag the terminal area.

**Linux (X11)** — record full screen with ffmpeg:

```bash
ffmpeg -video_size 1280x800 -framerate 30 -f x11grab -i :0.0+0,0 \
       -f pulse -ac 1 -i default \
       -c:v libx264 -crf 18 -preset ultrafast \
       -c:a aac -b:a 128k demo/raw.mp4
```

Press `q` to stop.

**Any platform** — OBS Studio works everywhere; export H.264 mp4, drop in
`demo/raw.mp4`.

## Convert to GIF

```bash
./demo/make-gif.sh demo/raw.mp4 demo/demo.gif
```

Defaults: 960 px wide, 15 fps, ~3–5 MB. Tune by editing `make-gif.sh` if you
want it smaller or smoother.

## Also publish the MP4 (recommended)

GIFs are silent and low-fidelity. The MP4 with sound is the better experience
for anyone who clicks through. Upload `demo/raw.mp4` (or a cut of it) to
YouTube as **unlisted** first, watch it back, then make it public and add the
link to the README under the GIF.

## File hygiene

`raw.mp4` and `raw.mov` are git-ignored (see `.gitignore`). Only `demo.gif`
and any final cut you want to ship are tracked. Keep the repo lean.

## Quick sanity check before pushing

```bash
ls -lh demo/demo.gif       # under 8 MB, ideally under 5 MB
file demo/demo.gif         # should say "GIF image data"
git diff --stat HEAD~1     # no surprise files
```
