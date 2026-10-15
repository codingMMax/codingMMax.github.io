# Repo memory — codingMMax.github.io

## Conventions / invariants
- **2026-10-14** `index.html` is *generated* from `config.json` by `build_local.py` (hero block at `build_local.py:564`). Any content change must be made in **both** places, or the next local build reverts the hand-edit.
- **2026-10-14** The four files in `.github/workflows/` (build-website, build-blog, deploy-waline, sync-scholar) are **not registered on GitHub** — `gh workflow list --all` shows only `pages-build-deployment`. So `.github/scripts/build-website.js` never runs in CI, and **the committed `index.html` is what Pages serves**. Editing only `config.json` publishes nothing; the generated HTML must be committed too.
- **2026-10-14** `images/dashu.jpg` does double duty: it is the Advent Lab **experience logo** (`index.html:187`, `config.json:241`). It is not avatar-only — never overwrite it in place when swapping the profile photo.
- **2026-10-14** Hero avatar renders as a **square** (`.profile-image` in `styles.css:132` — 260×260, `object-fit: cover`; 240px and 200px at the two mobile breakpoints). Source images must be square-cropped around the subject; a landscape photo gets center-cropped by the browser and lands on whatever is mid-frame.

## Image pipeline (this machine)
- **2026-10-14** No PIL, ImageMagick, or ffmpeg available to the system `python3`. Use macOS `sips`: crop with `sips -c H W --cropOffset Y X`, convert HEIC→JPEG with `-s format jpeg`, resize with `-Z`, and convert to sRGB with `-m "/System/Library/ColorSync/Profiles/sRGB Profile.icc"` (iPhone HEICs are Display P3).
- **2026-10-14** iPhone photos carry EXIF (GPS, device, timestamp) and XMP that `sips` preserves through conversion. Strip APP1+ segments before publishing — scratchpad helper `strip_jpeg_meta.py` (stdlib JPEG segment filter) does this.

## Settled decisions
- **2026-10-14** Profile photo is `images/profile.jpg` (800×800, sRGB, q80, metadata stripped) — a tight head-and-shoulders crop from a CarbonSet poster-session photo. Wider crops showing the full poster were considered and rejected: the face is illegible at 260px.
