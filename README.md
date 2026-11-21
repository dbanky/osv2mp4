# DJI Osmo 360 OSV to MP4 Converter

`osv2mp4` converts DJI Osmo 360 `.OSV` files into equirectangular 360° MP4, without DJI Studio.

An `.OSV` file holds the two raw fisheye images of the camera. Until they are converted into one panorama, no editor can use them. DJI Studio is the only official way to do that, and it has no batch mode and no command line. `osv2mp4` does the same job from a terminal, for one file or a whole folder tree.

```bash
osv2mp4 convert /path/to/footage/
```

Every `.OSV` under that folder gets a converted `.mp4` next to it, ready for Final Cut Pro, DaVinci Resolve, Premiere or YouTube.

## Why osv2mp4

- **No DJI Studio needed.** The conversion is done by this tool itself. Nothing from DJI has to be installed, opened or kept on screen.
- **Runs unattended.** It is a plain command, so it works in scripts, over SSH, on a server and in overnight batches. There is no window to click through and nothing that breaks when an app changes its layout.
- **Works on macOS and Linux.** On a Mac the whole chain runs on the GPU and the hardware encoder. On Linux it runs on the CPU.
- **One generation of compression, not two.** The fisheye images are decoded once and encoded once, at 10-bit, at the quality you choose.
- **Your audio stays as recorded.** The original track is copied bit for bit instead of being re-encoded at a lower bitrate.
- **The frame rate is preserved.** 29.97 fps footage stays 29.97 fps.
- **Whole folders in one go, resumable.** Point it at a card dump and it converts everything underneath. After an interruption, `--delta` continues where it stopped.
- **It tells you when something went wrong.** Each output is checked for frame count, frame rate, 360° metadata, audio and file layout. A bad file fails loudly instead of surfacing later in the edit.
- **Open and documented.** Every constant in the lens model and the stabiliser is explained, with how it was measured, in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Supported camera

**The first-generation DJI Osmo 360 only.** The lens model was measured on that camera. Files from any other model, including the Osmo 360 II, are refused by default, because they would convert with visible seams. `--any-model` overrides the check if you want to experiment.

Everything was validated on a single camera body. The lens constants are design values and should carry over to other first-generation bodies. If your seams look slightly off, run once with `--calibrate`, which measures the one per-body value from your own footage.

## What you get

- **Calibrated geometry.** The conversion uses the factory calibration stored in each file, with a lens model that stays accurate to the edge of the fisheye.
- **Clean seams.** The two lenses see nearby objects from slightly different positions. The converter measures that offset along both seams in every frame and corrects it before blending, so people and objects crossing a seam are not doubled.
- **Stabilisation.** The camera's own motion sensor data is used to level the horizon and smooth the heading. A 360° rotation has no crop, so this costs no picture.
- **10-bit throughout.** The picture is never reduced to 8-bit on the way.
- **Original audio.** The audio track is copied, not re-encoded.
- **A checked result.** Every conversion ends with a verification of the output file. The command exits non-zero if anything is wrong, such as a missing frame or missing 360° metadata.

## Install

You need Python 3.10 or newer and ffmpeg (version 6 or newer; 8 is recommended).

```bash
git clone <this repository>
cd osv2mp4
python3 -m venv .venv
.venv/bin/pip install -e .            # CPU only
.venv/bin/pip install -e '.[gpu]'     # with GPU acceleration (recommended on a Mac)
```

ffmpeg on macOS: `brew install ffmpeg`.

## Usage

```bash
# Convert one file, or every .OSV under a folder (output lands beside each source)
osv2mp4 convert clip.OSV
osv2mp4 convert /path/to/footage/ -o /path/to/output/

# Test five seconds first and look at the seams before committing hours
osv2mp4 convert clip.OSV --start 60 --duration 5

# Resume an interrupted batch: skip files that already have an output
osv2mp4 convert /path/to/footage/ --delta

# On a Mac: hardware encoding
osv2mp4 convert clip.OSV --encoder hevc_videotoolbox
```

Useful `convert` options:

| Option | Default | Meaning |
|---|---|---|
| `--width` | `7680` | Output width. Height is half of it. `5760` is much faster. |
| `--stabilize` | `smooth` | `smooth` levels the horizon and smooths the heading. `horizon` only levels. `lock` holds a fixed compass direction, for tripod-like shots. `none` keeps the camera's own tilt. |
| `--seam` | `flow` | `flow` corrects parallax at the seams. `static` is a plain cross-fade. |
| `--calibrate` | off | Measure the per-body rim scale from your footage. |
| `--crf` | `16` | Quality for the default software encoder. Lower is better. |
| `--encoder` | `libx265` | Use `hevc_videotoolbox` on a Mac. |
| `--delta` | off | Skip inputs whose output already exists. |

`osv2mp4 convert --help` lists the rest.

Other commands:

| Command | What it does |
|---|---|
| `inspect clip.OSV` | Print the streams, the camera identity and the decoded lens calibration. |
| `cover clip.OSV` | Extract the stitched preview JPEG the camera stores in the file. Instant. |
| `extract clip.OSV` | Save both fisheye images as a lossless MOV, unstitched. |
| `tag file.mp4` | Add 360° metadata to an existing equirectangular file. |
| `verify file.mp4` | Check that a file is a valid 360° video for Final Cut Pro. |
| `finalize exports/` | Fix up files exported by DJI Studio (metadata and file layout) so Final Cut Pro accepts them. |

## Speed

Converting 8K video is heavy. Measured on the same validation clip:

| Machine | Speed at 8K |
|---|---|
| Apple M4 Max, GPU and hardware encoder | about 12 frames per second |
| 16-core Linux machine, CPU only | about 0.7 to 1.5 frames per second |

On a CPU-only machine a 15-minute clip takes several hours. Test a short range first, or use `--width 5760`. For long runs on a Mac, run `caffeinate -disu` in another terminal so the machine does not sleep.

Peak memory at 8K is about 15 GB.

## Final Cut Pro

Final Cut Pro does not always recognise the 360° metadata on import. If the clip shows as a flat picture, select it and set, in the Info inspector, **360° Projection Mode** to **Equirectangular** and **Stereoscopic Conform** to **Monoscopic**.

## How it works

[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) has the full engineering notes: the file format, the lens model, how each constant was measured, and the approaches that failed.

## Tests

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest

# End-to-end test with a real clip
OSV2MP4_TEST_OSV=/path/to/clip.OSV .venv/bin/pytest
```

`tools/rotator_bench.py` cross-checks the GPU code against the CPU reference on your machine. `tools/transport_bench.py` measures how fast frames move between processes.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Copyright (C) 2026 Daniel Banky

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE) for the full text.

## Not affiliated with DJI

This is an independent project. DJI and Osmo are trademarks of their owner. The software does not contain or need any DJI code.
