# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-21

First public release.

### Added

- `convert` command (alias `stitch`): converts first-generation DJI Osmo 360 `.OSV` files into equirectangular 360° HEVC MP4, for one file or a whole folder tree, with `--delta` to resume a batch.
- Lens geometry from the factory calibration stored in each file, with `--calibrate` to measure the per-body rim scale.
- Parallax correction along both seams (`--seam flow`).
- Stabilisation from the camera's motion sensor data: `--stabilize smooth`, `horizon`, `lock` or `none`.
- GPU path on Apple Silicon (Metal) and hardware encoding with `--encoder hevc_videotoolbox`.
- Automatic verification of every output file.
- A check that refuses files from other camera models unless `--any-model` is given.
- `inspect`, `cover`, `extract`, `tag`, `verify` and `finalize` commands.
- Engineering notes in `docs/HOW_IT_WORKS.md`.
