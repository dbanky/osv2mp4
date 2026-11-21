"""Per-body rim-scale calibration from far scenery along the seams."""

import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

from .constants import (
    DEFAULT_EDGE_SLOPE,
    DEFAULT_RADIUS_MULT,
    RADIUS_MULT_RANGE,
    SEAM_MEASURE_RES,
    SEAM_MEASURE_SIZE,
)
from .geometry import WORLD_ROT, project
from .seamflow import _gather, _lrf_for, _measure_grids, seam_centers
from .util import run


def _bilinear(img, x, y):
    h, w = img.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    fx, fy = x - x0c, y - y0c
    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
         + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    return v, ok


def seam_disagreement(lenses, frames, radius_mult, edge_slope):
    """Mean abs difference between the two lenses across the overlap band.

    Compared on texture only, after per-lens normalisation, so neither a flat
    white wall nor an exposure step between lenses can dominate the score.
    """
    lon = np.linspace(-np.pi, np.pi, 1500, endpoint=False)
    lat = np.linspace(-np.pi / 3, np.pi / 3, 40)
    LON, LAT = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT),
                     np.cos(LAT) * np.cos(LON)], axis=-1) @ WORLD_ROT.T

    vals, oks = [], []
    for lens, img in zip(lenses, frames):
        px, py, theta = project(lens, dirs, radius_mult, edge_slope)
        v, ok = _bilinear(img, px, py)
        vals.append(v)
        oks.append(ok & (theta >= np.radians(84)) & (theta <= np.radians(96)))

    both = oks[0] & oks[1]
    if both.sum() < 200:
        return None
    grad = (np.abs(np.gradient(vals[0], axis=1))
            + np.abs(np.gradient(vals[0], axis=0)))
    tex = both & (grad > 3.0)
    if tex.sum() < 200:
        tex = both
    a, b = vals[0][tex], vals[1][tex]
    a = (a - a.mean()) / max(a.std(), 1e-6)
    b = (b - b.mean()) / max(b.std(), 1e-6)
    return float(np.abs(a - b).mean())


def sample_frames(osv, vids, times):
    """Grab greyscale fisheye frames at the given timestamps."""
    tmp = tempfile.mkdtemp(prefix='osvfrm_')
    try:
        for t in times:
            frames = []
            for v in vids[:2]:
                p = os.path.join(tmp, f'f{v}_{t}.pgm')
                try:
                    run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', osv,
                         '-map', f'0:{v}', '-frames:v', '1', '-pix_fmt', 'gray',
                         '-y', p], capture_output=True)
                except subprocess.CalledProcessError:
                    frames = []
                    break
                frames.append(read_pgm(p))
            if len(frames) == 2:
                yield frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_pgm(path):
    with open(path, 'rb') as fh:
        assert fh.readline().strip() == b'P5'
        line = fh.readline()
        while line.startswith(b'#'):
            line = fh.readline()
        w, h = map(int, line.split())
        maxv = int(fh.readline())
        dt = '>u2' if maxv > 255 else 'u1'
        return np.frombuffer(fh.read(), dtype=dt).reshape(h, w).astype(np.float64)


def _far_seam_shift(lenses, yaw, mult, slope, imgs_list, rows_sel_fn):
    """Static longitude offset (deg) between the two lens views along each
    seam, from a brute-force horizontal shift search on the measurement
    band -- the same views measure_seam_flow uses. Returns [seam0, seam1].

    Zero-mean abs difference under integer shifts of -3..+3 deg; the rows
    are restricted by rows_sel_fn(lat_axis) to latitudes where the content
    is far (no real parallax), so a non-zero minimum is a geometry error,
    not the scene."""
    centers = seam_centers(lenses, yaw, mult, slope)
    lat_axis, grids = _measure_grids(lenses, yaw, mult, slope, centers)
    rows = rows_sel_fn(lat_axis)
    out = []
    for g in grids:
        acc = np.zeros(61)
        for imgs in imgs_list:
            v0 = _gather(g['lens'][0], imgs[0])[rows]
            v1 = _gather(g['lens'][1], imgs[1])[rows]
            both = g['both'][rows]
            for i, s in enumerate(range(-30, 31)):
                a = np.roll(v0, s, axis=1)
                m = both & np.roll(both, s, axis=1)
                m[:, :35] = False
                m[:, -35:] = False
                if m.sum() < 200:
                    acc[i] += 99.0
                    continue
                aa, bb = a[m], v1[m]
                acc[i] += np.abs((aa - aa.mean()) - (bb - bb.mean())).mean()
        out.append((int(np.argmin(acc)) - 30) * SEAM_MEASURE_RES)
    return out


def calibrate_radius_mult(osv, lenses, yaw, duration):
    """Measure the per-body rim scale (radius_mult) from the footage.

    Searches the radius scale at which far scenery lines up along both seams
    (see the DEFAULT_RADIUS_MULT comment for why this is the right knob: a
    scale error shows as opposite-sign offsets on the two seams). Reads the
    LRF proxy at three points in the clip and uses the mid-latitude band
    (+55..-50), where content is normally far. Returns the winner, or the
    default if the measurement is unavailable or hits a search bound.
    """
    lrf, lrf_idx = _lrf_for(osv)
    if lrf is None:
        print('  ! no .LRF proxy next to the clip; cannot calibrate the rim '
              'scale, keeping default', file=sys.stderr)
        return DEFAULT_RADIUS_MULT
    size = SEAM_MEASURE_SIZE
    imgs_list = []
    for t in (duration * 0.25, duration * 0.5, duration * 0.75):
        if t <= 1:
            continue
        try:
            buf = run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', lrf,
                       '-map', f'0:{lrf_idx}', '-frames:v', '1',
                       '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                      capture_output=True).stdout
        except subprocess.CalledProcessError:
            continue
        if len(buf) != 2 * size * size:
            continue
        fr = np.frombuffer(buf, np.uint8).reshape(size, 2 * size)
        imgs_list.append([fr[:, :size].astype(np.float32),
                          fr[:, size:].astype(np.float32)])
    if not imgs_list:
        print('  ! could not sample LRF frames; keeping default', file=sys.stderr)
        return DEFAULT_RADIUS_MULT

    def rows_fn(lat_axis):
        return (lat_axis <= 55.0) & (lat_axis >= -50.0)

    lo, hi = RADIUS_MULT_RANGE
    best, best_score = DEFAULT_RADIUS_MULT, None
    for mult in np.arange(lo, hi + 1e-9, 0.001):
        s0, s1 = _far_seam_shift(lenses, yaw, float(mult), DEFAULT_EDGE_SLOPE,
                                 imgs_list, rows_fn)
        score = abs(s0) + abs(s1)
        if best_score is None or score < best_score:
            best, best_score = float(mult), score
    if best <= lo + 1e-9 or best >= hi - 1e-9:
        print(f"  ! radius scale hit the search bound ({best:.3f}); "
              f"treating as unreliable, keeping default "
              f"{DEFAULT_RADIUS_MULT:.3f}", file=sys.stderr)
        return DEFAULT_RADIUS_MULT
    print(f"  calibrated radius scale x{best:.4f} "
          f"(far-scenery seam offset {best_score / 2:.2f} deg, "
          f"{len(imgs_list)} frames)")
    return best
