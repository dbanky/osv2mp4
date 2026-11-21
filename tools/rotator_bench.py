#!/usr/bin/env python3
"""Benchmark and cross-check EquirectRotator backends on this machine.

Run on a Mac (or any machine with torch) from the repo root:
    python tools/rotator_bench.py [--width 7680] [--frames 5]

Renders a synthetic 10-bit frame through every available backend (Metal
kernel, generic torch on the GPU, torch CPU, numpy) with the same rotation,
prints ms per frame and the max/mean difference against numpy -- which is
the reference the v360 comparison in docs/HOW_IT_WORKS.md was made with. Anything above
1 LSB means a backend is wrong, not just slower.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))
from osv2mp4 import constants, djmd, geometry, gpu, stabilize, util  # noqa: E402


class _All:
    """Look a name up across the package's modules."""
    def __getattr__(self, name):
        for m in (gpu, geometry, stabilize, djmd, util, constants):
            if hasattr(m, name):
                return getattr(m, name)
        raise AttributeError(name)


c = _All()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=7680)
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--skip-cpu', action='store_true',
                    help='skip the slow torch-CPU / numpy reference at 8K')
    ap.add_argument('--from', dest='src', metavar='MP4',
                    help='use a real frame from this equirect MP4 instead of '
                         'the synthetic pattern (its size overrides --width)')
    ap.add_argument('--at', type=float, default=1.0,
                    help='seconds into --from / --osv (default 1.0)')
    ap.add_argument('--osv', metavar='OSV',
                    help='also check the GPU stitch kernel (remap x2 + '
                         'maskedmerge) on a real lens pair from this .OSV, '
                         'against the numpy reference -- must be exact')
    a = ap.parse_args()
    if a.osv:
        stitch_check(a.osv, a.at, a.width, a.frames)
    if a.src:
        import json
        import subprocess
        info = json.loads(subprocess.run(
            ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', a.src],
            capture_output=True, text=True, check=True).stdout)
        v = next(st for st in info['streams'] if st['codec_type'] == 'video')
        w, h = int(v['width']), int(v['height'])
        raw = subprocess.run(
            ['ffmpeg', '-v', 'error', '-ss', str(a.at), '-i', a.src, '-frames:v', '1',
             '-f', 'rawvideo', '-pix_fmt', 'gbrp10le', 'pipe:1'],
            capture_output=True, check=True).stdout
        frame = np.frombuffer(raw, np.uint16).reshape(3, h, w).copy()
        print(f"real frame {w}x{h} from {a.src} at {a.at}s")
    else:
        w, h = a.width, a.width // 2
        rng = np.random.default_rng(1)
        yy, xx = np.mgrid[0:h, 0:w]
        base = (300 + 200 * np.sin(xx / 97.0) * np.cos(yy / 61.0)
                + 150 * np.sin((xx + yy) / 23.0))
        frame = np.stack([np.clip(base + rng.integers(0, 60, (h, w)) + k * 40, 0, 1023)
                          for k in range(3)]).astype(np.uint16)
    T = c._euler_to_rot(12.3, 5.7, -3.1)
    M = c.stab_matrix(T)

    backends = []
    torch = c._torch()
    if torch is not None:
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            backends += [('mps', dict(metal=True)), ('mps', dict(metal=False))]
        if torch.cuda.is_available():
            backends += [('cuda', {})]
        if not a.skip_cpu:
            backends += [('cpu', {})]
    if not a.skip_cpu:
        backends += [('numpy', {})]

    ref = None
    results = []
    for dev, kw in backends:
        try:
            rot = c.EquirectRotator(w, h, dev, **kw)
        except Exception as e:      # noqa: BLE001
            print(f"{dev} {kw}: unavailable ({e})")
            continue
        out = rot.rotate_frame(frame, M)          # warm-up (Metal compile, allocations)
        rot.t_h2d = rot.t_rot = rot.t_d2h = 0.0
        t0 = time.perf_counter()
        for _ in range(a.frames):
            out = rot.rotate_frame(frame, M)
        dt = (time.perf_counter() - t0) / a.frames * 1000
        results.append((rot.describe(), dt, rot.timing(a.frames), out))
        print(f"{rot.describe():32s} {dt:7.1f} ms/frame  ({rot.timing(a.frames)})")
    # numpy (or the last backend) is the reference
    ref = results[-1][3]
    print()
    for name, dt, _, out in results:
        d = np.abs(out.astype(np.int32) - ref.astype(np.int32))
        print(f"{name:32s} vs {results[-1][0]}: max {d.max()} LSB, "
              f"mean {d.mean():.4f}, >1 LSB: {(d > 1).mean() * 100:.4f}%")
        bad = np.argwhere(d.max(axis=0) > 1)
        if len(bad):
            rows, cols = bad[:, 0], bad[:, 1]
            pole = np.sum((rows < 2) | (rows >= h - 2))
            edge = np.sum((cols < 2) | (cols >= w - 2))
            print(f"{'':32s}   {len(bad)} px >1 LSB: rows {rows.min()}-{rows.max()} "
                  f"(median {int(np.median(rows))}), {pole} in the 2 pole rows, "
                  f"{edge} in the 2 edge columns; worst at row {bad[d.max(axis=0)[bad[:,0], bad[:,1]].argmax()][0]}")


def stitch_check(osv, at, width, frames):
    import subprocess
    torch = c._torch()
    if torch is None or not torch.backends.mps.is_available():
        print("stitch check needs torch with an Apple GPU; skipped")
        return
    info = c.ffprobe_streams(osv)
    vids, auds, data = c.find_streams(info)
    lenses, dev = c.load_calibration(osv, data)
    yaw = c.default_yaw(lenses)
    w, h = width, width // 2
    xm, ym, mask = c.build_maps(lenses, w, h, c.DEFAULT_RADIUS_MULT, yaw,
                                c.DEFAULT_EDGE_SLOPE)
    L = next(int(s['width']) for s in info['streams'] if s['index'] == vids[0])
    raw = subprocess.run(
        ['ffmpeg', '-v', 'error', '-ss', str(at), '-i', osv, '-filter_complex',
         f'[0:{vids[0]}]format=gbrp10le[a];[0:{vids[1]}]format=gbrp10le[b];[a][b]vstack[v]',
         '-map', '[v]', '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gbrp10le', 'pipe:1'],
        capture_output=True, check=True).stdout
    lens = np.frombuffer(raw, np.uint16).reshape(3, 2 * L, L).copy()
    t0 = time.perf_counter()
    ref = c.stitch_frame_np(lens, xm, ym, mask, L)
    t_np = time.perf_counter() - t0
    st = c.GpuStitcher(xm, ym, mask, L, w, h)
    out = st.stitch(lens).cpu().numpy().view(np.uint16)       # warm-up
    st.t_upload = st.t_stitch = 0.0
    t0 = time.perf_counter()
    for _ in range(frames):
        out = st.stitch(lens).cpu().numpy().view(np.uint16)
    dt = (time.perf_counter() - t0) / frames * 1000
    d = np.abs(out.astype(np.int32) - ref.astype(np.int32))
    print(f"stitch {w}x{h} from {osv} at {at}s: Metal {dt:.1f} ms/frame "
          f"({st.timing(frames)} ms), numpy reference {t_np * 1000:.0f} ms")
    print(f"Metal stitch vs numpy reference: max {d.max()} LSB, exact "
          f"{(d == 0).mean() * 100:.3f}%  (must be 100%)")
    print()


if __name__ == '__main__':
    main()
