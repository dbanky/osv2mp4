"""Seam flow: per-frame parallax measurement along both seams, Kalman-filtered, streamed into the remap maps."""

import os
import queue
import subprocess
import sys
import threading
import time
import numpy as np

from .constants import (
    FADE_HALF_DEG,
    FOV_HALF_DEG,
    KALMAN_CONF_FLOOR,
    KALMAN_PROC_SIGMA,
    KALMAN_R0,
    SEAM_FLOW2D_DY_MAX,
    SEAM_FLOW2D_STORE_STEP,
    SEAM_FLOW_EDGE_ERODE,
    SEAM_FLOW_LAT_MAX,
    SEAM_FLOW_MAX_DEG,
    SEAM_FLOW_MAX_LON_DEG,
    SEAM_FLOW_NADIR_FADE,
    SEAM_FLOW_RAMP_HALF,
    SEAM_FLOW_TAPER,
    SEAM_FLOW_WIN_DEG,
    SEAM_MEASURE_HALF_LON,
    SEAM_MEASURE_RES,
    SEAM_MEASURE_SIZE,
)
from .geometry import WORLD_ROT, project
from .util import ffprobe_streams


def fps_float(fps):
    if '/' in str(fps):
        num, den = str(fps).split('/')
        return float(num) / float(den)
    return float(fps)


def seam_centers(lenses, yaw, mult, slope):
    """Output longitudes (deg) where the two seams sit, for any yaw."""
    lons = np.radians(np.arange(-180.0, 180.0, 0.25) + yaw)
    dirs = np.stack([np.sin(lons), np.zeros_like(lons), np.cos(lons)],
                    axis=-1) @ WORLD_ROT.T
    _, _, theta = project(lenses[0], dirs, mult, slope)
    t = np.degrees(theta) - 90.0
    centers = []
    for i in range(len(t)):
        j = (i + 1) % len(t)
        if t[i] == 0 or (t[i] < 0) != (t[j] < 0):
            frac = abs(t[i]) / max(abs(t[i]) + abs(t[j]), 1e-9)
            centers.append(-180.0 + (i + frac) * 0.25)
    return centers[:2] if len(centers) >= 2 else [-90.0, 90.0]


def _measure_grids(lenses, yaw, mult, slope, centers):
    """Precompute the bilinear gather tables for both seam bands at
    measurement scale, one grid per seam."""
    lat_axis = np.arange(SEAM_FLOW_LAT_MAX, -SEAM_FLOW_LAT_MAX - 1e-9,
                         -SEAM_MEASURE_RES)
    a0 = np.radians(90 - FADE_HALF_DEG)
    a1 = np.radians(90 + FADE_HALF_DEG)
    grids = []
    for lon_c in centers:
        lons = np.arange(lon_c - SEAM_MEASURE_HALF_LON,
                         lon_c + SEAM_MEASURE_HALF_LON + 1e-9, SEAM_MEASURE_RES)
        LON, LAT = np.meshgrid(np.radians(lons + yaw), np.radians(lat_axis))
        dirs = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT),
                         np.cos(LAT) * np.cos(LON)], axis=-1) @ WORLD_ROT.T
        per_lens, per_lens_theta, weights, valids = [], [], [], []
        for lens in lenses:
            px, py, theta = project(lens, dirs, mult, slope)
            s = SEAM_MEASURE_SIZE / lens['width']
            px, py = px * s, py * s
            valid = ((theta < np.radians(FOV_HALF_DEG)) & (px >= 0) & (py >= 0)
                     & (px <= SEAM_MEASURE_SIZE - 1)
                     & (py <= SEAM_MEASURE_SIZE - 1))
            x0 = np.clip(np.floor(px).astype(np.int32), 0, SEAM_MEASURE_SIZE - 2)
            y0 = np.clip(np.floor(py).astype(np.int32), 0, SEAM_MEASURE_SIZE - 2)
            per_lens.append((x0, y0, (px - x0).astype(np.float32),
                             (py - y0).astype(np.float32), valid))
            per_lens_theta.append(theta)
            weights.append(np.where(valid, np.clip((a1 - theta) / (a1 - a0),
                                                   0, 1), 0.0))
            valids.append(valid)
        m = weights[1] / np.maximum(weights[0] + weights[1], 1e-12)
        band = (m > 0.02) & (m < 0.98) & valids[0] & valids[1]
        both = valids[0] & valids[1]
        # per-row equidistance column (theta0 == theta1): the natural cut
        # line the routed seam is anchored to and biased toward
        tdiff = np.abs(np.degrees(per_lens_theta[0] - per_lens_theta[1]))
        tdiff = np.where(both, tdiff, 1e9)
        eq_col = np.argmin(tdiff, axis=1)
        grids.append({'lens': per_lens, 'band': band, 'both': both,
                      'eq_col': eq_col})
    return lat_axis, grids


def _gather(tab, img):
    x0, y0, fx, fy, valid = tab
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return np.where(valid, v, 0.0)


def _flow_max_lon(lat_deg):
    """Per-latitude clamp on the seam offset, in longitude degrees."""
    c = np.maximum(np.cos(np.radians(lat_deg)), 1e-3)
    return np.minimum(SEAM_FLOW_MAX_DEG / c, SEAM_FLOW_MAX_LON_DEG)




def kalman_flow_smooth(meas, conf, dt, proc_sigma=KALMAN_PROC_SIGMA,
                       r0=KALMAN_R0, conf_floor=KALMAN_CONF_FLOOR):
    """Random-walk Kalman filter + RTS smoother along time (axis 0).

    Reproduces DJI Studio's KalmanFlowFilter on the seam flow. Each element of
    the trailing dimensions is an independent scalar signal (a per-latitude
    offset or a per-pixel flow component) modelled as a random walk with
    per-frame process variance (proc_sigma*dt)^2 and measured each frame with
    noise r0/max(conf, conf_floor). Low-confidence measurements are averaged
    down to a steady value (far scenery -> ~0, no waving); confident ones are
    followed. The RTS backward pass makes it zero-lag. Returns the smoothed
    signal, same shape.

    ns (time) is the small axis; everything is vectorised over the E elements.
    """
    ns = meas.shape[0]
    tail = meas.shape[1:]
    m = meas.reshape(ns, -1).astype(np.float32)
    c = conf.reshape(ns, -1).astype(np.float32)
    E = m.shape[1]
    if ns < 3 or E == 0:
        return meas
    Q = (proc_sigma * dt) ** 2

    xf = np.empty((ns, E), np.float32)      # filtered mean
    Pf = np.empty((ns, E), np.float32)      # filtered variance
    Pp = np.empty((ns, E), np.float32)      # predicted variance (P_{t}^-)

    x = m[0].copy()
    P = np.full(E, 1.0, np.float32)
    for t in range(ns):
        if t > 0:
            P = P + Q                       # predict (random walk: x unchanged)
        Pp[t] = P
        R = r0 / np.maximum(c[t], conf_floor)
        K = P / (P + R)
        x = x + K * (m[t] - x)
        P = (1 - K) * P
        xf[t] = x
        Pf[t] = P

    # RTS backward smoother (F = 1, so C_t = Pf_t / Pp_{t+1})
    xs = xf.copy()
    for t in range(ns - 2, -1, -1):
        C = Pf[t] / np.maximum(Pp[t + 1], 1e-12)
        xs[t] = xf[t] + C * (xs[t + 1] - xf[t])   # xp_{t+1} == xf_t for RW
    return xs.reshape((ns,) + tail).astype(np.float32)


_DIS = None       # lazily built OpenCV flow + CLAHE objects (see _deep_flow)
_CLAHE = None


def _deep_flow(v0, v1, g, row_sel, max_dx, max_dy):
    """Dense 2-D flow between the two lens views over the deep-zone rows.

    Dense flow v0 -> v1 in measurement-grid samples; the warp needs
    d = p0 - p1 (same convention the 1-D field uses), which is -flow.
    Invalid pixels (outside either lens) carry zero flow; magnitudes are
    clamped per row (the horizontal limit grows as 1/cos(lat), the vertical
    one is physical). Returned decimated by SEAM_FLOW2D_STORE_STEP.
    """
    import cv2
    a = np.clip(v0[row_sel], 0, 255).astype(np.uint8)
    b = np.clip(v1[row_sel], 0, 255).astype(np.uint8)
    # DIS (dense inverse search) flow on locally contrast-equalised views.
    # Farneback was used first and returned an exact zero on the dark,
    # low-contrast ground rows below ~-60 lat (mean ~50, std ~7 on the 8-bit
    # LRF), so the nadir cut got no parallax shift there and a foot on the
    # cut line was duplicated (a detached heel). DIS's variational
    # refinement recovers a consistent ground parallax there (~1.3-1.7 deg
    # physical, growing toward the nadir, opposite sign on the two seams --
    # the expected geometry) and is ~3x faster; CLAHE takes out the per-lens
    # rim vignetting so brightness differences are not chased as motion.
    global _DIS, _CLAHE
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        _DIS.setUseSpatialPropagation(True)
        _CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    flow = _DIS.calc(_CLAHE.apply(a), _CLAHE.apply(b), None)
    d = -flow                          # p0 - p1, [rows, cols, (dx, dy)]
    # Only trust flow well inside the region both lenses cover: right at a
    # lens's validity edge one view goes black and Farneback fills the window
    # with a large, clamp-hitting bogus flow that then leaks into the warp.
    # Erode the mask by roughly the flow window's half-width.
    both = g['both'][row_sel]
    if SEAM_FLOW_EDGE_ERODE > 1:
        k = np.ones((SEAM_FLOW_EDGE_ERODE,) * 2, np.uint8)
        both = cv2.erode(both.astype(np.uint8), k).astype(bool)
    d[~both] = 0.0
    d[..., 0] = np.clip(d[..., 0], -max_dx[:, None], max_dx[:, None])
    d[..., 1] = np.clip(d[..., 1], -max_dy, max_dy)
    s = SEAM_FLOW2D_STORE_STEP
    return d[::s, ::s].astype(np.float16)


def _lrf_for(src):
    """The .LRF proxy next to an .OSV, if its frame geometry matches.

    The LRF is the two fisheyes side by side at 1024x1024 each (verified
    corr 1.000 against the downscaled main streams), so the measurement can
    decode it instead of the two 3840x3840 HEVC streams -- seconds vs minutes.
    """
    for cand in (os.path.splitext(src)[0] + ext for ext in ('.LRF', '.lrf')):
        if not os.path.exists(cand):
            continue
        try:
            info = ffprobe_streams(cand)
        except subprocess.CalledProcessError:
            continue
        v = next((s for s in info['streams']
                  if s.get('codec_type') == 'video'
                  and s.get('codec_name') != 'mjpeg'), None)
        if v and v.get('width') == 2 * SEAM_MEASURE_SIZE \
                and v.get('height') == SEAM_MEASURE_SIZE:
            return cand, v['index']
    return None, None


def measure_seam_flow(src, vids, lenses, yaw, mult, slope, fps_f,
                      start, dur, n_out, step, smooth_s):
    """Per-frame per-latitude seam offset profiles for the whole clip.

    Returns (lat_bins descending, profiles[n_out, 2, nbins] in degrees,
    seam centre longitudes), or None if measurement was not possible. The
    sign convention is fixed by the warp: lens 0 is sampled at
    x + d*m and lens 1 at x - d*(1-m), which the still-frame prototype
    verified reduces the band disagreement (the opposite sign increases it).
    """
    centers = seam_centers(lenses, yaw, mult, slope)
    lat_axis, grids = _measure_grids(lenses, yaw, mult, slope, centers)

    lrf, lrf_idx = _lrf_for(src)
    size = SEAM_MEASURE_SIZE
    sel = f"select='not(mod(n\\,{step}))'"
    cmd = ['ffmpeg', '-v', 'error']
    if start:
        cmd += ['-ss', str(start)]
    cmd += ['-i', lrf if lrf else src]
    if dur:
        cmd += ['-t', str(dur)]
    if lrf:
        # LRF proxy: the two fisheyes side by side at measurement size already
        cmd += ['-map', f'0:{lrf_idx}', '-vf', sel]
    else:
        # fall back to the main streams, stacked so one rawvideo pipe carries
        # both (the rawvideo muxer takes a single stream)
        cmd += ['-filter_complex',
                f"[0:{vids[0]}]{sel},scale={size}:{size}[a];"
                f"[0:{vids[1]}]{sel},scale={size}:{size}[b];"
                "[a][b]hstack[v]", '-map', '[v]']
    cmd += ['-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'gray',
            '-']

    # Dense 2-D flow now covers the WHOLE seam band, not just the nadir. A
    # per-latitude 1-D offset shears the image -- each latitude shifts
    # independently, so a vertical edge is sheared and, as the offset changes
    # over time, the shear ripples (the temporal "waving" at the seam). Dense
    # 2-D flow moves content as a coherent field, so edges stay straight and
    # only translate, and the Kalman temporal filter keeps that translation
    # steady. This mirrors DJI Studio, which runs dense optical flow along the
    # whole seam rather than a 1-D field.
    # Rows: from just above the zenith taper down to the nadir limit (see the
    # SEAM_FLOW_LAT_MAX comment for why the band is asymmetric).
    row_sel = ((lat_axis <= SEAM_FLOW_TAPER[1] + 2.0)
               & (lat_axis >= -SEAM_FLOW_LAT_MAX))
    lat_deep = lat_axis[row_sel]
    max_dx = _flow_max_lon(lat_deep) / SEAM_MEASURE_RES     # samples
    max_dy = SEAM_FLOW2D_DY_MAX / SEAM_MEASURE_RES
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("  ! OpenCV missing; seam flow disabled "
              "(pip install opencv-python-headless)", file=sys.stderr)
        return None

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = 2 * size * size
    raw_f = []
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if buf is None or len(buf) < frame_bytes:
                break
            fr = np.frombuffer(buf, np.uint8).reshape(size, 2 * size)
            imgs = [fr[:, :size].astype(np.float32),
                    fr[:, size:].astype(np.float32)]
            raw_f.append([_deep_flow(_gather(g['lens'][0], imgs[0]),
                                     _gather(g['lens'][1], imgs[1]),
                                     g, row_sel, max_dx, max_dy)
                          for g in grids])
            if len(raw_f) % 500 == 0:
                print(f"    measured {len(raw_f)} samples...")
    finally:
        proc.stdout.close()
        proc.wait()
    if len(raw_f) < 2:
        return None

    ns = len(raw_f)
    lat_bins = np.arange(SEAM_FLOW_LAT_MAX, -SEAM_FLOW_LAT_MAX - 1e-9, -1.0)
    dt = step / fps_f

    # dense 2-D flow over the whole seam band: [ns, 2, H, W, 2].
    flow = np.stack([np.stack(f, axis=0) for f in raw_f], axis=0).astype(np.float32)

    # latitude taper: fade the flow to zero toward the zenith, where the lens
    # overlap shrinks and the flow is unreliable (sky needs no seam warp);
    # keep it at full strength through the nadir zone and only fade in the
    # last few degrees before the measured limit so the warp has no hard edge.
    t0, t1 = SEAM_FLOW_TAPER
    lat_t = np.clip((t1 - lat_deep) / (t1 - t0), 0.0, 1.0)
    lat_t *= np.clip((lat_deep + SEAM_FLOW_LAT_MAX) / SEAM_FLOW_NADIR_FADE,
                     0.0, 1.0)
    s = SEAM_FLOW2D_STORE_STEP
    flow *= lat_t[None, None, ::s, None, None]

    # temporal: Kalman-filter every pixel's flow over time -- far/low-texture
    # pixels (near-zero, low confidence) settle to a steady value so nothing
    # waves; confident close parallax is followed without lag. Magnitude is the
    # confidence proxy. `smooth_s` scales the process noise (larger = steadier).
    conf2d = np.clip(np.abs(flow) / (0.4 / SEAM_MEASURE_RES), 0.12, 1.0)
    flow = kalman_flow_smooth(flow, conf2d, dt,
                              proc_sigma=KALMAN_PROC_SIGMA * 0.4 / max(smooth_s, 0.1))
    flow = flow.astype(np.float16)

    ncols_full = grids[0]['both'].shape[1]
    flow_meta = {
        'data': flow, 'step': step,
        'lats': lat_deep[::s].copy(),      # descending, whole band
        'lons_rel': ((np.arange(flow.shape[3]) * s)
                     - (ncols_full - 1) / 2.0) * SEAM_MEASURE_RES,
    }
    return lat_bins, None, centers, flow_meta




class SeamFlowWriter:
    """Streams per-frame warped remap maps into ONE fifo for ffmpeg.

    A producer thread folds the dense 2-D seam flow into the maps and sends
    only the seam window columns; the writer thread keeps one persistent
    full-frame buffer holding all four maps stacked vertically (x0, y0, x1,
    y1 -- 4*out_h rows), pastes the windows in and writes it, so no
    full-frame copies ever sit in a queue (the full-copy version of this
    pipeline was what pushed 8K encodes into the OOM killer).

    Why ONE stream and not four: with four fifos (one per map) ffmpeg's
    graph scheduler is free to pull lens-0 map frames a few frames ahead of
    lens-1's. Once that drift exceeds the slack in the chain (2-packet demux
    queue + the writer queue) it deadlocks: ffmpeg blocks reading the next
    lens-0 frame, the lens-1 writers block on fifos ffmpeg is not draining,
    and the producer blocks on the full lens-1 queue -- nobody can send the
    sentinel. Seen at the second-to-last frame of two concurrent 3840
    renders and again at frame 6595/6596 of a solo 8K render (recovered by
    hand-feeding one throwaway frame into the starved fifo). One stacked
    frame per video frame, split+cropped inside ffmpeg, makes the four maps
    inseparable and the cycle impossible.
    """

    def __init__(self, work, xm, ym, mask, lat_bins, prof, centers,
                 out_w, out_h, n_frames, flow_meta=None, workers=None,
                 fifo=True, frame_offset=0):
        # fifo=False: no ffmpeg stream at all -- next_windows() hands each
        # frame's windows to the caller (the GPU stitch pastes them itself).
        # frame_offset: this writer's frame 0 is that frame of the measured
        # flow (segment jobs).
        self.j0 = frame_offset
        self.fifo = os.path.join(work, 'maps') if fifo else None
        if fifo:
            os.mkfifo(self.fifo)
        # _frame_windows is ~150 ms of numpy per 8K frame single-threaded --
        # at 4 fps that alone is most of the frame period. numpy releases
        # the GIL in its gathers and arithmetic, so a few workers computing
        # consecutive frames overlap well; results are re-ordered before
        # they reach the writer. Each in-flight result is ~33 MB.
        self.workers = workers or max(1, min(4, (os.cpu_count() or 4) // 4))
        self.t_produce = self.t_paste = self.t_write = self.t_idle = 0.0
        self.n_written = 0
        self.xm, self.ym = xm, ym
        self.out_w, self.out_h = out_w, out_h
        self.flow = flow_meta
        self.n_frames = n_frames
        self.stop = threading.Event()
        # Keep this SMALL. The queue depth is what throttles the producer, and
        # every map frame the writers push ahead of the video is a full 59 MB
        # frame ffmpeg buffers inside its own graph (rawvideo demux queue +
        # framesync input queues, four streams). Raising this 3 -> 8 put an
        # 8K render at ~19 GB of ffmpeg RSS two minutes in (vs ~15 GB) on the
        # 23 GB box -- straight toward the OOM killer. Two renders running at
        # once on one box once stalled at the second-to-last frame (producer
        # blocked on a full queue whose writer was blocked in a fifo write
        # ffmpeg was not draining); the answer to that is not to run two
        # renders at once, not a deeper queue.
        self.queue = queue.Queue(maxsize=3)

        deg_px = out_w / 360.0
        self.deg_px = deg_px
        win = int(round(SEAM_FLOW_WIN_DEG * deg_px))
        # the seam window spans the whole (asymmetric) flow band plus a margin:
        # from above the zenith taper down to the nadir limit
        y0 = int(round(out_h * (90 - (SEAM_FLOW_TAPER[1] + 3)) / 180.0))
        y1 = int(round(out_h * (90 + SEAM_FLOW_LAT_MAX + 1) / 180.0))
        self.rows = slice(y0, min(y1, out_h))
        R = self.rows.stop - self.rows.start
        self.row_lats = 90.0 - (np.arange(self.rows.start, self.rows.stop)
                                + 0.5) * 180.0 / out_h
        if self.flow is not None:
            f_lats = self.flow['lats']          # descending
            # output rows spanned by the flow band, and the flow's own row
            # coordinate for each output row (for the vertical resample)
            self.r_flow_start = int(np.searchsorted(-self.row_lats, -f_lats[0]))
            self.r_flow_end = int(np.searchsorted(-self.row_lats, -f_lats[-1]))
            lons = self.flow['lons_rel']
            self.f_c0 = int(round(win + lons[0] * deg_px))
            self.f_cn = int(round(win + lons[-1] * deg_px)) + 1
            edge = 1.5
            lon_px = np.arange(self.f_cn - self.f_c0) / deg_px + lons[0]
            self.f_taper = np.clip((lons[-1] - np.abs(lon_px)) / edge,
                                   0.0, 1.0)[None, :, None]

        self.seams = []
        for s, lon_c in enumerate(centers):
            xc = int(round((lon_c + 180.0) / 360.0 * out_w))
            xpos = xc + np.arange(-win, win + 1)
            cols = np.mod(xpos, out_w)
            contig = bool(np.all(np.diff(cols) == 1))
            sub_m = (mask[self.rows][:, cols].astype(np.float32) / 255.0)
            ramp_px = SEAM_FLOW_RAMP_HALF * deg_px
            rel = (xpos - xc).astype(np.float32)
            rises = sub_m[:, -1].mean() > sub_m[:, 0].mean()
            s_ramp = np.clip((rel if rises else -rel) / (2 * ramp_px) + 0.5,
                             0.0, 1.0)[None, :]
            self.seams.append({
                'idx': s, 'xpos': xpos.astype(np.float32), 'cols': cols,
                'contig': contig, 'c0': int(cols[0]), 'c1': int(cols[-1]) + 1,
                's': s_ramp,
                'static': [(xm[i][self.rows][:, cols].copy(),
                            ym[i][self.rows][:, cols].copy())
                           for i in range(2)],
            })
        self.threads = []

    def _deep_field(self, j, seam_idx):
        """Upsampled (dx, dy) of the dense flow at output frame j, in output
        pixels, over the full window (zero where the flow has no data)."""
        import cv2
        fm = self.flow
        pos = (self.j0 + j) / fm['step']
        s0 = min(int(pos), fm['data'].shape[0] - 1)
        s1 = min(s0 + 1, fm['data'].shape[0] - 1)
        t = min(max(pos - s0, 0.0), 1.0)
        F = ((1 - t) * fm['data'][s0, seam_idx].astype(np.float32)
             + t * fm['data'][s1, seam_idx].astype(np.float32))
        R = self.rows.stop - self.rows.start
        C = len(self.seams[seam_idx]['cols'])
        Rf = self.r_flow_end - self.r_flow_start
        up = cv2.resize(F, (self.f_cn - self.f_c0, Rf),
                        interpolation=cv2.INTER_LINEAR)
        up *= self.f_taper
        out = np.zeros((R, C, 2), np.float32)
        out[self.r_flow_start:self.r_flow_end, self.f_c0:self.f_cn] = up
        # measurement samples -> degrees -> output pixels (the vertical
        # pixel scale out_h/180 equals the horizontal out_w/360)
        out *= SEAM_MEASURE_RES * self.deg_px
        return out

    def _frame_windows(self, j):
        """Per-seam warped map windows for output frame j.

        Returns [stream][seam] uint16 arrays, streams ordered x0,y0,x1,y1,
        or None for a frame that needs no warp (static windows are sent)."""
        wins = [[], [], [], []]
        for seam in self.seams:
            dxy = self._deep_field(j, seam['idx'])
            if np.abs(dxy).max() < 0.05:
                for i in range(2):
                    st_x, st_y = seam['static'][i]
                    wins[2 * i].append(st_x)
                    wins[2 * i + 1].append(st_y)
                continue
            for i in range(2):
                ramp = seam['s'] if i == 0 else -(1.0 - seam['s'])
                dx = dxy[..., 0] * ramp
                dy = dxy[..., 1] * ramp
                xs = seam['xpos'][None, :] + dx
                ys = (np.arange(self.rows.start, self.rows.stop,
                                dtype=np.float32)[:, None] + dy)
                np.clip(ys, 0, self.out_h - 2, out=ys)
                ix0 = np.floor(xs).astype(np.int64)
                iy0 = np.floor(ys).astype(np.int64)
                fx = (xs - ix0).astype(np.float32)
                fy = (ys - iy0).astype(np.float32)
                c0 = np.mod(ix0, self.out_w)
                c1 = np.mod(ix0 + 1, self.out_w)
                i00 = iy0 * self.out_w + c0
                i01 = iy0 * self.out_w + c1
                i10 = i00 + self.out_w
                i11 = i01 + self.out_w
                st_x, st_y = seam['static'][i]
                res = []
                for full, st in ((self.xm[i], st_x), (self.ym[i], st_y)):
                    fl = full.ravel()
                    a00 = fl[i00].astype(np.float32)
                    a01 = fl[i01].astype(np.float32)
                    a10 = fl[i10].astype(np.float32)
                    a11 = fl[i11].astype(np.float32)
                    bad = ((a00 == 65535) | (a01 == 65535)
                           | (a10 == 65535) | (a11 == 65535))
                    v = (a00 * (1 - fx) * (1 - fy) + a01 * fx * (1 - fy)
                         + a10 * (1 - fx) * fy + a11 * fx * fy)
                    res.append(np.where(bad, st,
                                        np.rint(v)).astype(np.uint16))
                wins[2 * i].append(res[0])
                wins[2 * i + 1].append(res[1])
        return wins

    def _timed_windows(self, j):
        t0 = time.perf_counter()
        w = self._frame_windows(j)
        self.t_produce += time.perf_counter() - t0
        return w

    def _produce(self):
        import collections
        from concurrent.futures import ThreadPoolExecutor
        pending = collections.deque()
        j = 0
        try:
            with ThreadPoolExecutor(self.workers) as ex:
                while (pending or j < self.n_frames) and not self.stop.is_set():
                    while j < self.n_frames and len(pending) < self.workers + 1:
                        pending.append(ex.submit(self._timed_windows, j))
                        j += 1
                    self.queue.put(pending.popleft().result())
                for f in pending:
                    f.cancel()
        finally:
            self.queue.put(None)

    def _write(self):
        # one stacked frame: x0, y0, x1, y1 blocks of out_h rows each
        buf = np.concatenate([self.xm[0], self.ym[0], self.xm[1], self.ym[1]],
                             axis=0)
        blocks = [slice(k * self.out_h + self.rows.start,
                        k * self.out_h + self.rows.stop) for k in range(4)]
        try:
            with open(self.fifo, 'wb') as fh:
                while True:
                    t0 = time.perf_counter()
                    wins = self.queue.get()
                    t1 = time.perf_counter()
                    self.t_idle += t1 - t0
                    if wins is None:
                        return
                    for rows, w in zip(blocks, wins):
                        for seam, win in zip(self.seams, w):
                            buf[rows][:, seam['cols']] = win
                    t2 = time.perf_counter()
                    fh.write(memoryview(buf))
                    self.t_paste += t2 - t1
                    self.t_write += time.perf_counter() - t2
                    self.n_written += 1
        except (BrokenPipeError, OSError):
            self.stop.set()
            while self.queue.get() is not None:
                pass

    def start(self):
        self.threads = [threading.Thread(target=self._produce, daemon=True)]
        if self.fifo:
            self.threads.append(threading.Thread(target=self._write, daemon=True))
        for t in self.threads:
            t.start()

    def next_windows(self):
        """fifo=False mode: the next frame's windows, in order (None at the
        end). The producer pool runs ahead by its queue depth."""
        wins = self.queue.get()
        if wins is not None:
            self.n_written += 1
        return wins

    def timing(self):
        """Where the map stream spent its time, per frame written."""
        n = self.n_written
        if not n:
            return ''
        k = 1000.0 / n
        if not self.fifo:
            return (f"seam maps: produce {self.t_produce * k / self.workers:.0f} "
                    f"ms/frame ({self.workers} workers)")
        return (f"seam maps: produce {self.t_produce * k / self.workers:.0f} ms/frame "
                f"({self.workers} workers), paste {self.t_paste * k:.0f}, "
                f"fifo write {self.t_write * k:.0f} (blocked = ffmpeg busy), "
                f"writer idle {self.t_idle * k:.0f} (waiting on producer)")

    def finish(self):
        self.stop.set()
        # unblock a producer waiting on a full queue, then let the sentinel
        # wind the writer down
        deadline = time.time() + 15
        while any(t.is_alive() for t in self.threads) \
                and time.time() < deadline:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            time.sleep(0.05)
        for t in self.threads:
            t.join(timeout=1)
