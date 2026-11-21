"""Lens model (Mei unified omnidirectional) and equirect remap-map construction."""

import numpy as np

from .constants import (
    DEFAULT_EDGE_SLOPE,
    DEFAULT_RADIUS_MULT,
    FADE_HALF_DEG,
    FOV_HALF_DEG,
    MEI_K1,
    MEI_K2,
    MEI_XI,
    NADIR_FADE_HALF_DEG,
    NADIR_NARROW_END,
    NADIR_NARROW_START,
)


def lut_radius(lens):
    """Mean radius of the calibration circle sampled in lut_x/lut_y.

    Entry 0 of each list is a header value (image centre / image-circle
    diameter), so only entries 1.. are points on the circle.
    """
    x = np.asarray(lens['lut_x'][1:], dtype=np.float64)
    y = np.asarray(lens['lut_y'][1:], dtype=np.float64)
    return float(np.hypot(x - lens['cx'], y - lens['cy']).mean())


def mei_shape(t, xi=None):
    """Dimensionless Mei radius vs theta(rad): sin/(cos+xi) then radial distort.

    No pixel scale -- that is per-body (see mei_scale). Well-defined and
    monotonic to the rim, unlike the factory theta polynomial.
    """
    if xi is None:
        xi = MEI_XI
    t = np.asarray(t, dtype=np.float64)
    rho = np.sin(t) / (np.cos(t) + xi)
    r2 = rho * rho
    return rho * (1.0 + MEI_K1 * r2 + MEI_K2 * r2 * r2)


def mei_scale(lens, xi=None):
    """Per-body pixel scale for the Mei model.

    Fit a single scale so the (dimensionless) Mei shape matches THIS lens's own
    factory theta polynomial over the region the polynomial is trustworthy in
    (2-70 deg) -- ~2.5 px RMS on our body. This keeps DJI's per-body factory
    calibration (the polynomial's absolute scale) while adopting the Mei shape
    for the rim the polynomial gets wrong. Memoised on the lens dict, keyed by
    xi so --calibrate can refit.
    """
    if xi is None:
        xi = MEI_XI
    cached = lens.get('_mei_scale')
    if cached is not None and abs(cached[0] - xi) < 1e-9:
        return cached[1]
    fx = lens['fx']
    k1, k2, k3, k4 = lens['dist']
    tt = np.radians(np.linspace(2.0, 70.0, 300))
    base = fx * (tt + k1 * tt**3 + k2 * tt**5 + k3 * tt**7 + k4 * tt**9)
    sh = mei_shape(tt, xi)
    s = float((sh @ base) / (sh @ sh))
    lens['_mei_scale'] = (xi, s)
    return s


def radial_fn(lens, radius_mult=DEFAULT_RADIUS_MULT,
              edge_slope=DEFAULT_EDGE_SLOPE):
    """Return theta(rad) -> image radius(px) for one lens, via the Mei model.

    DJI's calibration bundle shows the lens is a Mei unified omnidirectional
    projection: rho = sin(theta)/(cos(theta)+XI), radial-distorted by
    (1 + K1*rho^2 + K2*rho^4), then scaled to pixels. XI/K1/K2 are lens-design
    constants (module level); the pixel scale is fitted per body to that body's
    own factory polynomial. See the MEI_* constants block for why this replaces
    the old polynomial-plus-linear-extrapolation reading.

    edge_slope is accepted for call-site/CLI compatibility but unused: the Mei
    curve needs no extrapolation. XI is taken from the module global MEI_XI
    (which --calibrate may set).
    """
    s = mei_scale(lens)
    return lambda t: radius_mult * s * mei_shape(t)


def quat_matrix(q):
    """Rotation matrix from a [w, x, y, z] quaternion."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# Maps equirect world directions into the lens extrinsic frame. Two parts:
#   * +90 deg about x, because the equirect frame is y-up while the lens
#     extrinsic quaternions are z-up;
#   * a flip of x, because equirect longitude runs the opposite way round the
#     sphere from the extrinsics' handedness. Without it the whole panorama
#     comes out mirrored -- invisible on symmetric scenery, obvious the moment
#     any text is in shot (it reads backwards).
# The x flip makes this a reflection (det = -1), not a rotation. That is
# deliberate: it is a change of handedness, not an orientation tweak.
WORLD_ROT = (np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
             if np is not None else None)


def equirect_dirs(out_w, out_h, y0=0, y1=None, yaw_deg=0.0):
    """Unit world directions for equirect rows [y0, y1).

    yaw_deg rotates the panorama about the vertical axis, i.e. picks which way
    the centre of the frame looks.
    """
    y1 = out_h if y1 is None else y1
    lon = (np.arange(out_w) + 0.5) / out_w * 2 * np.pi - np.pi + np.radians(yaw_deg)
    lat = np.pi / 2 - (np.arange(y0, y1) + 0.5) / out_h * np.pi
    cl, sl = np.cos(lat), np.sin(lat)
    d = np.empty((y1 - y0, out_w, 3), dtype=np.float64)
    d[..., 0] = cl[:, None] * np.sin(lon)[None, :]
    d[..., 1] = sl[:, None]
    d[..., 2] = cl[:, None] * np.cos(lon)[None, :]
    return d @ WORLD_ROT.T


def default_yaw(lenses):
    """Yaw that puts the front lens (the one with yaw ~= 0) at frame centre.

    Without it the rear lens lands dead centre and the panorama comes out
    ~180 deg from how DJI Studio frames the same clip.
    """
    return float(lenses[0]['yaw'])


def project(lens, dirs, radius_mult, edge_slope=DEFAULT_EDGE_SLOPE):
    """Project world directions into one lens; returns px, py, theta."""
    dl = dirs @ quat_matrix(lens['quat']).T
    theta = np.arccos(np.clip(dl[..., 2], -1.0, 1.0))
    rho = np.maximum(np.hypot(dl[..., 0], dl[..., 1]), 1e-12)
    r = radial_fn(lens, radius_mult, edge_slope)(theta)
    px = lens['cx'] + r * dl[..., 0] / rho
    py = lens['cy'] + r * dl[..., 1] / rho
    return px, py, theta


def build_maps(lenses, out_w, out_h, radius_mult, yaw_deg=0.0,
               edge_slope=DEFAULT_EDGE_SLOPE, chunk_rows=256):
    """Build per-lens x/y remap arrays plus the blend mask.

    Done in row bands: the float64 direction array for a full 8K equirect
    frame is ~700 MB on its own, and several intermediates of that size at
    once are enough to get the process OOM-killed.
    """
    theta_max = np.radians(FOV_HALF_DEG)

    xmaps = [np.empty((out_h, out_w), np.uint16) for _ in lenses]
    ymaps = [np.empty((out_h, out_w), np.uint16) for _ in lenses]
    mask = np.empty((out_h, out_w), np.uint8)

    for y0 in range(0, out_h, chunk_rows):
        y1 = min(y0 + chunk_rows, out_h)
        dirs = equirect_dirs(out_w, out_h, y0, y1, yaw_deg)
        # per-row fade half-width: wide at mid latitudes, near-hard cut
        # toward the nadir (see NADIR_NARROW_* above)
        lat = 90.0 - (np.arange(y0, y1) + 0.5) * 180.0 / out_h
        t = np.clip((NADIR_NARROW_START - lat)
                    / (NADIR_NARROW_START - NADIR_NARROW_END), 0.0, 1.0)
        fh = FADE_HALF_DEG + (NADIR_FADE_HALF_DEG - FADE_HALF_DEG) * t
        a0 = np.radians(90 - fh)[:, None]
        a1 = np.radians(90 + fh)[:, None]
        weights, thetas, valids = [], [], []
        for i, lens in enumerate(lenses):
            px, py, theta = project(lens, dirs, radius_mult, edge_slope)
            valid = ((theta < theta_max) & (px >= 0) & (px <= lens['width'] - 1)
                     & (py >= 0) & (py <= lens['height'] - 1))
            xmaps[i][y0:y1] = np.where(valid, np.round(px), 65535).astype(np.uint16)
            ymaps[i][y0:y1] = np.where(valid, np.round(py), 65535).astype(np.uint16)
            weights.append(np.where(valid, np.clip((a1 - theta) / (a1 - a0), 0, 1), 0.0))
            thetas.append(theta)
            valids.append(valid)
        wsum = weights[0] + weights[1]
        # Where the (possibly very narrow) fade leaves both weights at zero
        # while both lenses still see the pixel, hard-pick the nearer valid
        # lens -- a 0.5 mix there would double-expose exactly the close
        # content the narrow fade exists to keep single-imaged.
        nearer1 = np.where(valids[0] & valids[1], thetas[1] < thetas[0],
                           valids[1])
        band = np.where(wsum > 0, weights[1] / np.maximum(wsum, 1e-12),
                        nearer1.astype(np.float64))
        mask[y0:y1] = (band * 255).round().astype(np.uint8)

    return xmaps, ymaps, mask


def write_pgm(path, arr, maxval):
    with open(path, 'wb') as fh:
        fh.write(b'P5\n%d %d\n%d\n' % (arr.shape[1], arr.shape[0], maxval))
        fh.write(arr.astype('>u2').tobytes() if maxval > 255 else arr.tobytes())
