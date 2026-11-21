"""IMU stabilisation: horizon levelling and heading smoothing, as per-frame rotations."""

import numpy as np

from .geometry import quat_matrix


def _minimal_rot(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def horizon_rot(q):
    """Rotation (equirect frame) that levels the horizon for orientation q.

    Gravity is the body -Z axis (world -Z through the quaternion, then
    IMU_TO_EQUIRECT into the equirect frame -- see _to_equirect for how
    that transport was measured).
    """
    g = _to_equirect(q, [0.0, 0.0, -1.0])
    g = g / np.linalg.norm(g)
    # Build the correction directly as pitch-then-roll, with no yaw. The
    # shortest-arc rotation would also spin about the vertical, and stripping
    # yaw from it afterwards leaves a large residual tilt (the three Euler
    # angles are coupled). Instead: the roll that brings g into the y-z
    # plane, then the pitch that brings it onto -y. Both are zero when the
    # camera is level, and neither touches heading.
    #   g = (gx, gy, gz); "down" target is (0, -1, 0)
    roll = np.degrees(np.arctan2(-g[0], -g[1]))      # about z: kills gx
    gy_after = -np.hypot(g[0], g[1])                  # y after that roll
    pitch = np.degrees(np.arctan2(g[2], -gy_after))   # about x: kills gz
    return _euler_to_rot(0.0, pitch, roll)


# IMU body frame -> rendered equirect frame (front lens centred). MEASURED,
# not assumed: a visual gyro (ORB features on an unstabilised render,
# unprojected to the sphere, Kabsch per frame pair) gave the camera's true
# per-frame rotation; the fixed rotation that maps the IMU quaternion's
# rotations onto it was fitted (AX = XB) and then refined by minimising the
# tilt left over after levelling. It converged to this exact axis
# permutation (every entry within 0.2 deg): body X -> -Z, body Y -> -X,
# body Z -> +Y (up). The earlier hand-derived transport (WORLD_ROT + x flip)
# had body X and Y swapped: it corrected a camera roll as if it were a
# pitch and vice versa, which is why the levelled output shook MORE than the
# raw one on a standing, swaying wearer (visual tilt jitter 0.454 vs 0.313
# deg/frame). Its horizon-tilt *magnitude* matched DJI, which is all the
# original validation looked at; the tilt's azimuth was never checked.
IMU_TO_EQUIRECT = (np.array([[0.0, -1.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [-1.0, 0.0, 0.0]]) if np is not None else None)


def _to_equirect(q, v_world):
    """Transport a world-frame vector into the (unlevelled) equirect frame.

    The quaternion maps body -> world (its matrix transposed takes a world
    vector into the body frame); IMU_TO_EQUIRECT then takes the body frame
    into the rendered equirect frame. Both facts were established by
    measurement against the visual gyro (see IMU_TO_EQUIRECT).
    """
    v = quat_matrix(q).T @ np.asarray(v_world, dtype=np.float64)
    return IMU_TO_EQUIRECT @ v


def heading_deg(q, L):
    """Heading (deg, wrapped) of the camera for orientation q: the
    longitude at which a fixed WORLD horizontal axis appears in the LEVELLED
    output. Computed with the very levelling matrix L that is applied to the
    picture, so it is the true output longitude of far scenery up to a
    constant -- whatever imperfections L has are included, which is what a
    smoother must see."""
    f = L @ _to_equirect(q, [1.0, 0.0, 0.0])
    return float(np.degrees(np.arctan2(f[0], f[2])))


def heading_track(quats):
    """Unwrapped heading (deg) at every IMU sample: (ts, psi)."""
    ts = np.array([t for t, _ in quats], dtype=np.float64)
    psi = np.array([heading_deg(q, horizon_rot(q)) for _, q in quats])
    return ts, np.unwrap(psi, period=360.0)


def smooth_heading(psi_deg, rate_hz, sigma_s):
    """Zero-phase gaussian low-pass of an UNWRAPPED heading track.

    Symmetric kernel (no lag -- the whole track is known), reflect-padded so
    the ends are not pulled toward the clip mean. sigma_s <= 0 returns the
    input unchanged."""
    psi = np.asarray(psi_deg, dtype=np.float64)
    if sigma_s <= 0 or len(psi) < 3:
        return psi
    s = sigma_s * rate_hz
    half = int(np.ceil(4 * s))
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / s) ** 2)
    k /= k.sum()
    padded = np.pad(psi, half, mode='reflect')
    return np.convolve(padded, k, mode='valid')


def _wrap180(a):
    return ((np.asarray(a, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _v360_angles(M):
    """Decompose a rotation into the yaw/pitch/roll v360 expects (degrees).

    v360 applies yaw about Y, then pitch about X, then roll about Z, on the
    equirect sphere. Solve for the equivalent Euler triple.
    """
    # M = Rz(roll) @ Rx(pitch) @ Ry(yaw). Rz leaves row 2 alone, so
    # row 2 = [-cos p sin y, sin p, cos p cos y]; and column 0 of Rz@Rx@Ry
    # gives the roll once yaw is known. Verified by round-trip against
    # _euler_to_rot.
    pitch = np.degrees(np.arcsin(np.clip(M[2, 1], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(-M[2, 0], M[2, 2]))
    roll = np.degrees(np.arctan2(-M[0, 1], M[1, 1]))
    return yaw, pitch, roll


def _euler_to_rot(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def stab_targets(quats, fps, start_s=0.0, mode='smooth', smooth=1.0,
                 yaw_sign=1.0, imu_offset=0.0):
    """Per-frame stabilising rotations (one 3x3 target matrix per frame).

    Modes:
      horizon -- level the horizon (pitch/roll) only; heading is the
                 camera's own, jitter and all. DJI "Direction Lock" without
                 its smoothing.
      smooth  -- level the horizon AND low-pass the heading: the forward
                 direction still follows the wearer through turns, but the
                 step-to-step yaw sway is gone. `smooth` is the gaussian
                 sigma in seconds (0 = identical to horizon). This is what
                 makes a walking clip watchable without an NLE stabiliser --
                 FCP's 2-D stabiliser shifts the equirect frame flat and
                 cannot wrap the sphere, which leaves a black line at the
                 +-180 edge. A 360 rotation has no crop, so the smoothing is
                 free.
      lock    -- world-locked heading (DJI "Free View"): the view does not
                 turn with the camera at all. For tripod-like shots.

    The heading is measured with heading_deg() from the very levelling
    matrix applied to the picture, over the WHOLE IMU track (so a
    --start/--duration snippet is identical to the same seconds of a full
    render), and the per-frame correction is Ry(delta) @ L: the yaw is
    applied in the LEVELLED frame, i.e. about the true vertical. Building
    it as _euler_to_rot(delta, pitch, roll) would instead be L @ Ry(delta),
    a spin about the camera's tilted axis -- second order for small deltas,
    but wrong by twice the tilt for lock mode's large ones. The sign of
    delta was established by measurement (yaw_sign; see docs/HOW_IT_WORKS.md), not by
    reasoning from v360's conventions.

    v360 driven by sendcmd COMPOSES each command with its current state
    rather than replacing it (verified: sending "roll 30" every frame
    accumulates 30 deg/frame). The first version of this sent per-frame
    DELTAS to compensate. That was wrong in a subtle way: the Euler
    decomposition of a small delta is not exactly the delta rotation, so the
    chain drifts, and the drift lands exactly where the two lenses are
    cross-faded -- it showed up as straight fence posts bending into arcs at
    both seams. The fix is v360's `reset_rot`, which is runtime-settable:
    every frame resets the rotation to identity and then applies the
    ABSOLUTE correction. No chain, nothing to accumulate. Verified with
    synthetic markers: reset+pitch 20 then reset+pitch 5 lands at -20 then
    -5, not -20 then -25.

    Returns (targets, stats): a list of 3x3 matrices, one per output frame,
    and the rms/max heading correction in degrees (both 0 for horizon mode).
    write_stab_sendcmd() turns the targets into v360 commands; stab_matrix()
    turns them into the rotation EquirectRotator applies directly.
    """
    if not quats:
        return [], {}
    if mode == 'smooth' and smooth <= 0:
        mode = 'horizon'          # exactly the raw-heading path, not just ~0.002 deg from it
    ts = np.array([t for t, _ in quats])
    fps_f = eval(fps) if isinstance(fps, str) and '/' in fps else float(fps)
    n_frames = int(round((ts[-1] - start_s) * fps_f)) + 1
    if n_frames <= 0:
        return [], {}

    # target correction per output frame: interpolate the IMU orientation to
    # the frame's own timestamp (slerp between the two neighbouring samples).
    # Snapping to a neighbouring sample instead adds up to half a frame of
    # timing error, which on fast head motion shows up as the horizon
    # leading or lagging the picture by a few degrees for a frame.
    def slerp(qa, qb, f):
        qa = np.asarray(qa, float)
        qb = np.asarray(qb, float)
        d = float(qa @ qb)
        if d < 0:
            qb, d = -qb, -d
        if d > 0.9995:
            q = qa + f * (qb - qa)
            return tuple(q / np.linalg.norm(q))
        th = np.arccos(d)
        return tuple((np.sin((1 - f) * th) * qa + np.sin(f * th) * qb) / np.sin(th))

    # reference heading track over the whole clip (smooth / lock modes)
    psi_ref = None
    if mode in ('smooth', 'lock'):
        ts_imu, psi = heading_track(quats)
        if mode == 'smooth':
            rate = 1.0 / max(float(np.median(np.diff(ts_imu))), 1e-6) \
                if len(ts_imu) > 1 else fps_f
            psi_ref = smooth_heading(psi, rate, smooth)
        else:
            psi_ref = np.full_like(psi, np.interp(start_s, ts_imu, psi))

    targets = []
    deltas = []
    for i in range(n_frames):
        # imu_offset: seconds added to the IMU sampling time of every frame
        # (calibration knob for the IMU-to-frame timing)
        t = start_s + i / fps_f + imu_offset
        j = int(np.searchsorted(ts, t))
        if j <= 0:
            q = quats[0][1]
        elif j >= len(quats):
            q = quats[-1][1]
        else:
            t0, q0 = quats[j - 1]
            t1, q1 = quats[j]
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            q = slerp(q0, q1, min(max(f, 0.0), 1.0))
        L = horizon_rot(q)
        if psi_ref is not None:
            ref = float(np.interp(t, ts_imu, psi_ref))
            delta = yaw_sign * float(_wrap180(ref - heading_deg(q, L)))
            deltas.append(delta)
            L = _euler_to_rot(delta, 0.0, 0.0) @ L      # yaw about the TRUE vertical
        targets.append(L)

    d = np.abs(deltas) if deltas else np.zeros(1)
    return targets, {'rms': float(np.sqrt(np.mean(d ** 2))),
                     'max': float(d.max())}


def write_stab_sendcmd(path, targets, fps_f):
    """Write the stab_targets() matrices as a v360 sendcmd script.

    Every command makes v360 rebuild its whole lookup table (0.22 s per
    command at 8K, four commands per frame -- see EquirectRotator for why
    this path is now the fallback, not the default, wherever torch is
    available).
    """
    lines = []
    for i, T in enumerate(targets):
        y, p, r = _v360_angles(T)
        # fire half a frame early so the command lands before the frame
        t = max(0.0, (i - 0.5) / fps_f)
        # v360's own convention, measured with synthetic markers rather than
        # assumed: R = Rz(roll) @ Rx(pitch) @ Ry(-yaw). Same order as ours,
        # yaw sign flipped -- so negate yaw and pass pitch/roll straight.
        # reset_rot first, so these are absolute, not composed.
        lines.append(f"{t:.6f} v360 reset_rot 1, v360 yaw {-y:.5f}, "
                     f"v360 pitch {p:.5f}, v360 roll {r:.5f};")
    with open(path, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
