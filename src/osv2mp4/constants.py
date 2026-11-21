"""Tuning constants of the stitcher. Every value here was measured; the comments say how."""

VIDEO_EXTS = {'.osv'}

# The camera every constant below was measured on: the first-generation
# Osmo 360, as it names itself in the djmd header. Other models (the
# Osmo 360 II included) have different optics and are refused by default.
SUPPORTED_MODEL = 'Osmo 360'

# theta beyond which a lens contributes nothing (deg from its optical axis)
FOV_HALF_DEG = 96.0
# half-width of the linear cross-fade centred on the 90 deg seam
FADE_HALF_DEG = 5.0
# Toward the nadir the wide cross-fade is wrong: the camera operator's own
# body/hand sits 10-30 cm away there, its parallax is tens of degrees, and a
# 10-deg blend double-exposes it into large curved smears along the seams.
# No warp can reconcile two genuinely different views of a body that close --
# the industry fix (and what DJI Studio's own export shows) is a near-hard cut
# so the close object is single-imaged. The fade half-width therefore shrinks
# from FADE_HALF_DEG above NADIR_NARROW_START to NADIR_FADE_HALF_DEG below
# NADIR_NARROW_END (latitudes, deg; nadir is -90). The zenith keeps the wide
# fade: sky is far, and a wide blend hides small radiometric steps there.
NADIR_NARROW_START = -42.0
NADIR_NARROW_END = -62.0
NADIR_FADE_HALF_DEG = 0.4
# --- radial lens model (Mei unified omnidirectional) -------------------------
# The lens maps theta (angle from the optical axis) to image radius via DJI's
# actual model: the Mei unified omnidirectional (catadioptric) projection,
# recovered from DJI Studio's own calibration bundle
# (DualStitcher .../case_00/intrinsic_extrinsic_list/*.json). A world ray at
# angle theta projects to a normalised-plane radius rho = sin(theta) /
# (cos(theta) + XI) (XI is the mirror parameter), then radial distortion
# (1 + K1*rho^2 + K2*rho^4) is applied, then a per-body pixel scale.
#
# This replaces the earlier reading of DJI's four factory coefficients as an
# OpenCV-fisheye theta polynomial. That reading was only trustworthy to ~80 deg:
# past it the polynomial's slope collapsed and it TURNED OVER near 88-92 deg
# (radius decreasing with angle -- impossible for a fisheye), so the whole
# 88-96 deg overlap band -- exactly where the two lenses blend -- had to be
# faked with a hand-fitted straight-line extrapolation (slope ~1.28*fx). The
# Mei model is well-defined and monotonic all the way to the rim, so no
# extrapolation hack is needed. Validated three independent ways on our body:
# it matches the OSV's own per-body polynomial to ~2.5 px RMS over 2-70 deg
# (the region the polynomial IS trustworthy in), stays monotonic to 96 deg, and
# lands within 0.4% of the OSV's measured image-circle radius at theta=93 deg
# (a ~186 deg fisheye). See docs/HOW_IT_WORKS.md.
#
# XI/K1/K2 are lens-DESIGN constants (both Osmo 360 lenses agree to <1%:
# xi 0.7129/0.7173, k1 -0.18304/-0.18293, k2 both 0.0184), so they are baked in;
# only the pixel SCALE is fitted per body, to that body's own factory
# polynomial (mei_scale()).
MEI_XI = 0.713
MEI_K1 = -0.1830
MEI_K2 = 0.0184
# Global scale on the radial curve, i.e. the per-body RIM correction. The Mei
# scale is fitted to the factory polynomial over 2-70 deg, but what matters
# for stitching is where the rim (theta ~ 90 deg) lands, and on our body it
# lands ~0.8% too far out: far scenery (hills at 60/120/200 s) sat ~1.7 deg
# off along BOTH seams with OPPOSITE signs (+1.9 on seam 0, -1.6 on seam 1).
# Opposite signs across the two seams is the signature of a radial-scale
# error (a relative lens yaw would give the same sign on both). The seam
# optical flow had been silently absorbing that offset -- and wherever it
# failed to (dark ground rows) the seam ghosted. A brute-force far-scenery
# shift search (calibrate_radius_mult) crosses zero at 0.9914 (seam 0) and
# 0.9927 (seam 1); at 0.992 the residual is <= 0.5 deg in every latitude band
# from +55 to -50 on both seams, so the flow now carries only real parallax.
# Per-body: `--calibrate` remeasures it for another camera.
DEFAULT_RADIUS_MULT = 0.992
# Bounds for --calibrate's radius-scale search; hitting a bound is treated as
# a failed measurement, not a result.
RADIUS_MULT_RANGE = (0.975, 1.015)
# Retained for CLI/signature compatibility only -- the Mei model needs no linear
# extrapolation, so radial_fn ignores the edge-slope entirely.
DEFAULT_EDGE_SLOPE = 1.28

# --- seam flow (parallax compensation) ---------------------------------------
# The two lenses do not share an entrance pupil, so anything closer than a few
# metres lands at different longitudes in the two hemispheres and a static
# blend double-exposes it. Measured on real footage the offset is almost purely
# horizontal, up to ~1.5 deg for ground at ~1 m. Seam flow measures that offset
# per latitude per frame and warps both lenses toward each other inside the
# blend band before merging -- the same idea as DJI Studio's seam optical flow,
# reduced to the 1-D (horizontal, per-latitude) field that actually dominates.
# Clamp on the measured offset, in PHYSICAL degrees. In equirect longitude the
# same physical parallax grows as 1/cos(lat), so the per-latitude clamp is
# SEAM_FLOW_MAX_DEG / cos(lat), capped at SEAM_FLOW_MAX_LON_DEG. A fixed
# longitude clamp would silently reject the (correct, large) offsets measured
# for feet/ground at -60..-70 lat and leave them duplicated at the seam cut.
SEAM_FLOW_MAX_DEG = 2.5
SEAM_FLOW_MAX_LON_DEG = 6.0
# The flow band is ASYMMETRIC. Toward the zenith it fades out across
# SEAM_FLOW_TAPER (sky needs no seam warp and the overlap thins out); toward
# the nadir it runs all the way down to -SEAM_FLOW_LAT_MAX, because that is
# where close parallax actually lives -- the wearer's body, the ground, and
# anyone standing next to the camera sit at -40..-75 lat. An earlier revision
# applied the taper to |lat| on both sides, which silently stopped correcting
# a person's torso and legs below -45 and brought the seam ghosting back.
SEAM_FLOW_LAT_MAX = 82.0       # nadir side: measure/warp down to this lat
SEAM_FLOW_TAPER = (45.0, 55.0)  # zenith side: fade the warp to zero across this lat
SEAM_FLOW_NADIR_FADE = 4.0     # fade to zero over the last degrees before -LAT_MAX
# The whole seam uses DENSE 2-D optical flow between the two lens views
# (Farneback on the LRF-scale band), Kalman-filtered over time and folded
# into the streamed remap maps -- the mechanism behind DJI Studio's seam
# optical flow. A 1-D per-latitude offset was tried first but shears the
# image (each latitude shifts independently), and the shear ripples over
# time into the visible seam "waving"; a coherent 2-D field does not.
SEAM_FLOW2D_DY_MAX = 2.5       # vertical flow clamp, physical degrees
SEAM_FLOW2D_STORE_STEP = 3     # store the flow field at this grid decimation
# Erode the both-lenses-valid mask by this many measurement samples before
# trusting the flow: at a lens's validity edge one view goes black and
# Farneback fills its window with a large, clamp-hitting bogus flow.
SEAM_FLOW_EDGE_ERODE = 13

# Temporal filtering of the seam flow. A straight edge near the seam is shifted
# by the measured per-frame offset, so sub-degree frame-to-frame jitter in that
# offset makes the edge wave side to side over time -- the defect that survived
# fixed-window (gaussian/EMA) smoothing. DJI's stitcher runs a KalmanFlowFilter
# on the seam flow; this reproduces it: a random-walk Kalman filter with RTS
# smoothing (an adaptive, zero-lag low-pass). Its measurement noise scales
# inversely with per-sample confidence, so far/textureless scenery (near-zero
# true parallax, noisy measurement) is averaged down to a steady ~0 -- no
# waving -- while genuine, confident close parallax is followed. A random-walk
# (position-only) state is used rather than constant-velocity: it has no
# velocity term to carry low-frequency noise, so it settles instead of
# wandering, and the RTS backward pass removes the lag on real ramps.
# PROC_SIGMA is how fast the true offset may drift, deg/s (higher = more
# responsive, less steady); R0 the base measurement variance; CONF_FLOOR caps
# how much a zero-confidence sample is trusted.
KALMAN_PROC_SIGMA = 0.9
KALMAN_R0 = 0.06
KALMAN_CONF_FLOOR = 0.02
# Half-width (longitude deg) of the fixed warp ramp around each seam. The warp
# must ramp over a support WIDER than the blend band: with the near-hard nadir
# cut, tying the ramp to the blend weight would shift each lens only where it
# is already invisible, doing nothing -- each lens must instead carry half the
# offset at the cut line so the two sides meet. Must stay below
# SEAM_FLOW_WIN_DEG minus the largest possible shift.
SEAM_FLOW_RAMP_HALF = 8.0
# Heading low-pass sigma (s) for --stabilize smooth. See docs/HOW_IT_WORKS.md for how
# it was set against DJI Studio's own Direction Lock export.
STAB_SMOOTH_DEFAULT = 1.0
# Sign of the heading correction and the IMU-to-frame timing offset (s),
# both established by measurement on the walking clip (see docs/HOW_IT_WORKS.md).
STAB_YAW_SIGN = 1.0
# +9 ms: the 1 kHz orientation samples are timestamped ~9 ms before the frame
# they belong to (exposure mid-point / IMU latency). Measured by scanning the
# offset against the visual gyro on a `--start 198` snippet: sharp, symmetric
# minimum of the leftover tilt (0.106 -> 0.022 deg/frame). That scan read
# "+40 ms" in the snippet's own frame times -- but a seeked ffmpeg decode
# starts at the first frame AT OR AFTER --start (198.0312 s here, not 198.0),
# so the snippet's frames were 31 ms later than assumed and the true offset
# is 40 - 31 = 9 ms. Applying 40 ms to an un-seeked full render put the IMU
# 31 ms late and brought back a 0.09 deg/frame residual (predicted by the
# scan curve at +71 ms snippet-time). The seek is now accounted for in
# stitch_one (start_s = first frame's real time), so this is the true value.
STAB_IMU_OFFSET = 0.009
SEAM_MEASURE_SIZE = 1024       # fisheye size the measurement runs at
SEAM_MEASURE_RES = 0.1         # measurement grid resolution, deg/sample
SEAM_MEASURE_HALF_LON = 8.0    # half-width of the measured band, deg
SEAM_FLOW_WIN_DEG = 16.0       # half-width of the map columns rewritten per frame
# setpts expression that puts a stream on the frame-index timeline (see the
# filter-graph comment in stitch_one)
FRAME_PTS = 'setpts=N/FRAME_RATE/TB'
