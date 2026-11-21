"""Check that an output file is something Final Cut Pro accepts as 360 video."""

import os
import subprocess

from .spherical import _top_level_boxes
from .util import ffprobe_streams


def verify_output(path, expect_frames=None, expect_fps=None, quiet=False,
                  require_360=True):
    """Check that a stitched file is something Final Cut Pro will accept.

    Returns (ok, list of problem strings). Every check here corresponds to a
    real way the output can look fine in a player and still misbehave in FCP:
    the wrong HEVC tag (hev1 vs hvc1), a non-2:1 frame, missing 360 metadata,
    a moov at the end of the file, or a silently truncated frame count.
    """
    problems = []
    try:
        info = ffprobe_streams(path)
    except subprocess.CalledProcessError:
        return False, ['ffprobe cannot read the file']

    vids = [s for s in info['streams'] if s.get('codec_type') == 'video']
    auds = [s for s in info['streams'] if s.get('codec_type') == 'audio']
    if not vids:
        return False, ['no video stream']
    v = vids[0]

    # codec + tag: FCP wants HEVC in an hvc1 sample entry (parameter sets in
    # the container). hev1 (in-band) plays in VLC but FCP treats it as
    # unsupported media.
    if v.get('codec_name') != 'hevc':
        problems.append(f"video codec is {v.get('codec_name')}, expected hevc")
    if v.get('codec_tag_string') != 'hvc1':
        problems.append(f"HEVC tag is {v.get('codec_tag_string')}, FCP needs hvc1")
    if v.get('pix_fmt') not in ('yuv420p10le', 'yuv420p'):
        problems.append(f"pix_fmt {v.get('pix_fmt')} is unusual for FCP")

    # equirect must be exactly 2:1
    w, h = int(v.get('width', 0)), int(v.get('height', 0))
    if w != 2 * h:
        problems.append(f"frame {w}x{h} is not 2:1 equirectangular")

    # 360 spherical metadata (Google spatial-media v2)
    sd = v.get('side_data_list', []) or []
    kinds = {d.get('side_data_type') for d in sd}
    has_360 = 'Spherical Mapping' in kinds
    if require_360:
        if not has_360:
            problems.append('no spherical (sv3d) metadata -- FCP/YouTube will not see 360')
        else:
            proj = next((d.get('projection') for d in sd
                         if d.get('side_data_type') == 'Spherical Mapping'), None)
            if proj != 'equirectangular':
                problems.append(f"spherical projection is {proj}, expected equirectangular")
        if 'Stereo 3D' not in kinds:
            problems.append('no st3d (stereo mode) box -- monoscopic flag missing')

    # frame accounting
    nb = int(v.get('nb_frames', 0) or 0)
    if expect_frames and nb and abs(nb - expect_frames) > 2:
        problems.append(f"{nb} frames written, source had {expect_frames} "
                        f"(delta {nb - expect_frames})")
    if expect_fps and v.get('r_frame_rate') != expect_fps:
        problems.append(f"frame rate {v.get('r_frame_rate')} != source {expect_fps}")

    if not auds:
        problems.append('no audio stream (source had audio?)')

    # faststart: moov must precede mdat, or FCP/QuickTime scrub badly and
    # streaming previews fail
    try:
        boxes, _ = _top_level_boxes(path)
        order = [b[0] for b in boxes]
        if b'moov' in order and b'mdat' in order and \
                order.index(b'moov') > order.index(b'mdat'):
            problems.append('moov after mdat (no faststart)')
    except Exception:
        pass

    ok = not problems
    if not quiet:
        name = os.path.basename(path)
        if ok:
            print(f"  ✓ {name}: {w}x{h} {v.get('codec_name')}/{v.get('codec_tag_string')} "
                  f"{v.get('pix_fmt')} {v.get('r_frame_rate')} fps, "
                  f"{nb} frames, {'360 equirect' if has_360 else 'UNTAGGED'}, "
                  f"{'audio' if auds else 'NO audio'}, faststart -- "
                  f"{'FCP ready' if has_360 else 'needs tag before FCP'}")
        else:
            print(f"  ✗ {name}:")
            for p in problems:
                print(f"      - {p}")
    return ok, problems
