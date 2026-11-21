"""ffmpeg/ffprobe helpers and input/output path handling."""

import json
import os
import subprocess
import sys

from .constants import VIDEO_EXTS


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ffprobe_streams(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams',
         '-show_format', path],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def find_streams(info):
    """Split an .OSV's streams into (fisheye video idx, audio idx, data idx)."""
    vids, auds, data = [], [], []
    for s in info['streams']:
        i, ct = s['index'], s.get('codec_type')
        if ct == 'video' and s.get('codec_name') == 'hevc':
            vids.append(i)
        elif ct == 'audio':
            auds.append(i)
        elif ct == 'data':
            data.append(i)
    return vids, auds, data



def iter_inputs(path, exts=None):
    """A file, or every matching file under a directory tree (recursive,
    sorted, hidden files such as macOS `._*` resource forks skipped)."""
    exts = exts or VIDEO_EXTS
    if os.path.isdir(path):
        found = []
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            for f in sorted(files):
                if f.startswith('.'):
                    continue
                if os.path.splitext(f)[1].lower() in exts:
                    found.append(os.path.join(root, f))
        if not found:
            sys.exit(f"no {'/'.join(sorted(exts))} files under {path}")
        return found
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    return [path]


def out_path(src, outdir, suffix):
    """Output path for src: same stem + suffix, next to the source unless an
    output directory is given. `stitch` uses suffix '.mp4', so a
    tree of DJI folders gets `CAM_0001.mp4` beside each `CAM_0001.OSV`."""
    stem = os.path.splitext(os.path.basename(src))[0]
    d = outdir or os.path.dirname(os.path.abspath(src))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, stem + suffix)
