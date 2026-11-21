"""Command-line interface."""

import argparse
import os
import sys

from . import __version__
from .constants import (
    DEFAULT_EDGE_SLOPE,
    DEFAULT_RADIUS_MULT,
    STAB_IMU_OFFSET,
    STAB_SMOOTH_DEFAULT,
    STAB_YAW_SIGN,
)
from .djmd import load_calibration
from .geometry import lut_radius
from .spherical import tag_spherical
from .stitch import cmd_stitch
from .util import ffprobe_streams, find_streams, iter_inputs, out_path, run
from .verify import verify_output


def cmd_inspect(args):
    info = ffprobe_streams(args.input)
    vids, auds, data = find_streams(info)
    dur = float(info['format'].get('duration', 0))
    size = int(info['format'].get('size', 0))

    print(f"{os.path.basename(args.input)}")
    print(f"  {size/1e9:.2f} GB   {int(dur//60)}m{dur%60:04.1f}s")
    print(f"  fisheye video streams: {vids}")
    print(f"  audio streams:         {auds}")
    print(f"  data streams:          {data}")
    for s in info['streams']:
        tag = s.get('codec_tag_string', '')
        extra = ''
        if s.get('codec_type') == 'video':
            extra = (f"{s.get('width')}x{s.get('height')} "
                     f"{s.get('pix_fmt','')} {s.get('r_frame_rate','')}")
        br = s.get('bit_rate')
        if br:
            extra += f"  {int(br)/1e6:.1f} Mbps"
        print(f"    [{s['index']}] {s.get('codec_type'):5s} "
              f"{s.get('codec_name','?'):8s} {tag:6s} {extra}")

    lenses, dev = load_calibration(args.input, data)
    if dev:
        print("\n  camera:")
        for k, v in dev.items():
            if v:
                print(f"    {k:14s} {v}")
    if not lenses:
        print("\n  ! no lens calibration decoded")
        return
    print("\n  lens calibration:")
    for i, l in enumerate(lenses):
        print(f"    lens {i}: fx={l['fx']:.2f} fy={l['fy']:.2f} "
              f"c=({l['cx']:.1f}, {l['cy']:.1f})")
        print(f"            dist={[round(d, 6) for d in l['dist']]}")
        print(f"            yaw={l['yaw']:.2f} pitch={l['pitch']:.2f} "
              f"calib circle r={lut_radius(l):.1f}px")


def cmd_cover(args):
    for src in iter_inputs(args.input):
        dst = out_path(src, args.output, '_cover.jpg')
        info = ffprobe_streams(src)
        mj = [s['index'] for s in info['streams']
              if s.get('codec_name') == 'mjpeg']
        if not mj:
            print(f"  ! no cover image in {os.path.basename(src)}")
            continue
        run(['ffmpeg', '-v', 'error', '-i', src, '-map', f'0:{mj[0]}',
             '-frames:v', '1', '-y', dst], capture_output=True)
        print(f"  {os.path.basename(dst)}")


def cmd_extract(args):
    """Lossless dual-fisheye: stream copy both lenses + audio, drop dbgi."""
    for src in iter_inputs(args.input):
        dst = out_path(src, args.output, '_fisheye.mov')
        info = ffprobe_streams(src)
        vids, auds, _ = find_streams(info)
        cmd = ['ffmpeg', '-v', 'error', '-stats', '-i', src]
        for v in vids[:2]:
            cmd += ['-map', f'0:{v}']
        for a in auds[:1]:
            cmd += ['-map', f'0:{a}']
        cmd += ['-c', 'copy', '-tag:v', 'hvc1', '-y', dst]
        print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}")
        run(cmd)


def cmd_tag(args):
    for src in iter_inputs(args.input, exts={'.mp4', '.mov', '.m4v'}):
        tag_spherical(src, args.output)


def cmd_finalize(args):
    """Make DJI Studio exports FCP-ready: faststart remux + 360 tag + verify.

    DJI Studio's panoramic export carries sv3d but no st3d (stereo mode) box
    and writes moov at the end of the file; both make FCP misbehave. The
    remux is a lossless stream copy.
    """
    all_ok = True
    for src in iter_inputs(args.input, exts={'.mp4', '.mov', '.m4v'}):
        print(f"  {os.path.basename(src)}")
        ok, problems = verify_output(src, quiet=True)
        if ok:
            print("    already FCP-ready; nothing to do")
            continue
        if any('moov after mdat' in p for p in problems):
            tmp = src + '.faststart.tmp.mp4'
            print("    remuxing with faststart (lossless)...")
            run(['ffmpeg', '-v', 'error', '-i', src, '-map', '0',
                 '-c', 'copy', '-movflags', '+faststart', '-y', tmp])
            os.replace(tmp, src)
        if any('sv3d' in p or 'st3d' in p for p in problems):
            tag_spherical(src)
        ok, _ = verify_output(src)
        all_ok = all_ok and ok
    if not all_ok:
        sys.exit(1)


def cmd_verify(args):
    bad = 0
    for src in iter_inputs(args.input, exts={'.mp4', '.mov', '.m4v'}):
        ok, _ = verify_output(src)
        bad += 0 if ok else 1
    if bad:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        prog='osv2mp4',
        description='Convert DJI Osmo 360 .OSV files into editable MP4/MOV.')
    ap.add_argument('--version', action='version',
                    version=f'%(prog)s {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('inspect', help='show streams, camera and calibration')
    p.add_argument('input')
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser('cover', help="extract DJI's stitched cover JPEG")
    p.add_argument('input')
    p.add_argument('-o', '--output', help='output directory')
    p.set_defaults(func=cmd_cover)

    p = sub.add_parser('extract', help='lossless dual-fisheye MOV (stream copy)')
    p.add_argument('input')
    p.add_argument('-o', '--output', help='output directory')
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser('convert', aliases=['stitch'],
                       help='convert .OSV to equirectangular 360 MP4 '
                            '(the main command)')
    p.add_argument('input')
    p.add_argument('-o', '--output', help='output directory')
    p.add_argument('--width', type=int, default=7680,
                   help='equirect width; height is width/2 (default 7680)')
    p.add_argument('--crf', type=int, default=16,
                   help='x265 quality, lower is better (default 16)')
    p.add_argument('--preset', default='medium', help='x265 preset')
    p.add_argument('--encoder', default='libx265',
                   help='video encoder; use hevc_videotoolbox on a Mac for '
                        'hardware encoding (default libx265)')
    p.add_argument('--vt-quality', type=int, default=70,
                   help='hevc_videotoolbox quality 0-100 (default 70)')
    p.add_argument('--radius-mult', type=float, default=DEFAULT_RADIUS_MULT,
                   help='global radial scale (default %(default)s)')
    p.add_argument('--edge-slope', type=float, default=DEFAULT_EDGE_SLOPE,
                   help='legacy no-op (the Mei radial model needs no linear '
                        'edge extrapolation); kept for compatibility')
    p.add_argument('--calibrate', action='store_true',
                   help='measure the per-body rim scale (radius mult) from '
                        'far scenery along the seams; use for a new camera body')
    p.add_argument('--yaw', type=float,
                   help='rotate the panorama about the vertical axis, degrees '
                        '(default: centre the front lens, matching DJI Studio)')
    p.add_argument('--seam', choices=['flow', 'static'], default='flow',
                   help='seam handling: flow measures the per-frame parallax '
                        'along both seams and warps the lenses into agreement '
                        'before blending (default); static is a plain '
                        'cross-fade, which double-exposes anything closer '
                        'than a few metres')
    p.add_argument('--seam-step', type=int, default=3,
                   help='measure seam parallax every Nth frame (default 3)')
    p.add_argument('--seam-smooth', type=float, default=0.4,
                   help='temporal smoothing of the seam offsets, seconds '
                        '(default 0.4)')
    p.add_argument('--stabilize', choices=['smooth', 'horizon', 'none', 'lock'],
                   default='smooth',
                   help='IMU stabilisation: smooth (default) levels the horizon '
                        'and low-passes the heading so the view follows the '
                        'wearer without the step-to-step sway (no crop -- a '
                        '360 rotation is free; use this instead of an NLE '
                        'stabiliser, which cannot wrap the sphere); horizon = '
                        "level only, raw heading; lock = world-locked heading "
                        "(DJI Free View, for tripod-like shots); none = the "
                        "camera's own tilt")
    p.add_argument('--stab-smooth', type=float, default=STAB_SMOOTH_DEFAULT,
                   help='heading low-pass sigma in seconds for --stabilize '
                        'smooth (0 = raw heading, same as horizon). Default '
                        '%(default)s')
    p.add_argument('--stab-yaw-sign', type=float, default=STAB_YAW_SIGN,
                   help=argparse.SUPPRESS)
    p.add_argument('--stab-offset', type=float, default=STAB_IMU_OFFSET,
                   help=argparse.SUPPRESS)
    p.add_argument('--rotator', default='auto',
                   choices=['auto', 'torch', 'v360', 'numpy'],
                   help='who applies the per-frame stabilising rotation: '
                        'torch runs it on the GPU (Apple mps / cuda; CPU if '
                        'neither), between a decode+stitch ffmpeg and an '
                        'encode ffmpeg; v360 is the ffmpeg filter driven by '
                        'sendcmd, which rebuilds its whole 8K table four '
                        'times per frame (~1 s/frame -- the reason a host '
                        'render sat at 0.9 fps); numpy is the CPU reference. '
                        'auto (default) = torch when a GPU is available, '
                        'else v360. Both render the same picture to 1 LSB.')
    p.add_argument('--stitch', default='auto', choices=['auto', 'gpu', 'ffmpeg'],
                   help='where the remap + blend run: gpu = Metal kernels on '
                        'an Apple GPU with the maps resident there (ffmpeg '
                        'only decodes and encodes); ffmpeg = the remap/'
                        'maskedmerge graph with fifo-streamed maps. auto '
                        '(default) = gpu whenever the Metal kernels compile. '
                        'Bit-exact with each other.')
    p.add_argument('--jobs', type=int, default=0,
                   help='parallel segment pipelines: the clip is split into '
                        'N contiguous frame ranges, each rendered by its own '
                        'decoder + GPU stage + encoder, then joined losslessly '
                        '(frames identical to a single run). Default 1. '
                        'WARNING: a second 8K pipeline pushed a 48 GB Mac '
                        'into swap -- only try 2 with plenty of free RAM.')
    p.add_argument('--transport', default='auto',
                   choices=['auto', 'mixed', 'socket', 'pipe'],
                   help='how frames travel between the decode ffmpeg, this '
                        'process and the encode ffmpeg: mixed (auto/default) '
                        '= pipe from the decoder, unix socket with a 4 MB '
                        'send buffer to the encoder; socket / pipe = both '
                        'directions. Measured on macOS: a pipe carries an 8K '
                        'frame in 23 ms, a socket in 8 ms when the sender sets '
                        'its buffer -- which only this process can do, not '
                        'ffmpeg. Output is identical.')
    p.add_argument('--hwaccel', default='auto',
                   help='ffmpeg decode acceleration for the two fisheye '
                        'streams: auto (default) = videotoolbox on macOS, '
                        'none elsewhere; or an explicit ffmpeg -hwaccel name')
    p.add_argument('--stab-interp', default=None,
                   choices=['near', 'linear', 'cubic', 'lanczos', 'spline16'],
                   help='resampling filter for the v360 horizon-lock pass '
                        '(the torch rotator is always v360-cubic). '
                        'Default: cubic, except linear at 8K on a box with '
                        "< 30 GB RAM -- v360's cubic tables alone are "
                        '~8.5 GB at 8K and are what pushed full runs into '
                        'the OOM killer. Pass a value to override.')
    p.add_argument('--frame-threads', type=int, default=None,
                   help='x265 frame threads (default: 2, or 1 at 8K on a '
                        'box with < 30 GB RAM)')
    p.add_argument('--lookahead', type=int, default=None,
                   help='x265 rc-lookahead frames (default: 10, or 8 at 8K '
                        'on a box with < 30 GB RAM; each is a full-res '
                        'frame in RAM)')
    p.add_argument('--start', type=float, help='start offset in seconds')
    p.add_argument('--duration', type=float, help='duration in seconds to encode')
    p.add_argument('--any-model', action='store_true',
                   help='stitch files from a camera other than the '
                        'first-generation Osmo 360 (untested: expect seams)')
    p.add_argument('--no-tag', action='store_true',
                   help='skip 360 spherical metadata injection')
    p.add_argument('--delta', action='store_true',
                   help='skip inputs whose output already exists')
    p.set_defaults(func=cmd_stitch)

    p = sub.add_parser('finalize',
                       help='make DJI Studio exports FCP-ready '
                            '(faststart + 360 tag + verify)')
    p.add_argument('input', help='exported .mp4 file or directory of them')
    p.set_defaults(func=cmd_finalize)

    p = sub.add_parser('tag', help='inject 360 spherical metadata')
    p.add_argument('input')
    p.add_argument('-o', '--output', help='output file (default: in place)')
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser('verify', help='check an MP4/MOV is FCP-ready 360')
    p.add_argument('input')
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
