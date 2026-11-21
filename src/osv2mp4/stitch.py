"""The stitch command: one .OSV -> one stitched equirectangular HEVC MP4."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np

from .calibrate import calibrate_radius_mult
from .constants import FRAME_PTS, MEI_XI, SEAM_MEASURE_RES, SUPPORTED_MODEL
from .djmd import load_calibration, read_orientation
from .geometry import build_maps, default_yaw, write_pgm
from .gpu import (
    EquirectRotator,
    GpuStitcher,
    metal_lib,
    resolve_rotator,
    run_piped,
    run_rotated,
    stab_matrix,
)
from .seamflow import SeamFlowWriter, fps_float, measure_seam_flow
from .spherical import tag_spherical
from .stabilize import stab_targets, write_stab_sendcmd
from .util import ffprobe_streams, find_streams, iter_inputs, out_path, run
from .verify import verify_output


def cmd_stitch(args):
    failed, done, skipped = [], 0, 0
    srcs = iter_inputs(args.input)
    for k, src in enumerate(srcs, 1):
        dst = out_path(src, args.output, '.mp4')
        if args.delta and os.path.exists(dst):
            print(f"[{k}/{len(srcs)}] skip (exists): {dst}")
            skipped += 1
            continue
        print(f"[{k}/{len(srcs)}] {src}")
        try:
            ok = stitch_one(src, dst, args)     # True; False = failed verify;
        except (subprocess.CalledProcessError, OSError) as e:   # None = skipped (no calibration...)
            print(f"  ! {os.path.basename(src)}: {e}", file=sys.stderr)
            ok = False
        if ok is True:
            done += 1
        else:
            failed.append(dst)
    print(f"\n  {done} converted, {skipped} skipped, {len(failed)} failed"
          f" of {len(srcs)} .OSV file(s)")
    if failed:
        print("  failed:\n    " + "\n    ".join(failed), file=sys.stderr)
        sys.exit(1)


def stitch_one(src, dst, args):
    info = ffprobe_streams(src)
    vids, auds, data = find_streams(info)
    if len(vids) < 2:
        print(f"  ! {os.path.basename(src)}: expected 2 fisheye streams, "
              f"found {len(vids)}", file=sys.stderr)
        return
    dur = float(info['format'].get('duration', 0))
    fps = next((s.get('r_frame_rate') for s in info['streams']
                if s['index'] == vids[0]), '30000/1001')

    lenses, dev = load_calibration(src, data)
    if not lenses:
        print(f"  ! {os.path.basename(src)}: no calibration; skipping",
              file=sys.stderr)
        return

    print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}")
    model = (dev.get('model') or '').strip()
    if model:
        print(f"  camera {model} {dev.get('serial') or ''}".rstrip())
    if model and model != SUPPORTED_MODEL and not args.any_model:
        print(f"  ! {os.path.basename(src)}: recorded by '{model}'. The lens "
              f"model here was measured on the first-generation "
              f"'{SUPPORTED_MODEL}' only; a different camera will stitch "
              f"with visible seams. Pass --any-model to try anyway.",
              file=sys.stderr)
        return

    mult = args.radius_mult
    slope = args.edge_slope
    yaw = default_yaw(lenses) if args.yaw is None else args.yaw
    if args.calibrate:
        print("  auto-calibrating the rim scale from far scenery...")
        mult = calibrate_radius_mult(src, lenses, yaw, dur)
    out_w = args.width
    out_h = out_w // 2
    fps_f = fps_float(fps)

    # RAM-aware defaults. The full 8K pipeline was measured at ~12 GB for the
    # stitch graph alone, +3.3 GB for the streamed seam-flow maps, +1.5 GB
    # for v360 cubic tables, and x265-medium ramps a further 2-4 GB over the
    # first hour -- on a 23 GB box the ceiling was hit repeatedly. On small
    # RAM at high resolution, trade some sharpness (linear stab resample)
    # and encode speed (one x265 frame thread, shorter lookahead) for
    # staying alive. Explicit CLI values always win.
    try:
        ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        ram = 0
    tight = out_w >= 5760 and ram and ram < 30 * 2 ** 30
    stab_interp = args.stab_interp or ('linear' if tight else 'cubic')
    frame_threads = args.frame_threads or (1 if tight else 2)
    lookahead = args.lookahead or (6 if tight else 10)   # x265 needs > bframes (4 at medium)
    if tight and not (args.stab_interp and args.frame_threads
                      and args.lookahead):
        print(f"  {ram / 2**30:.0f} GB box at {out_w} wide: "
              f"stab-interp {stab_interp}, x265 frame-threads "
              f"{frame_threads}, lookahead {lookahead}")
    out_dur = args.duration if args.duration else max(dur - float(args.start or 0), 0)
    n_out = int(round(out_dur * fps_f))

    # seam flow: measure the per-frame parallax offset along both seams
    flow = None
    if args.seam == 'flow':
        print("  measuring seam parallax "
              f"(every {args.seam_step} frames)...")
        try:
            flow = measure_seam_flow(src, vids, lenses, yaw, mult, slope,
                                     fps_f, args.start, args.duration,
                                     n_out + 8, args.seam_step,
                                     args.seam_smooth)
        except Exception as e:
            print(f"  ! seam flow measurement failed ({e}); "
                  "falling back to static seams", file=sys.stderr)
            flow = None
        if flow is not None and flow[3] is not None:
            mag = np.abs(flow[3]['data'].astype(np.float32)) * SEAM_MEASURE_RES
            print(f"  seam flow: dense 2-D optical flow over the whole seam "
                  f"band, Kalman-filtered (median {np.median(mag[mag > 0.01]):.2f} "
                  f"deg, max {mag.max():.2f} deg)" if np.any(mag > 0.01)
                  else "  seam flow: no measurable parallax")
        else:
            print("  ! seam flow: measurement unavailable; static seams",
                  file=sys.stderr)
            flow = None

    print(f"  building {out_w}x{out_h} remap maps "
          f"(Mei model xi {MEI_XI:.3f}, radius x{mult:.3f}, yaw {yaw:.1f} deg)...")
    xm, ym, mask = build_maps(lenses, out_w, out_h, mult, yaw, slope)

    # GPU stitch: remap + blend on the Apple GPU too, so ffmpeg only decodes
    # (and a second ffmpeg only encodes). auto = whenever the Metal kernels
    # compile; the sandbox and any non-Apple box keep the ffmpeg graph.
    gpu_stitch = False
    if args.stitch != 'ffmpeg':
        dev = resolve_rotator('torch' if args.stitch == 'gpu' else 'auto')
        if dev == 'mps':
            try:
                metal_lib()
                gpu_stitch = True
            except Exception as e:      # noqa: BLE001
                if args.stitch == 'gpu':
                    sys.exit(f'--stitch gpu: Metal kernels unavailable ({e})')
        elif args.stitch == 'gpu':
            sys.exit('--stitch gpu needs torch with an Apple GPU (mps)')
    lens_size = next(int(s_['width']) for s_ in info['streams'] if s_['index'] == vids[0])
    transport = args.transport
    if transport == 'auto':
        transport = 'mixed'

    work = tempfile.mkdtemp(prefix='osvmap_')
    writer = None
    try:
        mpath = os.path.join(work, 'mask.pgm')
        write_pgm(mpath, mask, 255)
        map_paths = []
        if flow is not None:
            lat_bins, prof, centers, flow_meta = flow
            writer = SeamFlowWriter(work, xm, ym, mask, lat_bins, prof,
                                    centers, out_w, out_h, n_out + 8,
                                    flow_meta, fifo=not gpu_stitch)
        elif gpu_stitch:
            pass                        # static maps live on the GPU
        else:
            for i in range(2):
                px = os.path.join(work, f'x{i}.pgm')
                py = os.path.join(work, f'y{i}.pgm')
                write_pgm(px, xm[i], 65535)
                write_pgm(py, ym[i], 65535)
                map_paths += [px, py]

        # horizon lock: per-frame target rotations from the IMU, applied
        # either by v360 (sendcmd script) or by EquirectRotator (torch GPU /
        # CPU / numpy) between a decode+stitch ffmpeg and an encode ffmpeg
        stab_path = None
        stab_mats = None
        rotator = None
        if args.stabilize != 'none':
            quats = read_orientation(src, data)
            if not quats:
                print("  ! no IMU orientation stream found; stitching unstabilised",
                      file=sys.stderr)
            else:
                # ffmpeg's accurate seek emits the first frame whose pts is
                # AT OR AFTER --start, so a seeked render's frame 0 is up to
                # one frame later than --start itself (198.0 -> 198.0312 s).
                # Sample the IMU at the frames' real times, or every seeked
                # snippet is stabilised 0-33 ms early -- which is how the
                # IMU offset was first mis-measured (see STAB_IMU_OFFSET).
                start_s = 0.0
                if args.start:
                    start_s = math.ceil(float(args.start) * fps_f - 1e-6) / fps_f
                targets, st = stab_targets(quats, fps, start_s=start_s,
                                           mode=args.stabilize,
                                           smooth=args.stab_smooth,
                                           yaw_sign=args.stab_yaw_sign,
                                           imu_offset=args.stab_offset)
                engine = resolve_rotator(args.rotator)
                if engine == 'v360' and args.stitch == 'gpu':
                    engine = 'mps'          # gpu stitch implies the torch rotator
                if engine == 'v360':
                    stab_path = os.path.join(work, 'stab.cmd')
                    write_stab_sendcmd(stab_path, targets, fps_f)
                else:
                    rotator = EquirectRotator(out_w, out_h, engine)
                    stab_mats = [stab_matrix(T) for T in targets]
                desc = {'horizon': 'horizon lock, raw heading',
                        'smooth': f'horizon lock + heading smoothed '
                                  f'(sigma {args.stab_smooth:g}s)',
                        'lock': 'horizon lock + world-locked heading'}[args.stabilize]
                print(f"  stabilise: {desc}: {len(quats)} IMU samples -> "
                      f"{len(targets)} per-frame corrections"
                      + (f", heading correction rms {st['rms']:.2f} deg, "
                         f"max {st['max']:.1f} deg" if st.get('max') else "")
                      + f"; rotation by {'ffmpeg v360' if stab_path else rotator.describe()}")

        enc = ['-c:v', args.encoder, '-tag:v', 'hvc1',
               '-pix_fmt', 'yuv420p10le']
        if args.encoder == 'libx265':
            # An 8K 10-bit frame is ~88 MB and x265 holds several per frame
            # thread plus the lookahead queue; unbounded at 8K that OOMs.
            enc += ['-crf', str(args.crf), '-preset', args.preset,
                    '-x265-params',
                    f'frame-threads={frame_threads}:'
                    f'rc-lookahead={lookahead}']
            enc_desc = f'{args.encoder} crf {args.crf} {args.preset}'
        elif 'videotoolbox' in args.encoder:
            # Apple hardware HEVC: quality is -q:v (0-100), not CRF.
            enc += ['-q:v', str(args.vt_quality)]
            enc_desc = f'{args.encoder} q {args.vt_quality}'
        elif args.encoder == 'null':
            # benchmark only: run the whole graph, encode nothing
            enc = ['-f', 'null', '-']
            enc_desc = 'no encode (benchmark)'
        else:
            enc += ['-crf', str(args.crf)]
            enc_desc = f'{args.encoder} crf {args.crf}'
        if args.encoder != 'null':
            enc += ['-max_muxing_queue_size', '256',
                    '-movflags', '+faststart', '-y', dst]

        # ---- how many independent pipelines (segment jobs) render this clip
        F0 = int(math.ceil(float(args.start) * fps_f - 1e-6)) if args.start else 0
        src_frames = next((int(s_.get('nb_frames', 0) or 0)
                           for s_ in info['streams'] if s_['index'] == vids[0]), 0)
        whole_clip = not args.start and not args.duration
        n_total = src_frames if (whole_clip and src_frames) else n_out
        piped = gpu_stitch or rotator is not None
        jobs = args.jobs
        if jobs <= 0:
            # two whole pipelines on two halves of the clip would double the
            # encoder side up, but on a 48 GB M4 Max the second 8K pipeline
            # drove the machine into swap (each holds several 177 MB frames
            # in flight plus the GPU's caching allocator) -- one pipeline at
            # ~12 fps is the default; --jobs 2 is opt-in, watch memory.
            jobs = 1
        if not piped or args.encoder == 'null':
            jobs = 1
        jobs = max(1, min(jobs, max(1, n_total // 30)))
        bounds = [int(round(n_total * k / jobs)) for k in range(jobs + 1)]

        def decode_cmd(k, f0, count, writer_k):
            """The decode(+stitch) ffmpeg for job k: frames f0..f0+count-1 of
            the run, i.e. source frames F0+f0... A single job reproduces the
            exact command of an unsplit run (-ss start, -t duration); a
            segment seeks to half a frame before its first frame (ffmpeg's
            accurate seek emits the first frame AT OR AFTER -ss) and stops
            after exactly count frames."""
            c = ['ffmpeg', '-v', 'error'] + (['-stats'] if k == 0 else [])
            if jobs == 1:
                if args.start:
                    c += ['-ss', str(args.start)]
            elif F0 + f0 > 0:
                c += ['-ss', f'{(F0 + f0 - 0.5) / fps_f:.6f}']
            if tight:
                # HEVC frame-threading holds one 3840x3840 10-bit frame
                # (~44 MB) per thread per decoder; with two decoders the
                # default thread count is >1 GB of buffers. Decoding is
                # nowhere near the bottleneck at 8K (x265 is), so cap it.
                c += ['-threads', '4']
            hw = args.hwaccel
            if hw == 'auto':
                # VideoToolbox decodes both 3840x3840 HEVC 10-bit fisheyes on
                # the media engine (measured 103 fps vs 11 in software);
                # ffmpeg falls back to software by itself if the session
                # cannot be created. Frames come out as plain software
                # frames, so the 10-bit graph below is untouched.
                hw = 'videotoolbox' if sys.platform == 'darwin' else 'none'
            if hw != 'none':
                c += ['-hwaccel', hw]
            c += ['-i', src]
            if gpu_stitch:
                # decode only: both lens frames, gbrp10le (the very
                # conversion the graph did before remap), vstacked into one
                # rawvideo frame
                g = (f"[0:{vids[0]}]{FRAME_PTS},format=gbrp10le[a];"
                     f"[0:{vids[1]}]{FRAME_PTS},format=gbrp10le[b];"
                     "[a][b]vstack[st]")
            else:
                if writer_k:
                    # seam flow: the x/y maps arrive one frame at a time over
                    # one fifo, already carrying that frame's seam
                    # corrections. thread_queue_size counts PACKETS and a
                    # rawvideo packet here is a full 59 MB frame -- 8 per
                    # input across four inputs is ~1.9 GB of pure buffer,
                    # which at 8K is the difference between fitting in RAM
                    # and the OOM killer. 2 is enough: the producer is local
                    # and always ahead.
                    c += ['-f', 'rawvideo', '-pixel_format', 'gray16le',
                          '-video_size', f'{out_w}x{4 * out_h}', '-framerate', fps,
                          '-thread_queue_size', '1' if tight else '2',
                          '-i', writer_k.fifo]
                else:
                    for p_ in map_paths:
                        c += ['-framerate', fps, '-i', p_]
                c += ['-framerate', fps, '-i', mpath]
                # Static maps are one frame; framesync repeats a secondary
                # input's last frame after EOF, so one frame per map is all
                # remap needs (see the note on why they must NOT be looped).
                # Streamed (seam flow) maps come one per frame over the fifo.
                # 10-bit throughout: remap and maskedmerge both handle
                # gbrp10le, so the source's 10-bit depth survives all the
                # way to the encoder.
                if writer_k:
                    # one stacked map frame per video frame (see
                    # SeamFlowWriter): split into the four maps by cropping
                    # row blocks (no copies)
                    g = f"[1:v]{FRAME_PTS},split=4[s0][s1][s2][s3];" + "".join(
                        f"[s{q}]crop={out_w}:{out_h}:0:{q * out_h},setsar=1[{nm}];"
                        for q, nm in enumerate(('mx0', 'my0', 'mx1', 'my1')))
                    mk_in = 2
                else:
                    g = "".join(
                        f"[{q + 1}:v]{FRAME_PTS},setsar=1[{nm}];"
                        for q, nm in enumerate(('mx0', 'my0', 'mx1', 'my1')))
                    mk_in = 5
                # Static maps and the mask are SINGLE-FRAME inputs: framesync
                # repeats a secondary input's last frame after its EOF, so
                # nothing else is needed. The earlier `loop=-1:1:0` on them
                # was a bug with a long shadow: the looped frames were extra
                # framesync events (~1.8 output frames per source frame) and
                # the output vsync's dedup then kept the duplicate and
                # dropped the real frame about once in six -- a 5 Hz judder
                # (frame shown twice, next frame skipped) in every render up
                # to this point, read as "shake" and invisible to frame
                # counts (drop=149 on a 179-frame clip in the ffmpeg stats
                # was the tell). Every input is also put on the same
                # frame-index timeline (pts = N/fps) so that any remaining
                # duplicate event carries an identical pts and identical
                # content, which cfr vsync drops deterministically without
                # ever touching a real frame.
                g += (
                    f"[{mk_in}:v]{FRAME_PTS},format=gray,setsar=1[mk];"
                    f"[0:{vids[0]}]{FRAME_PTS},format=gbrp10le[b0];"
                    "[b0][mx0][my0]remap=fill=black[e0];"
                    f"[0:{vids[1]}]{FRAME_PTS},format=gbrp10le[b1];"
                    "[b1][mx1][my1]remap=fill=black[e1];"
                    "[e0][e1][mk]maskedmerge[st]"
                )
                if stab_path:
                    # Second resample: v360 equirect->equirect, orientation
                    # driven per frame by sendcmd. The remap maps are static
                    # so per-frame rotation cannot go into them; this is the
                    # cost of horizon lock. Kept 10-bit (gbrp10le in,
                    # format=yuv420p10le after).
                    g += (
                        f";[st]sendcmd=f='{stab_path}',"
                        f"v360=input=e:output=e:interp={stab_interp}:yaw=0:pitch=0:roll=0,"
                        "format=yuv420p10le[v]"
                    )
                elif rotator is None:
                    g += ";[st]format=yuv420p10le[v]"
                # (with a rotator, [st] stays free and is mapped raw to the pipe)
            c += ['-filter_complex', g]
            return c

        def bound(c, k, count):
            # The map inputs never end, so bound the output explicitly.
            # `-shortest` would instead truncate against the audio track,
            # which ends a few frames early and silently costs video frames.
            return c + (['-t', str(out_dur)] if jobs == 1 else ['-frames:v', str(count)])

        def encode_cmd(k, count, out):
            c = ['ffmpeg', '-v', 'error',
                 '-f', 'rawvideo', '-pixel_format', 'gbrp10le',
                 '-video_size', f'{out_w}x{out_h}', '-framerate', fps,
                 '-thread_queue_size', '2', '-i', 'pipe:0']
            if jobs == 1 and auds:
                # audio is muxed by the encoder side straight from the
                # source, stream-copied as before
                if args.start:
                    c += ['-ss', str(args.start)]
                c += ['-i', src, '-map', '0:v', '-map', f'1:{auds[0]}',
                      '-c:a', 'copy']
            else:
                c += ['-map', '0:v']
            return bound(c, k, count) + enc[:-1] + [out]

        if not piped:
            # one ffmpeg does everything (v360 rotation or no stabilisation)
            if writer:
                writer.start()
            cmd = decode_cmd(0, 0, n_total, writer) + ['-map', '[v]']
            if auds:
                cmd += ['-map', f'0:{auds[0]}', '-c:a', 'copy']
            cmd = bound(cmd, 0, n_total) + enc
            print(f"  encoding ({enc_desc}"
                  + (", seam flow" if writer else "") + ")...")
            run(cmd)
            if writer and writer.timing():
                print(f"  {writer.timing()}")
        else:
            rot_desc = rotator.describe() if rotator else 'none'
            print(f"  encoding ({enc_desc}"
                  + (", seam flow" if writer else "")
                  + (", stitch + rotation on Apple GPU (Metal)" if gpu_stitch
                     else f", rotation on {rot_desc}")
                  + (f", {jobs} parallel segments" if jobs > 1 else "") + ")...")
            if writer:
                writer.finish()         # the unsplit writer; each job gets its own
            results = [None] * jobs
            errors = [None] * jobs

            def job(k):
                f0, f1 = bounds[k], bounds[k + 1]
                count = f1 - f0
                wk = os.path.join(work, f'job{k}')
                os.makedirs(wk, exist_ok=True)
                writer_k = None
                if flow is not None:
                    writer_k = SeamFlowWriter(wk, xm, ym, mask, lat_bins, prof,
                                              centers, out_w, out_h, count + 8,
                                              flow_meta, fifo=not gpu_stitch,
                                              frame_offset=f0)
                mats_k = stab_mats[f0:f1] if stab_mats else None
                if mats_k is not None and len(mats_k) < count and stab_mats:
                    mats_k = mats_k + [stab_mats[-1]] * (count - len(mats_k))
                rot_k = rotator if k == 0 else EquirectRotator(out_w, out_h, rotator.device) \
                    if rotator is not None else None
                seg = dst if jobs == 1 else os.path.join(work, f'seg{k}.mp4')
                # passthrough: the remap/maskedmerge graph reports no frame
                # rate, and a rawvideo output then defaults to 25 fps cfr,
                # which silently drops one source frame in six.
                c1 = bound(decode_cmd(k, f0, count, writer_k)
                           + ['-map', '[st]'], k, count) \
                    + ['-fps_mode', 'passthrough',
                       '-f', 'rawvideo', '-pix_fmt', 'gbrp10le', 'pipe:1']
                c2 = encode_cmd(k, count, seg)
                if jobs > 1:
                    print(f"  job {k}: source frames {F0 + f0}..{F0 + f1 - 1}")
                try:
                    if writer_k:
                        writer_k.start()
                    if gpu_stitch:
                        stitcher = GpuStitcher(xm, ym, mask, lens_size, out_w, out_h)
                        dl = [0.0]

                        def process(frame, n):
                            if writer_k:
                                wins = writer_k.next_windows()
                                if wins is not None:
                                    stitcher.update_maps(wins, writer_k.seams, writer_k.rows)
                            out = stitcher.stitch(frame)
                            if rot_k is not None:
                                out = rot_k.rotate_device(out, mats_k[min(n, len(mats_k) - 1)])
                            t0 = time.perf_counter()
                            o = out.cpu().numpy().view(np.uint16)
                            dl[0] += time.perf_counter() - t0
                            return o

                        def timing(n):
                            kk = 1000.0 / n
                            return (stitcher.timing(n)
                                    + (f" + rotate {rot_k.t_rot * kk:.0f}" if rot_k else '')
                                    + f" + download {dl[0] * kk:.0f} ms/frame")
                        n = run_piped(c1, c2, (3, 2 * lens_size, lens_size),
                                      process, timing, transport, wk)
                    else:
                        n = run_rotated(c1, c2, rot_k, mats_k, out_w, out_h,
                                        transport, wk)
                    results[k] = (n, seg, writer_k.timing() if writer_k else '')
                except Exception as e:      # noqa: BLE001 -- reported by the caller
                    errors[k] = e
                finally:
                    if writer_k:
                        writer_k.finish()

            threads = [threading.Thread(target=job, args=(k,)) for k in range(jobs)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for e in errors:
                if e is not None:
                    raise e
            n_done = sum(r[0] for r in results)
            print(f"  {'stitched' if gpu_stitch else 'rotated'} {n_done} frames"
                  + (f" ({', '.join(str(r[0]) for r in results)} per job)" if jobs > 1 else ""))
            if results[0][2]:
                print(f"  {results[0][2]}" + (" (job 0)" if jobs > 1 else ""))
            if jobs > 1:
                # lossless join: concat demuxer, stream copy, audio muxed
                # once from the source exactly as the single-job encoder
                # side does
                lst = os.path.join(work, 'segments.txt')
                with open(lst, 'w') as fh:
                    for _n, seg, _t in results:
                        fh.write(f"file '{seg}'\n")
                cc = ['ffmpeg', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', lst]
                if auds:
                    if args.start:
                        cc += ['-ss', str(args.start)]
                    cc += ['-i', src, '-map', '0:v', '-map', f'1:{auds[0]}']
                else:
                    cc += ['-map', '0:v']
                cc += ['-c', 'copy', '-t', str(out_dur),
                       '-max_muxing_queue_size', '256',
                       '-movflags', '+faststart', '-y', dst]
                run(cc)
    finally:
        if writer:
            writer.finish()
        shutil.rmtree(work, ignore_errors=True)

    if args.encoder == 'null':
        return None                     # benchmark run: nothing to tag or verify
    if not args.no_tag:
        tag_spherical(dst)

    # Always verify. Expected frame count is source frames over the encoded
    # window; only checked when the whole clip was converted, since a
    # --start/--duration test range has no clean frame expectation.
    src_frames = next((int(s.get('nb_frames', 0) or 0) for s in info['streams']
                       if s['index'] == vids[0]), None)
    whole = not args.start and not args.duration
    ok, _ = verify_output(dst,
                          expect_frames=src_frames if whole else None,
                          expect_fps=fps,
                          require_360=not args.no_tag)
    return ok
