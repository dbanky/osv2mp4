"""Decode DJI's `djmd` protobuf metadata: camera identity, per-lens factory calibration, IMU orientation."""

import os
import shutil
import struct
import subprocess
import tempfile

from .util import run


def read_varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if not (c & 0x80):
            return r, i
        s += 7


def parse_fields(b):
    """Generic protobuf wire-format decode: {field_number: [(wiretype, value)]}."""
    i, out, n = 0, {}, len(b)
    while i < n:
        try:
            tag, i = read_varint(b, i)
            f, w = tag >> 3, tag & 7
            if f == 0:
                break
            if w == 0:
                v, i = read_varint(b, i)
            elif w == 5:
                v = b[i:i + 4]
                i += 4
            elif w == 1:
                v = b[i:i + 8]
                i += 8
            elif w == 2:
                ln, i = read_varint(b, i)
                v = b[i:i + ln]
                i += ln
            else:
                break
        except IndexError:
            break
        out.setdefault(f, []).append((w, v))
    return out


def _f32(b):
    return struct.unpack('<f', b)[0]


def _floats(b):
    return list(struct.unpack('<%df' % (len(b) // 4), b)) if len(b) % 4 == 0 else None


def decode_device(sample):
    """Camera identity from the djmd header (top.1.1)."""
    try:
        hdr = parse_fields(parse_fields(sample)[1][0][1])[1][0][1]
        h = parse_fields(hdr)
        txt = lambda k: h[k][0][1].decode(errors='replace') if k in h else None
        return {'proto': txt(1), 'firmware': txt(2), 'proto_version': txt(3),
                'serial': txt(5), 'build': txt(6), 'model': txt(10)}
    except Exception:
        return {}


def decode_lenses(sample):
    """Per-lens factory calibration from djmd (top.2.6.*).

    The block repeats once per capture mode, so keep the first pair whose
    optical axes point in opposite directions.
    """
    top = parse_fields(sample)
    if 2 not in top:
        return []
    blk = parse_fields(top[2][0][1])
    if 6 not in blk:
        return []
    cal = parse_fields(blk[6][0][1])

    found = []
    for fnum in sorted(cal):
        for _w, v in cal[fnum]:
            lp = parse_fields(v)
            if 1 not in lp:
                continue
            g = lambda k: _f32(lp[k][0][1]) if k in lp else None
            found.append({
                'fx': g(1), 'fy': g(2), 'cx': g(3), 'cy': g(4),
                'dist': [g(k) for k in (5, 6, 7, 8)],
                'width': g(10), 'height': g(11),
                'yaw': g(12), 'pitch': g(13),
                'quat': _floats(lp[21][0][1]) if 21 in lp else None,
                'lut_x': _floats(lp[22][0][1]) if 22 in lp else None,
                'lut_y': _floats(lp[23][0][1]) if 23 in lp else None,
            })
    if not found:
        return []
    out = [found[0]]
    for l in found[1:]:
        if abs(((l['yaw'] - out[0]['yaw'] + 180) % 360) - 180) > 90:
            out.append(l)
            break
    return out if len(out) == 2 else []


def load_calibration(osv, data_indices):
    """Pull the djmd track that carries calibration and decode it."""
    tmp = tempfile.mkdtemp(prefix='osvcal_')
    try:
        for idx in data_indices:
            blob = os.path.join(tmp, f'd{idx}.bin')
            try:
                run(['ffmpeg', '-v', 'error', '-i', osv, '-map', f'0:{idx}',
                     '-c', 'copy', '-f', 'data', '-t', '1', '-y', blob],
                    capture_output=True)
            except subprocess.CalledProcessError:
                continue
            sample = open(blob, 'rb').read()
            lenses = decode_lenses(sample)
            if lenses:
                return lenses, decode_device(sample)
        return [], {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)




def read_orientation(osv, data_indices, max_seconds=None):
    """Per-frame orientation quaternions from the djmd track.

    Layout (reverse-engineered):
      top.3            one record per video frame
      top.3.3.2.{1,2}  orientation samples
          .1           timestamp, microseconds since boot
          .3.{1..4}    four tagged float32 -- a unit quaternion, [w,x,y,z]

    Returns [(seconds_from_first_sample, (w,x,y,z)), ...] or [].
    """
    tmp = tempfile.mkdtemp(prefix='osvimu_')
    try:
        for idx in data_indices:
            blob = os.path.join(tmp, f'd{idx}.bin')
            cmd = ['ffmpeg', '-v', 'error', '-i', osv, '-map', f'0:{idx}',
                   '-c', 'copy', '-f', 'data']
            if max_seconds:
                cmd += ['-t', str(max_seconds)]
            cmd += ['-y', blob]
            try:
                run(cmd, capture_output=True)
            except subprocess.CalledProcessError:
                continue
            top = parse_fields(open(blob, 'rb').read())
            got = []
            for _w, v in top.get(3, []):
                f = parse_fields(v)
                if 3 not in f:
                    continue
                inner = parse_fields(f[3][0][1])
                if 2 not in inner:
                    continue
                items = parse_fields(inner[2][0][1])
                for k in sorted(items):
                    for _w2, vv in items[k]:
                        sub = parse_fields(vv)
                        if 1 not in sub or 3 not in sub:
                            continue
                        rec = []
                        for _w3, item in sub[3]:
                            qs = parse_fields(item)
                            q = [_f32(qs[i][0][1]) for i in sorted(qs)
                                 if qs[i][0][0] == 5]
                            if len(q) == 4:
                                n = sum(x * x for x in q) ** 0.5
                                if abs(n - 1.0) < 0.01:
                                    rec.append(tuple(q))
                        if rec:
                            got.append((sub[1][0][1], rec))
                    break
            if got:
                got.sort(key=lambda r: r[0])
                t0 = got[0][0]
                out = []
                for r, (ts, rec) in enumerate(got):
                    # spread this record's samples evenly up to the next
                    # record's timestamp (last record: assume the same span)
                    if r + 1 < len(got):
                        span = got[r + 1][0] - ts
                    elif r > 0:
                        span = ts - got[r - 1][0]
                    else:
                        span = 1000.0 * len(rec)
                    step = span / len(rec)
                    for k, q in enumerate(rec):
                        out.append(((ts - t0 + k * step) / 1e6, q))
                return out
        return []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
