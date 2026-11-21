"""Spherical (360) MP4 metadata: sv3d/st3d box injection and the box-level MP4 helpers it needs."""

import os
import shutil
import struct


def _boxes(buf, start, end):
    """Yield (type, header_start, body_start, body_end) for boxes in a range."""
    i = start
    while i + 8 <= end:
        size = struct.unpack('>I', buf[i:i + 4])[0]
        btype = buf[i + 4:i + 8]
        body = i + 8
        if size == 1:
            size = struct.unpack('>Q', buf[i + 8:i + 16])[0]
            body = i + 16
        elif size == 0:
            size = end - i
        if size < 8 or i + size > end:
            return
        yield btype, i, body, i + size
        i += size


def _find(buf, start, end, path):
    """Walk a slash-free box path, returning (body_start, body_end)."""
    cur = (start, end)
    for want in path:
        nxt = None
        for btype, _h, body, stop in _boxes(buf, cur[0], cur[1]):
            if btype == want:
                nxt = (body, stop)
                break
        if nxt is None:
            return None
        cur = nxt
    return cur


def spherical_boxes():
    """Google spatial-media v2 st3d (monoscopic) + sv3d (equirectangular)."""
    st3d = struct.pack('>I4sIB', 13, b'st3d', 0, 0)

    svhd = b'Spherical Metadata Tool\x00'
    svhd = struct.pack('>I4sI', 12 + len(svhd), b'svhd', 0) + svhd
    # projection bounds: full sphere (top/bottom/left/right = 0)
    equi = struct.pack('>I4sIIIII', 28, b'equi', 0, 0, 0, 0, 0)
    prhd = struct.pack('>I4sIiii', 24, b'prhd', 0, 0, 0, 0)
    proj = struct.pack('>I4s', 8 + len(prhd) + len(equi), b'proj') + prhd + equi
    sv3d = struct.pack('>I4s', 8 + len(svhd) + len(proj), b'sv3d') + svhd + proj
    return st3d + sv3d


def _top_level_boxes(path):
    """Walk only the top-level box headers, without reading the payloads."""
    boxes = []
    with open(path, 'rb') as fh:
        fh.seek(0, os.SEEK_END)
        fsize = fh.tell()
        off = 0
        while off + 8 <= fsize:
            fh.seek(off)
            hdr = fh.read(16)
            if len(hdr) < 8:
                break
            size = struct.unpack('>I', hdr[0:4])[0]
            btype = hdr[4:8]
            if size == 1:
                if len(hdr) < 16:
                    break
                size = struct.unpack('>Q', hdr[8:16])[0]
            elif size == 0:
                size = fsize - off
            if size < 8 or off + size > fsize:
                break
            boxes.append((btype, off, size))
            off += size
    return boxes, fsize


def tag_spherical(path, out=None):
    """Insert st3d/sv3d into the video sample entry so players see 360.

    Only the moov box is held in memory. An 8K clip's mdat runs to many GB;
    slurping the whole file just to add ~90 bytes of metadata would need as
    much RAM as the file is large.
    """
    boxes, fsize = _top_level_boxes(path)
    moov_box = next((b for b in boxes if b[0] == b'moov'), None)
    if not moov_box:
        raise SystemExit(f'no moov box found in {path}')
    _mt, moov_off, moov_size = moov_box
    mdat_off = next((b[1] for b in boxes if b[0] == b'mdat'), None)

    with open(path, 'rb') as fh:
        fh.seek(moov_off)
        buf = bytearray(fh.read(moov_size))
    # inside `buf`, the moov box itself starts at 0
    moov = _find(buf, 0, len(buf), [b'moov'])
    if not moov:
        raise SystemExit('moov box unreadable')

    target = None
    for btype, hstart, body, stop in _boxes(buf, moov[0], moov[1]):
        if btype != b'trak':
            continue
        hdlr = _find(buf, body, stop, [b'mdia', b'hdlr'])
        if not hdlr or buf[hdlr[0] + 8:hdlr[0] + 12] != b'vide':
            continue
        stsd = _find(buf, body, stop, [b'mdia', b'minf', b'stbl', b'stsd'])
        if stsd:
            target = (hstart, stop, stsd)
            break
    if not target:
        raise SystemExit('no video sample description found')

    # first sample entry inside stsd (skip its 8-byte version/entry-count)
    entries = list(_boxes(buf, target[2][0] + 8, target[2][1]))
    if not entries:
        raise SystemExit('empty stsd')
    _bt, ehstart, _eb, eend = entries[0]

    if buf[ehstart:eend].find(b'sv3d') != -1:
        print(f"  already tagged: {os.path.basename(path)}")
        if out and out != path:
            shutil.copy2(path, out)
        return

    # Collect the enclosing boxes BEFORE mutating, so the offsets stay valid.
    chain = _path_to(buf, 0, len(buf), ehstart)

    payload = spherical_boxes()
    buf[eend:eend] = payload
    delta = len(payload)

    # every box on the path now contains `delta` more bytes
    for hstart in chain:
        size = struct.unpack('>I', buf[hstart:hstart + 4])[0]
        if size == 1:
            big = struct.unpack('>Q', buf[hstart + 8:hstart + 16])[0]
            struct.pack_into('>Q', buf, hstart + 8, big + delta)
        else:
            struct.pack_into('>I', buf, hstart, size + delta)

    # With faststart the moov sits before the mdat, so growing it pushes every
    # sample later in the file; the chunk offset tables have to follow.
    # The box walk is relative to `buf`, but stco/co64 hold absolute file
    # offsets, so the threshold stays absolute.
    if mdat_off is not None and moov_off < mdat_off:
        _shift_chunk_offsets(buf, moov[0], len(buf), delta, mdat_off)

    dest = out or path
    tmp = dest + '.tagging'
    with open(path, 'rb') as src, open(tmp, 'wb') as dst:
        _copy_range(src, dst, 0, moov_off)
        dst.write(buf)
        src.seek(moov_off + moov_size)
        _copy_range(src, dst, moov_off + moov_size, fsize)
    os.replace(tmp, dest)
    print(f"  tagged equirectangular: {os.path.basename(dest)}")


def _copy_range(src, dst, start, end, chunk=8 << 20):
    src.seek(start)
    left = end - start
    while left > 0:
        block = src.read(min(chunk, left))
        if not block:
            break
        dst.write(block)
        left -= len(block)


# boxes whose children must be descended into when locating a sample entry
_CONTAINERS = (b'moov', b'trak', b'mdia', b'minf', b'stbl', b'stsd')


def _path_to(buf, start, end, probe):
    """Header offsets of every box enclosing byte `probe`, outermost first."""
    chain = []

    def rec(s, e):
        for btype, hstart, body, stop in _boxes(buf, s, e):
            if not (hstart <= probe < stop):
                continue
            chain.append(hstart)
            if btype in _CONTAINERS:
                # stsd holds version/flags + entry_count before its children
                rec(body + 8 if btype == b'stsd' else body, stop)
            return

    rec(start, end)
    return chain


def _shift_chunk_offsets(buf, moov_start, moov_end, delta, mdat_start):
    """Add delta to stco/co64 entries that reference data after moov."""
    for btype, _h, body, stop in _iter_all(buf, moov_start, moov_end):
        if btype == b'stco':
            count = struct.unpack('>I', buf[body + 4:body + 8])[0]
            for k in range(count):
                off = body + 8 + 4 * k
                if off + 4 > stop:
                    break
                val = struct.unpack('>I', buf[off:off + 4])[0]
                if val >= mdat_start:
                    struct.pack_into('>I', buf, off, val + delta)
        elif btype == b'co64':
            count = struct.unpack('>I', buf[body + 4:body + 8])[0]
            for k in range(count):
                off = body + 8 + 8 * k
                if off + 8 > stop:
                    break
                val = struct.unpack('>Q', buf[off:off + 8])[0]
                if val >= mdat_start:
                    struct.pack_into('>Q', buf, off, val + delta)


def _iter_all(buf, start, end):
    for btype, hstart, body, stop in _boxes(buf, start, end):
        yield btype, hstart, body, stop
        if btype in (b'trak', b'mdia', b'minf', b'stbl', b'moov', b'edts'):
            yield from _iter_all(buf, body, stop)
