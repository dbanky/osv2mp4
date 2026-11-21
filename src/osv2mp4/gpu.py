"""Per-frame rotation and GPU stitch stages (torch / Metal), and the ffmpeg -> process -> ffmpeg plumbing."""

import math
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np


_STAB_P = None


def stab_matrix(T):
    """The rotation EquirectRotator must apply to reproduce what v360 renders
    for stab_targets() matrix T.

    write_stab_sendcmd hands v360 (-yaw, pitch, roll) of T = Rz Rx Ry, and
    v360 rotates each OUTPUT direction by M = Ry(yaw) Rx(pitch) Rz(roll)
    (its matrices are the standard ones; the order and the yaw sign were
    found by brute force over all orders/signs against v360's own output).
    Algebraically that is M = P T^T P with P = diag(-1, 1, -1) -- exact, no
    Euler round trip, verified to 1e-17 against the angle path.
    """
    global _STAB_P
    if _STAB_P is None:
        _STAB_P = np.diag([-1.0, 1.0, -1.0])
    return _STAB_P @ np.asarray(T, dtype=np.float64).T @ _STAB_P


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_rotator(name):
    """--rotator -> 'v360' | 'numpy' | torch device ('mps' / 'cuda' / 'cpu').

    auto picks torch only when a GPU is there: on a plain CPU box v360 is
    faster than torch-CPU at 8K, and it is the path every earlier render
    was validated on.
    """
    if name in ('v360', 'numpy'):
        return name
    torch = _torch()
    if torch is None:
        if name == 'auto':
            return 'v360'
        sys.exit('--rotator torch needs torch (pip install torch in a venv); '
                 'or --rotator v360 / numpy')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'v360' if name == 'auto' else 'cpu'


# Metal compute kernels (torch >= 2.7 compiles them from source at runtime --
# no extra dependency). Two kernels:
#
# rotate_equirect -- the same resample as EquirectRotator.rotate(). The
#   generic torch path issues ~1000 small ops per frame with int64 index
#   maths and gathers, which on MPS cost 372 ms per 8K frame (measured);
#   this does the whole per-pixel job in one pass over int16 planes, in the
#   numpy version's float32 order, so the two agree to 1 LSB.
#
# stitch_equirect -- ffmpeg's `remap` x2 + `maskedmerge`, bit-exact
#   (stitch_frame_np is the numpy statement of the same arithmetic, verified
#   100% exact against ffmpeg's output): nearest-neighbour lens lookup, any
#   map value outside the lens (the 65535 sentinel reads as -1 in int16)
#   fills black, then dst = (a*(1023-m) + b*m + 511) / 1023 with the 8-bit
#   mask scaled x4 as ffmpeg's gray->gbrp10 conversion does (255 -> 1020, so
#   a "full" blend is 99.7% -- reproduced, not fixed, to stay identical).
#   With this, the fifo-streamed 236 MB map frame, both 8K remaps and the
#   blend leave ffmpeg; the static maps live on the GPU and only the ~33 MB
#   of seam windows change per frame.
METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// prm: [w, h, lens]   lens planes are vstacked: lens 0 rows 0..L-1, lens 1 rows L..2L-1
kernel void stitch_equirect(device const short* lens   [[buffer(0)]],
                            device const short* x0     [[buffer(1)]],
                            device const short* y0     [[buffer(2)]],
                            device const short* x1     [[buffer(3)]],
                            device const short* y1     [[buffer(4)]],
                            device const uchar* mask   [[buffer(5)]],
                            device short* dst          [[buffer(6)]],
                            device const float* prm    [[buffer(7)]],
                            uint idx [[thread_position_in_grid]]) {
    const uint w = (uint)prm[0];
    const uint h = (uint)prm[1];
    const int L = (int)prm[2];
    const uint plane = w * h;
    if (idx >= plane) return;
    const uint lplane = (uint)(2 * L * L);
    const int xa = x0[idx], ya = y0[idx], xb = x1[idx], yb = y1[idx];
    const bool va = xa >= 0 && xa < L && ya >= 0 && ya < L;
    const bool vb = xb >= 0 && xb < L && yb >= 0 && yb < L;
    const uint pa = (uint)(ya * L + xa);
    const uint pb = (uint)((L + yb) * L + xb);
    const int m = (int)mask[idx] * 4;
    for (uint p = 0; p < 3; p++) {
        const int a = va ? (int)lens[p * lplane + pa] : 0;
        const int b = vb ? (int)lens[p * lplane + pb] : 0;
        dst[p * plane + idx] = (short)((a * (1023 - m) + b * m + 511) / 1023);
    }
}

inline float4 lagrange(float t) {
    const float tt = t * t;
    const float ttt = tt * t;
    return float4(-t / 3.0f + tt / 2.0f - ttt / 6.0f,
                  1.0f - t / 2.0f - tt + ttt / 2.0f,
                  t + tt / 2.0f - ttt / 2.0f,
                  -t / 6.0f + ttt / 6.0f);
}

// prm: [w, h, max_value, m00 m01 m02 m10 m11 m12 m20 m21 m22]
kernel void rotate_equirect(device const short* src   [[buffer(0)]],
                            device short* dst         [[buffer(1)]],
                            device const float* prm   [[buffer(2)]],
                            uint idx [[thread_position_in_grid]]) {
    const uint w = (uint)prm[0];
    const uint h = (uint)prm[1];
    const uint plane = w * h;
    if (idx >= plane) return;
    const float maxv = prm[2];
    const uint i = idx % w;
    const uint j = idx / w;
    const float phi = ((2.0f * i + 1.0f) / w - 1.0f) * M_PI_F;
    const float theta = ((2.0f * j + 1.0f) / h - 1.0f) * M_PI_2_F;
    const float st = precise::sin(theta);
    const float ct = precise::cos(theta);
    const float3 d = float3(ct * precise::sin(phi), st, ct * precise::cos(phi));
    const float rx = prm[3] * d.x + prm[4] * d.y + prm[5] * d.z;
    const float ry = prm[6] * d.x + prm[7] * d.y + prm[8] * d.z;
    const float rz = prm[9] * d.x + prm[10] * d.y + prm[11] * d.z;
    const float phi2 = precise::atan2(rx, rz);
    const float theta2 = precise::asin(clamp(ry, -1.0f, 1.0f));
    const float uf = (phi2 / M_PI_F + 1.0f) * ((w - 1) * 0.5f);
    const float vf = (theta2 / M_PI_2_F + 1.0f) * ((h - 1) * 0.5f);
    const float uff = floor(uf);
    const float vff = floor(vf);
    const int ui = (int)uff;
    const int vi = (int)vff;
    const float4 cu = lagrange(uf - uff);
    const float4 cv = lagrange(vf - vff);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    const int W = (int)w, H = (int)h;
    for (int a = 0; a < 4; a++) {
        int rr = vi + a - 1;
        int shift = 0;
        if (rr < 0) { rr = -rr - 1; shift = W / 2; }
        else if (rr >= H) { rr = 2 * H - 1 - rr; shift = W / 2; }
        const uint base = (uint)rr * w;
        for (int b = 0; b < 4; b++) {
            int cc = ui + b - 1 + shift;
            cc = ((cc % W) + W) % W;
            const uint p = base + (uint)cc;
            const float wgt = cv[a] * cu[b];
            a0 += (float)src[p] * wgt;
            a1 += (float)src[p + plane] * wgt;
            a2 += (float)src[p + 2 * plane] * wgt;
        }
    }
    dst[idx] = (short)clamp(floor(a0), 0.0f, maxv);
    dst[idx + plane] = (short)clamp(floor(a1), 0.0f, maxv);
    dst[idx + 2 * plane] = (short)clamp(floor(a2), 0.0f, maxv);
}
"""


_METAL_LIB = None


def metal_lib():
    """The compiled Metal library (both kernels), built once per process."""
    global _METAL_LIB
    if _METAL_LIB is None:
        _METAL_LIB = _torch().mps.compile_shader(METAL_SRC)
    return _METAL_LIB


def stitch_frame_np(lens, xm, ym, mask, L):
    """numpy statement of ffmpeg's remap x2 + maskedmerge (bit-exact, see
    METAL_SRC). lens: (3, 2L, L) uint16 vstacked lens frames; xm/ym: two
    (h, w) uint16 maps each; mask: (h, w) uint8. Returns (3, h, w) uint16."""
    def remap(img, x, y):
        valid = (x < L) & (y < L)
        xs = np.where(valid, x, 0).astype(np.int64)
        ys = np.where(valid, y, 0).astype(np.int64)
        out = img[:, ys, xs].astype(np.int64)
        out[:, ~valid] = 0
        return out
    e0 = remap(lens[:, :L], xm[0], ym[0])
    e1 = remap(lens[:, L:], xm[1], ym[1])
    m = mask.astype(np.int64) * 4
    return ((e0 * (1023 - m) + e1 * m + 511) // 1023).astype(np.uint16)


class GpuStitcher:
    """remap x2 + maskedmerge on the Apple GPU (Metal), maps resident on the
    device; per frame only the seam windows from SeamFlowWriter are pasted
    in (~33 MB instead of a 236 MB fifo frame) before one kernel launch."""

    def __init__(self, xm, ym, mask, L, out_w, out_h):
        tc = self.torch = _torch()
        self.lib = metal_lib()
        self.L, self.w, self.h = L, out_w, out_h
        self.dev = 'mps'
        self.maps = [tc.from_numpy(np.ascontiguousarray(a).view(np.int16)).to(self.dev)
                     for a in (xm[0], ym[0], xm[1], ym[1])]
        self.mask = tc.from_numpy(np.ascontiguousarray(mask)).to(self.dev)
        self.prm = tc.tensor([float(out_w), float(out_h), float(L)],
                             dtype=tc.float32, device=self.dev)
        self.t_maps = self.t_upload = self.t_stitch = 0.0

    def update_maps(self, wins, seams, rows):
        """Paste this frame's seam windows ([stream][seam] uint16 arrays,
        streams x0 y0 x1 y1) into the resident maps."""
        tc = self.torch
        t0 = time.perf_counter()
        # one upload for all eight windows (eight small ones cost 18 ms of
        # launch/copy overhead per frame; one 33 MB copy is a few ms), then
        # device-side slice assignments
        flat = [np.ascontiguousarray(win).view(np.int16).reshape(-1)
                for w in wins for win in w]
        sizes = [f.size for f in flat]
        big = tc.from_numpy(np.concatenate(flat)).to(self.dev)
        off = 0
        i = 0
        for k, w in enumerate(wins):
            for seam, win in zip(seams, w):
                t = big[off:off + sizes[i]].view(win.shape)
                off += sizes[i]
                i += 1
                if seam.get('contig'):
                    self.maps[k][rows, seam['c0']:seam['c1']] = t
                else:
                    cols = tc.from_numpy(seam['cols'].astype(np.int64)).to(self.dev)
                    self.maps[k][rows, cols] = t
        self.t_maps += time.perf_counter() - t0

    def stitch(self, frame):
        """frame: (3, 2L, L) uint16 numpy. Returns (3, h, w) int16 tensor on
        the device (10-bit gbrp planes)."""
        tc = self.torch
        t0 = time.perf_counter()
        src = tc.from_numpy(frame.view(np.int16)).to(self.dev)
        tc.mps.synchronize()
        t1 = time.perf_counter()
        dst = tc.empty((3, self.h, self.w), dtype=tc.int16, device=self.dev)
        self.lib.stitch_equirect(src, *self.maps, self.mask, dst, self.prm,
                                 threads=self.w * self.h)
        tc.mps.synchronize()
        self.t_upload += t1 - t0
        self.t_stitch += time.perf_counter() - t1
        return dst

    def timing(self, n):
        if n <= 0:
            return ''
        k = 1000.0 / n
        return (f"upload {self.t_upload * k:.0f} + seam maps {self.t_maps * k:.0f} "
                f"+ stitch {self.t_stitch * k:.0f}")


class EquirectRotator:
    """Rotates 10-bit equirect frames by a 3x3 matrix, sampling exactly as
    ffmpeg's v360 does, on a GPU (torch mps/cuda), torch-CPU or numpy.

    Why: v360 driven by sendcmd regenerates its entire lookup table on EVERY
    command, and the stabiliser sends four per frame (reset_rot, yaw, pitch,
    roll). Measured at 8K on 16 cores: 0.22 s per command, 0.99 s per frame
    with four -- the whole 0.9 fps of a host render, while the hardware
    encoder idled and linear/cubic made no difference (the cost is per-pixel
    trig, not table size). Here the same resample is one small matmul, a
    handful of elementwise ops and 48 gathers per frame.

    Every convention below was MEASURED against v360's output on a real
    frame rather than read off its source; the two paths agree to 1 LSB
    on >99.99% of samples (max 5/1023 on a few contrasty pixels):
      * output pixel (i, j) looks along phi = ((2i+1)/W - 1) pi,
        theta = ((2j+1)/H - 1) pi/2, d = (cos t sin p, sin t, cos t cos p),
        y down;
      * the rotated direction is read at u = (phi'/pi + 1) (W-1)/2,
        v = (theta'/(pi/2) + 1) (H-1)/2. v360 scales the INPUT side by
        W-1 / H-1, so its zero rotation is a 0.5 px shift at the frame
        edges and 0 at the centre. Replicated on purpose: a torch render
        and a v360 render of the same clip are interchangeable;
      * 4x4 Lagrange cubic (what v360 calls cubic; Keys was 5x worse),
        TRUNCATED to the integer (v360 shifts its 14-bit fixed-point sum
        without rounding -- rint left 44% of samples 1 LSB high, floor 7%);
      * columns wrap; rows reflect across the poles with a 180 deg shift.
    """

    def __init__(self, w, h, device='numpy', max_value=1023, chunk_rows=None,
                 metal=True):
        self.w, self.h = w, h
        self.max_value = max_value
        self.torch = None if device == 'numpy' else _torch()
        self.device = device
        if self.torch is None and device != 'numpy':
            raise RuntimeError('torch is not installed')
        # fused Metal kernel on Apple GPUs (falls back to the generic torch
        # path if this torch cannot compile shaders or the first call fails)
        self.metal = None
        if metal and device == 'mps' and hasattr(self.torch.mps, 'compile_shader'):
            try:
                self.metal = metal_lib()
            except Exception as e:          # noqa: BLE001 -- any compile failure
                print(f"  ! Metal kernels unavailable ({e}); using the generic "
                      "torch path", file=sys.stderr)
        # a GPU is happiest with few big launches; CPUs with cache-sized rows
        self.chunk = chunk_rows or (1024 if device in ('mps', 'cuda') else 128)
        self._dirs = None               # built on first generic-path use (354 MB at 8K)
        self.su = (w - 1) / 2.0
        self.sv = (h - 1) / 2.0
        self.t_h2d = self.t_rot = self.t_d2h = 0.0

    @property
    def dirs(self):
        if self._dirs is None:
            w, h = self.w, self.h
            i = np.arange(w, dtype=np.float64)
            j = np.arange(h, dtype=np.float64)
            phi = ((2 * i + 1) / w - 1) * math.pi
            theta = ((2 * j + 1) / h - 1) * math.pi / 2
            dirs = np.empty((h, w, 3), np.float32)
            dirs[..., 0] = np.cos(theta)[:, None] * np.sin(phi)[None, :]
            dirs[..., 1] = np.sin(theta)[:, None]
            dirs[..., 2] = np.cos(theta)[:, None] * np.cos(phi)[None, :]
            self._dirs = self._dev(dirs)
        return self._dirs

    def describe(self):
        if self.torch is None:
            return 'numpy'
        name = self.device
        if name == 'mps':
            name = 'Apple GPU (Metal kernel)' if self.metal else 'Apple GPU (torch mps)'
        elif name == 'cuda':
            name = f'{self.torch.cuda.get_device_name(0)} (torch cuda)'
        else:
            name = f'torch cpu, {self.torch.get_num_threads()} threads'
        return name

    # -- backend shims (torch and numpy spell these differently)
    def _dev(self, a):
        if self.torch is None:
            return np.ascontiguousarray(a)
        return self.torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

    def _zeros(self, shape):
        if self.torch is None:
            return np.zeros(shape, np.float32)
        return self.torch.zeros(shape, dtype=self.torch.float32, device=self.device)

    def _where(self, c, a, b):
        return np.where(c, a, b) if self.torch is None else self.torch.where(c, a, b)

    def _floor_int(self, a):
        if self.torch is None:
            f = np.floor(a)
            return f, f.astype(np.int64)
        f = self.torch.floor(a)
        return f, f.to(self.torch.int64)

    @staticmethod
    def _lagrange(t):
        tt = t * t
        ttt = tt * t
        return (-t / 3.0 + tt / 2.0 - ttt / 6.0,
                1.0 - t / 2.0 - tt + ttt / 2.0,
                t + tt / 2.0 - ttt / 2.0,
                -t / 6.0 + ttt / 6.0)

    def rotate(self, planes, M):
        """planes: list of (h, w) float32 arrays on the backend; M: 3x3 numpy
        (see stab_matrix). Returns the rotated planes, float32, already
        truncated and clipped to 0..max_value."""
        tc, w, h = self.torch, self.w, self.h
        xp = np if tc is None else tc
        Rt = self._dev(np.ascontiguousarray(np.asarray(M).T, dtype=np.float32))
        outs = [self._zeros((h, w)) for _ in planes]
        flats = [p.reshape(-1) for p in planes]
        for r0 in range(0, h, self.chunk):
            r1 = min(r0 + self.chunk, h)
            d = self.dirs[r0:r1].reshape(-1, 3) @ Rt          # = M d per row
            phi = xp.arctan2(d[:, 0], d[:, 2]) if tc is None else tc.atan2(d[:, 0], d[:, 2])
            y = d[:, 1]
            y = np.clip(y, -1.0, 1.0) if tc is None else tc.clamp(y, -1.0, 1.0)
            theta = np.arcsin(y) if tc is None else tc.asin(y)
            uf = (phi / math.pi + 1.0) * self.su
            vf = (theta / (math.pi / 2) + 1.0) * self.sv
            uff, ui = self._floor_int(uf)
            vff, vi = self._floor_int(vf)
            cu = self._lagrange(uf - uff)
            cv = self._lagrange(vf - vff)
            acc = [self._zeros((r1 - r0) * w) for _ in planes]
            for a in range(4):
                rr = vi + (a - 1)
                below = rr < 0
                above = rr >= h
                rr_ref = self._where(below, -rr - 1,
                                     self._where(above, 2 * h - 1 - rr, rr))
                shift = self._where(below | above, w // 2, 0 * rr)
                base = rr_ref * w
                for b in range(4):
                    cc = ui + (b - 1) + shift
                    cc = np.mod(cc, w) if tc is None else tc.remainder(cc, w)
                    idx = base + cc
                    wgt = cv[a] * cu[b]
                    for p, fl in enumerate(flats):
                        acc[p] = acc[p] + fl[idx] * wgt
            for p in range(len(planes)):
                v = acc[p]
                v = np.floor(v) if tc is None else tc.floor(v)
                v = (np.clip(v, 0, self.max_value) if tc is None
                     else tc.clamp(v, 0, self.max_value))
                outs[p][r0:r1] = v.reshape(r1 - r0, w)
        return outs

    def rotate_frame(self, frame, M):
        """frame: (3, h, w) uint16 numpy (gbrp10le planes). Returns the
        rotated frame as (3, h, w) uint16 numpy."""
        tc = self.torch
        if tc is None:
            t0 = time.perf_counter()
            planes = [p.astype(np.float32) for p in frame]
            out = np.stack(self.rotate(planes, M)).astype(np.uint16)
            self.t_rot += time.perf_counter() - t0
            return out
        # 10-bit samples are < 32768, so the uint16 bytes viewed as int16 are
        # the same numbers -- and int16 is a dtype every torch backend moves
        # and casts, which uint16 is not.
        t0 = time.perf_counter()
        if self.metal is not None:
            src = tc.from_numpy(frame.view(np.int16)).to(self.device)
            self._sync()
            t1 = time.perf_counter()
            try:
                o = self._rotate_metal(src, M)
            except Exception as e:      # noqa: BLE001 -- kernel launch failed
                print(f"  ! Metal kernel failed ({e}); using the generic "
                      "torch path", file=sys.stderr)
                self.metal = None
                o = tc.stack(self.rotate([p.to(tc.float32) for p in src], M)).to(tc.int16)
            self._sync()
            t2 = time.perf_counter()
            o = o.cpu().numpy().view(np.uint16)
            t3 = time.perf_counter()
            self.t_h2d += t1 - t0
            self.t_rot += t2 - t1
            self.t_d2h += t3 - t2
            return o
        t = tc.from_numpy(frame.view(np.int16)).to(self.device).to(tc.float32)
        self._sync()
        t1 = time.perf_counter()
        outs = self.rotate([t[0], t[1], t[2]], M)
        o = tc.stack(outs).to(tc.int16)
        self._sync()
        t2 = time.perf_counter()
        o = o.cpu().numpy().view(np.uint16)
        t3 = time.perf_counter()
        self.t_h2d += t1 - t0
        self.t_rot += t2 - t1
        self.t_d2h += t3 - t2
        return o

    def rotate_device(self, src, M):
        """src: (3, h, w) int16 tensor already on the device. Returns the
        rotated int16 tensor (Metal kernel, or the generic path)."""
        tc = self.torch
        t0 = time.perf_counter()
        if self.metal is not None:
            o = self._rotate_metal(src, M)
        else:
            o = tc.stack(self.rotate([p.to(tc.float32) for p in src], M)).to(tc.int16)
        self._sync()
        self.t_rot += time.perf_counter() - t0
        return o

    def _rotate_metal(self, src, M):
        """src: (3, h, w) int16 tensor on mps. Returns the rotated (3, h, w)
        int16 tensor."""
        tc = self.torch
        src = src.contiguous()
        dst = tc.empty_like(src)
        prm = tc.tensor([float(self.w), float(self.h), float(self.max_value)]
                        + [float(x) for x in np.asarray(M, dtype=np.float64).ravel()],
                        dtype=tc.float32, device=self.device)
        self.metal.rotate_equirect(src, dst, prm, threads=self.w * self.h)
        return dst

    def _sync(self):
        tc = self.torch
        if self.device == 'mps':
            tc.mps.synchronize()
        elif self.device == 'cuda':
            tc.cuda.synchronize()

    def timing(self, n):
        """Per-frame stage times in ms after n frames."""
        if n <= 0:
            return ''
        k = 1000.0 / n
        if self.torch is None:
            return f"rotate {self.t_rot * k:.0f} ms/frame"
        return (f"upload {self.t_h2d * k:.0f} + rotate {self.t_rot * k:.0f} "
                f"+ download {self.t_d2h * k:.0f} ms/frame")


def run_rotated(cmd1, cmd2, rotator, mats, w, h, transport='pipe', work=None):
    """decode/stitch ffmpeg -> rotate every frame -> encode ffmpeg."""
    def process(frame, n):
        return rotator.rotate_frame(frame, mats[min(n, len(mats) - 1)])
    return run_piped(cmd1, cmd2, (3, h, w), process,
                     lambda n: rotator.timing(n), transport, work)


SOCK_BUF = 4 << 20


def _listen_unix(path):
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    srv.settimeout(120)
    return srv


def _accept(srv, opt):
    conn, _ = srv.accept()
    conn.settimeout(None)
    try:
        conn.setsockopt(socket.SOL_SOCKET, opt, SOCK_BUF)
    except OSError:
        pass
    srv.close()
    return conn


def run_piped(cmd1, cmd2, in_shape, process, timing=None, transport='pipe',
              work=None):
    """cmd1 writes rawvideo uint16 frames of in_shape to 'pipe:1'; every
    frame goes through process(frame, index) -> (3, h, w) uint16; cmd2 reads
    the results from 'pipe:0'. Reader and writer threads keep both transports
    busy while the main thread processes; queues are shallow because a frame
    is up to 177 MB at 8K. Returns the number of frames processed.

    transport: 'pipe' (both directions), 'socket' (both unix-domain sockets,
    ffmpeg's `unix:` protocol, ffmpeg connecting to sockets this process
    listens on) or 'mixed' = pipe in, socket out -- the default, from
    measurements on an M4 Max (tools/transport_bench.py): a pipe moves a
    177 MB frame in 23 ms, a unix socket in 109 ms with default buffers but
    8 ms once the SENDER sets a multi-MB SO_SNDBUF. This process can set
    that on the socket it sends on (to the encoder), but not inside ffmpeg,
    so the decoder->here direction stays a pipe (an all-socket run fell to
    5 fps waiting on the decoder). Output is bit-identical either way.
    """
    frame_bytes = int(np.prod(in_shape)) * 2
    srv_in = srv_out = None
    sock_in = transport == 'socket'
    sock_out = transport in ('socket', 'mixed')
    if sock_in or sock_out:
        work = work or tempfile.mkdtemp(prefix='osvsock_')
    if sock_in:
        in_path = os.path.join(work, 'in.sock')
        srv_in = _listen_unix(in_path)
        cmd1 = [f'unix:{in_path}' if a == 'pipe:1' else a for a in cmd1]
    if sock_out:
        out_path_ = os.path.join(work, 'out.sock')
        srv_out = _listen_unix(out_path_)
        cmd2 = [f'unix:{out_path_}' if a == 'pipe:0' else a for a in cmd2]
    p1 = subprocess.Popen(cmd1, stdout=None if sock_in else subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=None if sock_out else subprocess.PIPE)
    inq = queue.Queue(maxsize=2)
    outq = queue.Queue(maxsize=2)
    dead = threading.Event()

    def reader():
        conn = None
        try:
            if srv_in:
                conn = _accept(srv_in, socket.SO_RCVBUF)
                read = conn.recv_into
            else:
                read = p1.stdout.readinto
            while not dead.is_set():
                buf = bytearray(frame_bytes)
                view = memoryview(buf)
                got = 0
                while got < frame_bytes:
                    n = read(view[got:])
                    if not n:
                        break
                    got += n
                if got < frame_bytes:       # EOF; ffmpeg only writes whole frames
                    break
                inq.put(buf)
        except OSError:
            dead.set()
        finally:
            inq.put(None)
            if conn:
                conn.close()

    def writer():
        conn = None
        try:
            if srv_out:
                conn = _accept(srv_out, socket.SO_SNDBUF)
                write = conn.sendall
            else:
                write = p2.stdin.write
            while True:
                b = outq.get()
                if b is None:
                    break
                write(memoryview(b))
        except (BrokenPipeError, OSError):
            dead.set()
        finally:
            try:
                if conn:
                    conn.close()
                else:
                    p2.stdin.close()
            except OSError:
                pass

    threads = [threading.Thread(target=reader, daemon=True),
               threading.Thread(target=writer, daemon=True)]
    for t in threads:
        t.start()
    n = 0
    t_in = t_out = 0.0
    t_start = time.perf_counter()
    try:
        while not dead.is_set():
            t0 = time.perf_counter()
            buf = inq.get()
            t_in += time.perf_counter() - t0
            if buf is None:
                break
            frame = np.frombuffer(buf, np.uint16).reshape(in_shape)
            out = process(frame, n)
            t0 = time.perf_counter()
            outq.put(out)
            t_out += time.perf_counter() - t0
            n += 1
    finally:
        outq.put(None)
        if dead.is_set():
            p1.terminate()
        threads[1].join()
        if p1.stdout:
            p1.stdout.close()
        rc1 = p1.wait()
        rc2 = p2.wait()
        threads[0].join(timeout=5)
    if rc1:
        raise subprocess.CalledProcessError(rc1, cmd1)
    if rc2:
        raise subprocess.CalledProcessError(rc2, cmd2)
    if n:
        # where the time went: waiting on the stitch side (decode + remap +
        # maskedmerge + seam maps), the rotation itself, or the encoder side
        wall = time.perf_counter() - t_start
        print(f"  pipeline: {n / wall:.2f} fps; per frame: waited for decode "
              f"{t_in / n * 1000:.0f} ms, {timing(n) if timing else ''}, "
              f"waited for encoder {t_out / n * 1000:.0f} ms")
    return n
