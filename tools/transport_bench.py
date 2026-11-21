"""How fast can 177 MB frames move between two processes on this machine?
pipe vs unix-domain socket, various buffer sizes. Run: python3 tools/transport_bench.py"""
import os, socket, sys, time, multiprocessing as mp

FRAME = 3 * 7680 * 3840 * 2
N = 12

def sink_pipe(r):
    buf = bytearray(FRAME); view = memoryview(buf); f = os.fdopen(r, 'rb', buffering=0)
    for _ in range(N):
        got = 0
        while got < FRAME:
            n = f.readinto(view[got:]); got += n
            if not n: return

def sink_sock(path, bufsz):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(path); srv.listen(1)
    c, _ = srv.accept()
    if bufsz: 
        try: c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, bufsz)
        except OSError as e: print('  rcvbuf', bufsz, 'refused:', e)
    buf = bytearray(FRAME); view = memoryview(buf)
    for _ in range(N):
        got = 0
        while got < FRAME:
            n = c.recv_into(view[got:]); got += n
            if not n: return

def run_pipe():
    r, w = os.pipe(); p = mp.Process(target=sink_pipe, args=(r,)); p.start(); os.close(r)
    data = memoryview(bytearray(FRAME)); f = os.fdopen(w, 'wb', buffering=0)
    t = time.perf_counter()
    for _ in range(N): f.write(data)
    f.close(); p.join(); return time.perf_counter() - t

def run_sock(bufsz):
    path = '/tmp/tb_%d.sock' % os.getpid()
    if os.path.exists(path): os.unlink(path)
    p = mp.Process(target=sink_sock, args=(path, bufsz)); p.start()
    while not os.path.exists(path): time.sleep(0.01)
    time.sleep(0.05)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.connect(path)
    if bufsz:
        try: c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, bufsz)
        except OSError as e: print('  sndbuf', bufsz, 'refused:', e)
    data = memoryview(bytearray(FRAME))
    t = time.perf_counter()
    for _ in range(N): c.sendall(data)
    c.close(); p.join(); os.unlink(path); return time.perf_counter() - t

if __name__ == '__main__':
    mp.set_start_method('fork')
    total = FRAME * N / 1e9
    dt = run_pipe(); print(f"pipe                 : {total/dt:5.2f} GB/s  ({dt/N*1000:.0f} ms per 177 MB frame)")
    for b in (0, 1 << 20, 4 << 20, 8 << 20):
        dt = run_sock(b); print(f"unix socket buf {b>>20:2d} MB: {total/dt:5.2f} GB/s  ({dt/N*1000:.0f} ms per frame)")
