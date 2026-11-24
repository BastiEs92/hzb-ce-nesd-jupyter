"""
bruker_raw.py  --  minimal, dependency-light reader for Bruker RAW1.01 (ver.3) powder files.

Why this exists:
    GSAS-II has its own Bruker RAW importer and we use it first. But its "ver.3"
    reader occasionally fails on files from some diffractometers. This module is a
    robust fallback that (a) reads the header fields we verified on Basti's files
    and (b) auto-detects them if the fixed offsets ever move, then writes a plain
    2-column .xy that *any* program can read.

Verified on HZB "Base.raw" (Dr. Tovar, Cu Kalpha):
    int32  @ 716  -> number of points N
    float64@ 728  -> start 2theta (deg)
    float64@ 888  -> step size (deg)
    float32 @1056 -> first intensity (N contiguous float32 values)

Usage:
    python bruker_raw.py pattern.raw            # -> pattern.xy  (+ prints header)
    from bruker_raw import read_raw
    x, y, meta = read_raw("pattern.raw")
"""
import struct, sys, os
import numpy as np


def _try_fixed_offsets(b):
    """Read the header at the offsets verified on the HZB instrument."""
    N     = struct.unpack_from("<i", b, 716)[0]
    start = struct.unpack_from("<d", b, 728)[0]
    step  = struct.unpack_from("<d", b, 888)[0]
    data_off = 1056
    ok = (100 <= N <= 100000 and 0 < start < 20 and 1e-4 < step < 1.0
          and data_off + 4 * N <= len(b))
    return (N, start, step, data_off) if ok else None


def _good_f64(v):
    """True for a normal float64 we'd accept as an angle (reject NaN/denormals)."""
    return v == v and (v == 0.0 or abs(v) >= 1e-6) and abs(v) < 1e6


def _auto_detect(b):
    """
    Fallback if the fixed offsets don't validate (different firmware / extra
    header blocks). Strategy:
      - step  : first plausible float64 angular step in the header.
      - start : the documented float64 @728 if valid, else first small float64.
      - data  : the first LONG run (>=200) of consecutive real counts, with
                leading zero/padding trimmed by requiring a sustained signal.
    Raises ValueError with guidance if nothing sensible is found.
    """
    n = len(b)
    step = None
    for off in range(8, min(n - 8, 2000), 4):
        v = struct.unpack_from("<d", b, off)[0]
        if 0.002 <= v <= 0.2 and _good_f64(v):
            step = v
            break
    arr = np.frombuffer(b[: (n // 4) * 4], dtype="<f4")
    # Real counts form the longest run of STRICTLY POSITIVE, finite values.
    # This naturally skips the zero padding and any isolated stray value in it.
    pos = np.isfinite(arr) & (arr > 0) & (arr < 5e8)
    best_start, best_len, cur, cur_start = 0, 0, 0, 0
    for i, f in enumerate(pos):
        if f:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, cur_start
        else:
            cur = 0
    if best_len < 200 or step is None:
        raise ValueError("Auto-detect failed to locate the data block; let "
                         "GSAS-II's Bruker importer read this file, or export .xy "
                         "from DIFFRAC.EVA / the instrument software.")
    data_idx, N = best_start, best_len
    data_off = data_idx * 4
    start = struct.unpack_from("<d", b, 728)[0]
    if not (0.0 <= start <= 20.0 and _good_f64(start)):
        start = 0.0
        for off in range(8, min(n - 8, 2000), 4):
            v = struct.unpack_from("<d", b, off)[0]
            if 0.0 < v <= 20.0 and _good_f64(v) and abs(v - step) > 1e-3:
                start = v
                break
    return (N, start, step, data_off)


def read_raw(path):
    """Return (two_theta, intensity, meta). meta has start/step/N and operator/sample if present."""
    with open(path, "rb") as fh:
        b = fh.read()
    if b[:7] != b"RAW1.01":
        raise ValueError(f"{path}: not a Bruker RAW1.01 file (magic={b[:7]!r}). "
                         "Use GSAS-II's importer for other RAW versions.")
    hdr = _try_fixed_offsets(b) or _auto_detect(b)
    N, start, step, data_off = hdr
    y = np.frombuffer(b[data_off : data_off + 4 * N], dtype="<f4").astype(float)
    x = start + np.arange(len(y)) * step
    # pull a couple of human-readable header strings (operator / sample), best-effort
    def _txt(off, length):
        try:
            return b[off:off + length].split(b"\x00")[0].decode("latin-1", "ignore").strip()
        except Exception:
            return ""
    meta = dict(N=len(y), start=start, step=step, data_off=data_off,
                end=float(x[-1]) if len(x) else start,
                operator=_txt(34, 36), sample=_txt(326, 60))
    return x, y, meta


def write_xy(path, x, y, header=""):
    """Write a 2-column .xy (2theta, intensity) that GSAS-II/Topas/etc. read."""
    with open(path, "w") as fh:
        if header:
            fh.write(f"# {header}\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:.5f} {yi:.3f}\n")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        x, y, meta = read_raw(p)
        out = os.path.splitext(p)[0] + ".xy"
        write_xy(out, x, y, header=f"{os.path.basename(p)} start={meta['start']} step={meta['step']} N={meta['N']}")
        print(f"{p}: start={meta['start']:.4f} step={meta['step']:.6f} N={meta['N']} "
              f"range {x[0]:.2f}-{x[-1]:.2f} -> {out}")
