#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sfd_curve_fit.py
Robust SFD (FontForge) curve fitting for Hershey-like polylines.

- Parses SplineSet blocks tolerant of indentation and trailing flags
  (e.g., "187.5 147.5 m 1", "c 0/1/2").
- Preserves existing cubic segments unless --force is given.
- Emits valid SFD commands with trailing flags so glyphs remain visible in FontForge.
- Uses a Schneider-style cubic fitting (chord-length parameterization + refinement).
"""

import re
import math
import argparse

# -----------------------------
# Vector and cubic utilities
# -----------------------------


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def v_mul(a, s):
    return (a[0] * s, a[1] * s)


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def v_len(a):
    return math.hypot(a[0], a[1])


def v_norm(a):
    l = v_len(a)
    return (a[0] / l, a[1] / l) if l != 0 else (0.0, 0.0)


def bez_eval(p0, p1, p2, p3, t):
    mt = 1 - t
    return (
        (mt**3) * p0[0]
        + 3 * (mt**2) * t * p1[0]
        + 3 * mt * (t**2) * p2[0]
        + (t**3) * p3[0],
        (mt**3) * p0[1]
        + 3 * (mt**2) * t * p1[1]
        + 3 * mt * (t**2) * p2[1]
        + (t**3) * p3[1],
    )


def bez_deriv(p0, p1, p2, p3, t):
    mt = 1 - t
    return (
        3 * mt * mt * (p1[0] - p0[0])
        + 6 * mt * t * (p2[0] - p1[0])
        + 3 * t * t * (p3[0] - p2[0]),
        3 * mt * mt * (p1[1] - p0[1])
        + 6 * mt * t * (p2[1] - p1[1])
        + 3 * t * t * (p3[1] - p2[1]),
    )


def chord_length_u(points, first, last):
    u = [0.0]
    for i in range(first + 1, last + 1):
        u.append(u[-1] + v_len(v_sub(points[i], points[i - 1])))
    total = u[-1] if u[-1] > 0 else 1.0
    return [val / total for val in u]


def generate_bezier(points, first, last, u, tan_l, tan_r):
    p0 = points[first]
    p3 = points[last]
    n = last - first + 1

    A = []
    for i in range(n):
        t = u[i]
        mt = 1 - t
        b1 = 3 * t * (mt**2)
        b2 = 3 * (t**2) * mt
        A.append((v_mul(tan_l, b1), v_mul(tan_r, b2)))

    C = [[0.0, 0.0], [0.0, 0.0]]
    X = [0.0, 0.0]
    for i in range(n):
        a1, a2 = A[i]
        C[0][0] += v_dot(a1, a1)
        C[0][1] += v_dot(a1, a2)
        C[1][0] += v_dot(a1, a2)
        C[1][1] += v_dot(a2, a2)
        t = u[i]
        mt = 1 - t
        s = (
            p0[0] * (mt**3) + p3[0] * (t**3),
            p0[1] * (mt**3) + p3[1] * (t**3),
        )
        tmp = v_sub(points[first + i], s)
        X[0] += v_dot(a1, tmp)
        X[1] += v_dot(a2, tmp)

    det = C[0][0] * C[1][1] - C[0][1] * C[1][0]
    if abs(det) > 1e-12:
        inv = [[C[1][1] / det, -C[0][1] / det], [-C[1][0] / det, C[0][0] / det]]
        alpha_l = inv[0][0] * X[0] + inv[0][1] * X[1]
        alpha_r = inv[1][0] * X[0] + inv[1][1] * X[1]
    else:
        dist = v_len(v_sub(p3, p0)) / 3.0
        alpha_l = alpha_r = dist

    seg_len = v_len(v_sub(p3, p0))
    eps = seg_len / 1000.0 if seg_len > 0 else 1e-6
    alpha_l = max(alpha_l, eps)
    alpha_r = max(alpha_r, eps)

    p1 = v_add(p0, v_mul(tan_l, alpha_l))
    p2 = v_add(p3, v_mul(tan_r, -alpha_r))
    return p0, p1, p2, p3


def compute_max_error(points, first, last, bez, u):
    p0, p1, p2, p3 = bez
    max_err = 0.0
    split = (first + last) // 2
    for i in range(1, last - first):
        t = u[i]
        p = bez_eval(p0, p1, p2, p3, t)
        err = v_len(v_sub(p, points[first + i]))
        if err > max_err:
            max_err = err
            split = first + i
    return max_err, split


def reparameterize(points, first, last, u, bez):
    p0, p1, p2, p3 = bez

    def newton(t, point):
        q = bez_eval(p0, p1, p2, p3, t)
        q1 = bez_deriv(p0, p1, p2, p3, t)
        q2 = (
            6
            * ((1 - t) * (p2[0] - 2 * p1[0] + p0[0]) + t * (p3[0] - 2 * p2[0] + p1[0])),
            6
            * ((1 - t) * (p2[1] - 2 * p1[1] + p0[1]) + t * (p3[1] - 2 * p2[1] + p1[1])),
        )
        diff = v_sub(q, point)
        numerator = v_dot(diff, q1)
        denominator = v_dot(q1, q1) + v_dot(diff, q2)
        if denominator == 0:
            return t
        return max(0.0, min(1.0, t - numerator / denominator))

    new_u = [u[0]]
    for i in range(1, last - first):
        new_u.append(newton(u[i], points[first + i]))
    new_u.append(u[-1])
    return new_u


def fit_cubic(points, first, last, tan_l, tan_r, error, curves):
    if last - first == 1:
        p0 = points[first]
        p3 = points[last]
        dist = v_len(v_sub(p3, p0)) / 3.0
        p1 = v_add(p0, v_mul(tan_l, dist))
        p2 = v_add(p3, v_mul(tan_r, -dist))
        curves.append((p0, p1, p2, p3))
        return
    u = chord_length_u(points, first, last)
    bez = generate_bezier(points, first, last, u, tan_l, tan_r)
    for _ in range(12):
        max_err, split = compute_max_error(points, first, last, bez, u)
        if max_err <= error:
            curves.append(bez)
            return
        u = reparameterize(points, first, last, u, bez)
        bez = generate_bezier(points, first, last, u, tan_l, tan_r)
    max_err, split = compute_max_error(points, first, last, bez, u)
    tan_c_l = v_norm(v_sub(points[split + 1], points[split]))
    tan_c_r = v_norm(v_sub(points[split - 1], points[split]))
    fit_cubic(points, first, split, tan_l, tan_c_r, error, curves)
    fit_cubic(points, split, last, tan_c_l, tan_r, error, curves)


def fit_stroke(points, error):
    if len(points) < 2:
        return []
    # dedupe
    pts = [points[0]]
    for p in points[1:]:
        if p != pts[-1]:
            pts.append(p)
    if len(pts) < 2:
        return []
    tan_l = v_norm(v_sub(pts[1], pts[0]))
    tan_r = v_norm(v_sub(pts[-2], pts[-1]))
    curves = []
    fit_cubic(pts, 0, len(pts) - 1, tan_l, tan_r, error, curves)
    return curves


# -----------------------------
# SFD parsing/writing
# -----------------------------

# Matches lines like:
# "187.5 147.5 m 1"
# "352.921461549 -82.614893432 352.921461549 -82.614893432 352.723318355 -81.8223206573 c 0"
RE_CMD = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"  # x y
    r"(?:\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?))?"  # opt x2 y2 (for 'c' first handle)
    r"(?:\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?))?"  # opt x3 y3 (for 'c' second handle or end)
    r"\s+([mMlLcC])\s*(\d+)?"  # command + optional trailing flag int
    r"\s*$"
)


def parse_spline_set(lines):
    """
    Return:
      strokes: list of list of (x,y) floats for m/l runs
      segments: original tokens to preserve if we choose not to refit
      has_cubic: True if 'c' present
    """
    strokes = []
    current = []
    segments = []
    has_cubic = False

    for raw in lines:
        s = raw.strip()
        if not s:
            segments.append(("raw", raw))
            continue
        m = RE_CMD.match(s)
        if not m:
            # Pass through as-is (comments or unknown), keep to preserve verbatim if needed
            segments.append(("raw", raw))
            continue

        # Extract numbers found
        nums = [m.group(i) for i in range(1, 7)]
        nums = [float(n) if n is not None else None for n in nums]
        cmd = m.group(7).lower()
        flag = m.group(8)
        flag = int(flag) if flag is not None else None

        if cmd == "m":
            # Start a new stroke
            if current:
                strokes.append(current)
                current = []
            current = [(nums[0], nums[1])]
            segments.append(("m", (nums[0], nums[1], flag)))
        elif cmd == "l":
            current.append((nums[0], nums[1]))
            segments.append(("l", (nums[0], nums[1], flag)))
        elif cmd == "c":
            has_cubic = True
            # For cubic we keep the entire line verbatim (there are 6 coords before 'c')
            # Note: some files repeat points (as seen in your sample); we preserve them.
            segments.append(("c", (nums, flag)))
        else:
            segments.append(("raw", raw))

    if current:
        strokes.append(current)
    return strokes, segments, has_cubic


def rebuild_spline_set(strokes, error, force, original_segments):
    """
    Build new SplineSet lines. If not force and we detect cubic segments, we preserve original.
    For fitted curves:
      - Emit 'm' with trailing flag 1 (on-curve, reasonable default).
      - Emit 'c' lines with trailing flag 0 (corner-like end); FontForge accepts these.
    """
    # If there are cubic segments and not forcing, preserve original SplineSet
    if not force:
        if any(t[0] == "c" for t in original_segments):
            out = []
            for t in original_segments:
                if t[0] == "raw":
                    out.append(t[1].rstrip())
                elif t[0] == "m":
                    x, y, fl = t[1]
                    fl = 1 if fl is None else fl
                    out.append(f"{fmt(x)} {fmt(y)} m {fl}")
                elif t[0] == "l":
                    x, y, fl = t[1]
                    fl = 1 if fl is None else fl
                    out.append(f"{fmt(x)} {fmt(y)} l {fl}")
                elif t[0] == "c":
                    nums, fl = t[1]
                    fl = 0 if fl is None else fl
                    # nums: [x1,y1,x2,y2,x3,y3]
                    out.append(
                        f"{fmt(nums[0])} {fmt(nums[1])} {fmt(nums[2])} {fmt(nums[3])} {fmt(nums[4])} {fmt(nums[5])} c {fl}"
                    )
            return out

    # Otherwise, fit polylines to cubic
    out = []
    for stroke in strokes:
        if not stroke:
            continue
        # Move
        x0, y0 = stroke[0]
        out.append(f"{fmt(x0)} {fmt(y0)} m 1")
        if len(stroke) == 1:
            continue
        curves = fit_stroke([(float(x), float(y)) for x, y in stroke], error)
        if not curves:
            # If fitter returns nothing (e.g., fully degenerate), fall back to lines
            for p in stroke[1:]:
                out.append(f"{fmt(p[0])} {fmt(p[1])} l 1")
            continue
        for p0, p1, p2, p3 in curves:
            c1x, c1y = round(p1[0], 6), round(p1[1], 6)
            c2x, c2y = round(p2[0], 6), round(p2[1], 6)
            ex, ey = round(p3[0], 6), round(p3[1], 6)
            # Use 'c 0' as a safe default flag
            out.append(
                f"{fmt(c1x)} {fmt(c1y)} {fmt(c2x)} {fmt(c2y)} {fmt(ex)} {fmt(ey)} c 0"
            )
    return out


def fmt(x):
    # Keep integers as integers; otherwise emit up to 6 decimals (like your SFD)
    if abs(x - int(round(x))) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def process_sfd(input_text, error=1.0, force=False):
    lines = input_text.splitlines()
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        out.append(line)
        if stripped.startswith("SplineSet"):
            # Collect block
            block = []
            i += 1
            while i < n and "EndSplineSet" not in lines[i]:
                block.append(lines[i])
                i += 1
            # Parse and rebuild
            strokes, segments, has_cubic = parse_spline_set(block)
            rebuilt = rebuild_spline_set(strokes, error, force, segments)
            # Replace block content
            for ln in rebuilt:
                out.append(ln)
            out.append("EndSplineSet")
        i += 1
    # Preserve trailing newline
    res = "\n".join(out)
    if input_text.endswith("\n") and not res.endswith("\n"):
        res += "\n"
    return res


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Fit cubic curves to SFD polyline strokes (Hershey-like)."
    )
    ap.add_argument("input", help="Input .sfd")
    ap.add_argument("output", help="Output .sfd")
    ap.add_argument(
        "--error",
        type=float,
        default=1.0,
        help="Max fitting error in font units (default 1.0)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Refit even glyphs that already contain cubic segments",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        sfd = f.read()

    out = process_sfd(sfd, error=args.error, force=args.force)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
