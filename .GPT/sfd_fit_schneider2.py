#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sfd_fit_schneider.py — Refit SFD SplineSets to Schneider-style cubics,
now with:
  • Corner/cusp-aware segmentation (preserves serifs & sharp joins)
  • Adaptive flattening of existing cubic segments
  • Optional extrema locking
  • Exact straight-segment cubics (no unintended bowing)

Usage:
  python3 sfd_fit_schneider.py input.sfd [-o output.sfd]
    [--tolerance 0.6] [--flatness 0.25] [--corner-angle 35]
    [--layers fore|all] [--min-stroke 0.0] [--lock-extrema]
    [--verbose]

Notes:
  - Degenerate strokes and unparsable contours are copied verbatim.
  - Existing cubic segments are always resampled (adaptive) and refit.
  - Only Fore SplineSets are processed by default; use --layers all to
    process SplineSets under other Layer: blocks.
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

Point = Tuple[float, float]
EPS = 1e-9

# ----------------------------- tiny vec math ------------------------------


def v_add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def v_scale(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def v_dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def v_norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def v_unit(a: Point) -> Point:
    n = v_norm(a)
    return (0.0, 0.0) if n < EPS else (a[0] / n, a[1] / n)


def almost_eq(a: Point, b: Point, tol: float = 1e-7) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def angle_between(u: Point, v: Point) -> float:
    # degrees between two vectors; 0 = collinear, 180 = U-turn
    nu, nv = v_unit(u), v_unit(v)
    d = max(-1.0, min(1.0, v_dot(nu, nv)))
    return math.degrees(math.acos(d))


# ---------------------- Cubic evaluation & derivatives ---------------------


def bezier_eval(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3 * u * u * t
    b2 = 3 * u * t * t
    b3 = t * t * t
    return (
        b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
        b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
    )


def bezier_eval_dt(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    # First derivative (3 * quadratic)
    u = 1.0 - t
    q0 = v_sub(p1, p0)
    q1 = v_sub(p2, p1)
    q2 = v_sub(p3, p2)
    return (
        3 * (u * u * q0[0] + 2 * u * t * q1[0] + t * t * q2[0]),
        3 * (u * u * q0[1] + 2 * u * t * q1[1] + t * t * q2[1]),
    )


def bezier_eval_ddt(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    # Second derivative (6 * linear)
    a = v_add(v_sub(p2, v_scale(p1, 2.0)), p0)
    b = v_add(v_sub(p3, v_scale(p2, 2.0)), p1)
    return (6 * ((1 - t) * a[0] + t * b[0]), 6 * ((1 - t) * a[1] + t * b[1]))


# ---------------- Schneider (Graphics Gems) fitter, unchanged core ----------


def chord_length_parameterize(pts: List[Point]) -> List[float]:
    u = [0.0]
    total = 0.0
    for i in range(1, len(pts)):
        d = v_norm(v_sub(pts[i], pts[i - 1]))
        total += d
        u.append(total)
    if total < EPS:
        return [0.0 for _ in pts]
    return [ui / total for ui in u]


def reparameterize(
    pts: List[Point], bezi: Tuple[Point, Point, Point, Point], u: List[float]
) -> List[float]:
    p0, p1, p2, p3 = bezi

    def newton(u_k, p_k):
        q = bezier_eval(p0, p1, p2, p3, u_k)
        q1 = bezier_eval_dt(p0, p1, p2, p3, u_k)
        q2 = bezier_eval_ddt(p0, p1, p2, p3, u_k)
        diff = v_sub(q, p_k)
        num = v_dot(diff, q1)
        den = v_dot(q1, q1) + v_dot(diff, q2)
        return u_k if abs(den) < 1e-12 else u_k - num / den

    out = []
    m = len(u)
    for i in range(m):
        if i == 0:
            out.append(0.0)
        elif i == m - 1:
            out.append(1.0)
        else:
            out.append(max(0.0, min(1.0, newton(u[i], pts[i]))))
    return out


def bezier_from_endpoints(
    pts: List[Point], u: List[float], tHat1: Point, tHat2: Point
) -> Tuple[Point, Point, Point, Point]:
    p0 = pts[0]
    p3 = pts[-1]
    n = len(pts)
    A = []
    for i in range(n):
        ui = u[i]
        u1 = 1 - ui
        b1 = 3 * u1 * u1 * ui
        b2 = 3 * u1 * ui * ui
        A.append((v_scale(tHat1, b1), v_scale(tHat2, b2)))
    C00 = C01 = C11 = X0 = X1 = 0.0
    for i in range(n):
        ui = u[i]
        u1 = 1 - ui
        b0 = u1 * u1 * u1
        b3 = ui * ui * ui
        tmp = v_sub(pts[i], v_add(v_scale(p0, b0), v_scale(p3, b3)))
        a0, a1 = A[i]
        C00 += v_dot(a0, a0)
        C01 += v_dot(a0, a1)
        C11 += v_dot(a1, a1)
        X0 += v_dot(a0, tmp)
        X1 += v_dot(a1, tmp)
    det = C00 * C11 - C01 * C01
    if abs(det) < 1e-12:
        seglen = v_norm(v_sub(p3, p0))
        alpha_l = alpha_r = seglen / 3.0
    else:
        alpha_l = (X0 * C11 - X1 * C01) / det
        alpha_r = (C00 * X1 - C01 * X0) / det
        seglen = max(EPS, v_norm(v_sub(p3, p0)))
        max_a = 1e3 * seglen
        if not math.isfinite(alpha_l) or abs(alpha_l) > max_a:
            alpha_l = seglen / 3.0
        if not math.isfinite(alpha_r) or abs(alpha_r) > max_a:
            alpha_r = seglen / 3.0
    p1 = v_add(p0, v_scale(tHat1, alpha_l))
    p2 = v_add(p3, v_scale(tHat2, alpha_r))
    return (p0, p1, p2, p3)


def find_max_error(
    pts: List[Point], bezi: Tuple[Point, Point, Point, Point], u: List[float]
) -> Tuple[int, float]:
    p0, p1, p2, p3 = bezi
    max_err = -1.0
    split_idx = len(pts) // 2
    for i in range(1, len(pts) - 1):
        q = bezier_eval(p0, p1, p2, p3, u[i])
        d2 = (q[0] - pts[i][0]) ** 2 + (q[1] - pts[i][1]) ** 2
        if d2 > max_err:
            max_err = d2
            split_idx = i
    return split_idx, (0.0 if max_err < 0.0 else max_err)


def fit_cubic_recursive(
    pts: List[Point], tHat1: Point, tHat2: Point, max_err2: float, max_iter: int = 12
) -> List[Tuple[Point, Point, Point, Point]]:
    if len(pts) == 2:
        p0, p3 = pts[0], pts[1]
        seg = v_sub(p3, p0)
        seglen = v_norm(seg)
        p1 = v_add(
            p0, v_scale(v_unit(seg if v_norm(tHat1) < EPS else tHat1), seglen / 3.0)
        )
        p2 = v_add(
            p3, v_scale(v_unit(tHat2 if v_norm(tHat2) >= EPS else seg), seglen / 3.0)
        )
        return [(p0, p1, p2, p3)]
    u = chord_length_parameterize(pts)
    bezi = bezier_from_endpoints(pts, u, tHat1, tHat2)
    for _ in range(max_iter):
        split_idx, err2 = find_max_error(pts, bezi, u)
        if err2 <= max_err2:
            return [bezi]
        u2 = reparameterize(pts, bezi, u)
        bezi2 = bezier_from_endpoints(pts, u2, tHat1, tHat2)
        s2, err2b = find_max_error(pts, bezi2, u2)
        if err2b < err2:
            u, bezi = u2, bezi2
        else:
            break
    split_idx, _ = find_max_error(pts, bezi, u)
    split_idx = max(1, min(len(pts) - 2, split_idx))
    t_center = compute_center_tangent(pts, split_idx)
    left = fit_cubic_recursive(
        pts[: split_idx + 1], tHat1, t_center, max_err2, max_iter
    )
    right = fit_cubic_recursive(
        pts[split_idx:], v_scale(t_center, -1.0), tHat2, max_err2, max_iter
    )
    return left + right


def compute_left_tangent(pts: List[Point], i: int) -> Point:
    return v_unit(v_sub(pts[i + 1], pts[i]))


def compute_right_tangent(pts: List[Point], i: int) -> Point:
    return v_unit(v_sub(pts[i - 1], pts[i]))


def compute_center_tangent(pts: List[Point], i: int) -> Point:
    v1 = v_unit(v_sub(pts[i], pts[i - 1]))
    v2 = v_unit(v_sub(pts[i + 1], pts[i]))
    t = v_add(v1, v2)
    return v1 if v_norm(t) < EPS else v_unit(t)


def fit_cubic(
    pts: List[Point], max_error: float
) -> List[Tuple[Point, Point, Point, Point]]:
    # remove duplicates
    dedup = []
    for p in pts:
        if not dedup or not almost_eq(dedup[-1], p):
            dedup.append(p)
    pts = dedup
    if len(pts) < 2:
        return []
    tHat1 = compute_left_tangent(pts, 0)
    tHat2 = compute_right_tangent(pts, len(pts) - 1)
    if v_norm(tHat1) < EPS and len(pts) > 2:
        tHat1 = v_unit(v_sub(pts[2], pts[0]))
    if v_norm(tHat2) < EPS and len(pts) > 2:
        tHat2 = v_unit(v_sub(pts[-3], pts[-1]))
    return fit_cubic_recursive(pts, tHat1, tHat2, max_error * max_error)


# ---------------------- Adaptive cubic flattening --------------------------


def dist_point_to_line(p: Point, a: Point, b: Point) -> float:
    ab = v_sub(b, a)
    denom = v_norm(ab)
    if denom < EPS:
        return v_norm(v_sub(p, a))
    t = max(0.0, min(1.0, v_dot(v_sub(p, a), ab) / (denom * denom)))
    proj = v_add(a, v_scale(ab, t))
    return v_norm(v_sub(p, proj))


def flatten_cubic_adaptive(
    p0: Point,
    c1: Point,
    c2: Point,
    p3: Point,
    flatness: float,
    out: Optional[List[Point]] = None,
) -> List[Point]:
    # returns points AFTER p0, including p3; recursive subdivision at t=0.5
    if out is None:
        out = []
    d1 = dist_point_to_line(c1, p0, p3)
    d2 = dist_point_to_line(c2, p0, p3)
    if max(d1, d2) <= flatness:
        out.append(p3)
        return out
    # de Casteljau split
    p01 = v_scale(v_add(p0, c1), 0.5)
    p12 = v_scale(v_add(c1, c2), 0.5)
    p23 = v_scale(v_add(c2, p3), 0.5)
    p012 = v_scale(v_add(p01, p12), 0.5)
    p123 = v_scale(v_add(p12, p23), 0.5)
    p0123 = v_scale(v_add(p012, p123), 0.5)
    flatten_cubic_adaptive(p0, p01, p012, p0123, flatness, out)
    flatten_cubic_adaptive(p0123, p123, p23, p3, flatness, out)
    return out


# ----------------------------- SFD parsing ---------------------------------


@dataclass
class Segment:
    op: str  # 'm','l','c'
    pt: Optional[Point] = None  # endpoint for m/l/c
    flags: Optional[int] = None
    c1: Optional[Point] = None  # for 'c'
    c2: Optional[Point] = None  # for 'c'
    extras: List[str] = field(default_factory=list)


@dataclass
class Contour:
    raw_lines: List[str] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    has_spiro: bool = False
    parse_ok: bool = True


def is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False


def parse_splineset_block(lines: List[str]) -> List[Contour]:
    contours: List[Contour] = []
    current: Optional[Contour] = None
    in_spiro = False

    def start_new_contour(line: str):
        nonlocal current
        if current is not None:
            contours.append(current)
        current = Contour(raw_lines=[line])

    for line in lines:
        s = line.strip()
        if s.startswith("Spiro"):
            in_spiro = True
            current = current or Contour()
            current.has_spiro = True
            current.raw_lines.append(line)
            continue
        if s.startswith("EndSpiro"):
            in_spiro = False
            current = current or Contour()
            current.raw_lines.append(line)
            continue
        if in_spiro:
            current.raw_lines.append(line)
            continue
        if not s:
            current = current or Contour()
            current.raw_lines.append(line)
            continue

        toks = s.split()
        op_idx = None
        for i, tk in enumerate(toks):
            if tk in ("m", "l", "c"):
                op_idx = i
                break
        if op_idx is None:
            current = current or Contour()
            current.parse_ok = False
            current.raw_lines.append(line)
            continue

        op = toks[op_idx]
        nums = toks[:op_idx]
        after = toks[op_idx + 1 :]

        if op == "m":
            start_new_contour(line)
            if len(nums) != 2 or not after:
                current.parse_ok = False
                continue
            if not (
                is_float_token(nums[0])
                and is_float_token(nums[1])
                and after[0].lstrip("-").isdigit()
            ):
                current.parse_ok = False
                continue
            pt = (float(nums[0]), float(nums[1]))
            flags = int(after[0])
            extras = after[1:]
            current.segments.append(Segment(op="m", pt=pt, flags=flags, extras=extras))
            continue

        if current is None:
            current = Contour()
            current.parse_ok = False
            current.raw_lines.append(line)
            continue

        current.raw_lines.append(line)

        if op == "l":
            if (
                len(nums) != 2
                or not after
                or not (
                    is_float_token(nums[0])
                    and is_float_token(nums[1])
                    and after[0].lstrip("-").isdigit()
                )
            ):
                current.parse_ok = False
                continue
            pt = (float(nums[0]), float(nums[1]))
            flags = int(after[0])
            extras = after[1:]
            current.segments.append(Segment(op="l", pt=pt, flags=flags, extras=extras))
        elif op == "c":
            if (
                len(nums) != 6
                or not after
                or not all(is_float_token(x) for x in nums)
                or not after[0].lstrip("-").isdigit()
            ):
                current.parse_ok = False
                continue
            x1, y1, x2, y2, x3, y3 = map(float, nums)
            flags = int(after[0])
            extras = after[1:]
            current.segments.append(
                Segment(
                    op="c",
                    c1=(x1, y1),
                    c2=(x2, y2),
                    pt=(x3, y3),
                    flags=flags,
                    extras=extras,
                )
            )

    if current is not None:
        contours.append(current)
    return contours


# ----------------------- Corner-aware segmentation -------------------------


@dataclass
class Anchor:
    pt: Point
    # outgoing edge index in `edges` list; incoming is computed by search
    out_edge: Optional[int] = None


@dataclass
class Edge:
    start_idx: int
    end_idx: int
    op: str  # 'l' or 'c'
    c1: Optional[Point] = None
    c2: Optional[Point] = None


def build_topology(contour: Contour) -> Tuple[List[Anchor], List[Edge], bool]:
    if not contour.segments or contour.segments[0].op != "m":
        return [], [], False
    anchors: List[Anchor] = [Anchor(pt=contour.segments[0].pt)]  # m
    edges: List[Edge] = []
    for seg in contour.segments[1:]:
        if seg.op == "l":
            anchors.append(Anchor(pt=seg.pt))
            edges.append(
                Edge(start_idx=len(anchors) - 2, end_idx=len(anchors) - 1, op="l")
            )
        elif seg.op == "c":
            anchors.append(Anchor(pt=seg.pt))
            edges.append(
                Edge(
                    start_idx=len(anchors) - 2,
                    end_idx=len(anchors) - 1,
                    op="c",
                    c1=seg.c1,
                    c2=seg.c2,
                )
            )
        else:
            # ignore unknown ops (already handled by parse_ok)
            pass
    # link outgoing edges
    for idx, e in enumerate(edges):
        anchors[e.start_idx].out_edge = idx
    closed = len(anchors) >= 2 and almost_eq(anchors[0].pt, anchors[-1].pt, tol=1e-7)
    return anchors, edges, closed


def incoming_edge_index(edges: List[Edge], anchor_idx: int) -> Optional[int]:
    for i, e in enumerate(edges):
        if e.end_idx == anchor_idx:
            return i
    return None


def tangent_out_at_anchor(anchors: List[Anchor], edges: List[Edge], i: int) -> Point:
    oe = anchors[i].out_edge
    if oe is None:
        return (0.0, 0.0)
    e = edges[oe]
    if e.op == "c" and e.c1 is not None:
        return v_sub(e.c1, anchors[i].pt)
    # line
    return v_sub(anchors[e.end_idx].pt, anchors[i].pt)


def tangent_in_at_anchor(anchors: List[Anchor], edges: List[Edge], i: int) -> Point:
    ie = incoming_edge_index(edges, i)
    if ie is None:
        return (0.0, 0.0)
    e = edges[ie]
    if e.op == "c" and e.c2 is not None:
        return v_sub(anchors[i].pt, e.c2)
    return v_sub(anchors[i].pt, anchors[e.start_idx].pt)


def detect_corners(
    anchors: List[Anchor], edges: List[Edge], closed: bool, corner_angle_deg: float
) -> Set[int]:
    n = len(anchors)
    if n <= 2:
        return set()
    corners: Set[int] = set()
    for i in range(n):
        if i == 0 and not closed:
            continue
        if i == n - 1 and not closed:
            continue
        vin = tangent_in_at_anchor(anchors, edges, i)
        vout = tangent_out_at_anchor(anchors, edges, i)
        if v_norm(vin) < EPS or v_norm(vout) < EPS:
            continue
        ang = angle_between(vin, vout)
        if ang >= corner_angle_deg:
            corners.add(i)
    return corners


# Build polylines between corner breaks using geometry of the original segments
def polylines_from_breaks(
    anchors: List[Anchor],
    edges: List[Edge],
    closed: bool,
    breaks: List[int],
    flatness: float,
) -> List[Tuple[List[Point], bool]]:
    """
    Returns list of (polyline, end_is_corner) pieces that cover the contour.
    For closed contours, pieces wrap around; for open, pieces are contiguous.
    """
    pieces: List[Tuple[List[Point], bool]] = []
    n = len(anchors)
    if n < 2:
        return pieces

    # Helper to advance one edge index circularly
    def next_anchor_idx(idx: int) -> int:
        # find edge starting at idx
        eidx = anchors[idx].out_edge
        return edges[eidx].end_idx if eidx is not None else idx

    # Order of breaks
    ordered = sorted(set(breaks))
    if not ordered:
        # single piece
        start = 0
        end = n - 1
        pieces.append(
            (polyline_between(anchors, edges, start, end, closed, flatness), False)
        )
        return pieces

    # For closed, ensure wrap-around
    segs = []
    if closed:
        # pick first break as start
        for i in range(len(ordered)):
            a = ordered[i]
            b = ordered[(i + 1) % len(ordered)]
            segs.append((a, b, True))
    else:
        # open: split from 0 -> first, then between, then last -> n-1
        if ordered[0] != 0:
            segs.append((0, ordered[0], True))
        for i in range(len(ordered) - 1):
            segs.append((ordered[i], ordered[i + 1], True))
        if ordered[-1] != n - 1:
            segs.append((ordered[-1], n - 1, True))

    for start, end, end_is_corner in segs:
        poly = polyline_between(anchors, edges, start, end, closed, flatness)
        pieces.append((poly, end_is_corner))
    return pieces


def polyline_between(
    anchors: List[Anchor],
    edges: List[Edge],
    start: int,
    end: int,
    closed: bool,
    flatness: float,
) -> List[Point]:
    """
    Collect a polyline following the original segments from anchor[start] to anchor[end].
    Includes start and end points; follows contour order (wraps if closed).
    """
    pts: List[Point] = [anchors[start].pt]
    if start == end:
        return pts

    n = len(anchors)
    i = start
    while True:
        eidx = anchors[i].out_edge
        if eidx is None:
            break
        e = edges[eidx]
        s_pt = anchors[e.start_idx].pt
        t_pt = anchors[e.end_idx].pt
        if e.op == "l":
            if not almost_eq(t_pt, pts[-1]):
                pts.append(t_pt)
        else:
            seg = flatten_cubic_adaptive(s_pt, e.c1, e.c2, t_pt, flatness)
            for p in seg:
                if not almost_eq(p, pts[-1]):
                    pts.append(p)
        i = e.end_idx
        if i == end:
            break
        if not closed and i == n - 1:
            break
        if closed and i == start:
            # Safety (shouldn't loop indefinitely)
            break
    return pts


# ----------------------------- Emission ------------------------------------


def format_num(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return s


def emit_exact_line_as_cubic(p0: Point, p3: Point) -> Tuple[Point, Point, Point, Point]:
    # exact straight segment cubic: handles at 1/3 and 2/3 along the chord
    v = v_sub(p3, p0)
    p1 = v_add(p0, v_scale(v, 1 / 3))
    p2 = v_add(p0, v_scale(v, 2 / 3))
    return (p0, p1, p2, p3)


def is_near_straight(pts: List[Point], tol: float = 1e-4) -> bool:
    if len(pts) < 3:
        return True
    a, b = pts[0], pts[-1]
    ab = v_sub(b, a)
    L = v_norm(ab)
    if L < EPS:
        return True
    for p in pts[1:-1]:
        if dist_point_to_line(p, a, b) > tol:
            return False
    return True


def emit_fitted_from_pieces(
    pieces: List[Tuple[List[Point], bool]],
    tolerance: float,
    closed: bool,
    corner_indices_flags: Optional[Set[int]] = None,
    anchors: Optional[List[Anchor]] = None,
    breaks_ordered: Optional[List[int]] = None,
) -> List[str]:
    """
    Emit 'm' + 'c' lines from corner-aware pieces.
    Flags: endpoint of each piece is corner (1) if it was a break; smooth (0) otherwise.
    """
    out: List[str] = []
    if not pieces:
        return out
    # Start 'm' at the first piece's first point
    p0 = pieces[0][0][0]
    out.append(f" {format_num(p0[0])} {format_num(p0[1])} m 1")

    for poly, end_is_corner in pieces:
        if len(poly) < 2:
            continue
        if is_near_straight(poly, tol=min(1e-3, tolerance * 0.25)):
            q0, q1, q2, q3 = emit_exact_line_as_cubic(poly[0], poly[-1])
            flag = 1 if end_is_corner else 0
            out.append(
                f" {format_num(q1[0])} {format_num(q1[1])} {format_num(q2[0])} {format_num(q2[1])} {format_num(q3[0])} {format_num(q3[1])} c {flag}"
            )
            continue
        segs = fit_cubic(poly, tolerance)
        if not segs:
            # fallback: exact straight
            q0, q1, q2, q3 = emit_exact_line_as_cubic(poly[0], poly[-1])
            flag = 1 if end_is_corner else 0
            out.append(
                f" {format_num(q1[0])} {format_num(q1[1])} {format_num(q2[0])} {format_num(q2[1])} {format_num(q3[0])} {format_num(q3[1])} c {flag}"
            )
            continue
        # Emit all but the last with smooth flag 0
        for b0, b1, b2, b3 in segs[:-1]:
            out.append(
                f" {format_num(b1[0])} {format_num(b1[1])} {format_num(b2[0])} {format_num(b2[1])} {format_num(b3[0])} {format_num(b3[1])} c 0"
            )
        # Last segment's endpoint flag reflects whether the piece ends at a corner
        b0, b1, b2, b3 = segs[-1]
        flag = 1 if end_is_corner else 0
        out.append(
            f" {format_num(b1[0])} {format_num(b1[1])} {format_num(b2[0])} {format_num(b2[1])} {format_num(b3[0])} {format_num(b3[1])} c {flag}"
        )

    return out


# ------------------------------ Pipeline -----------------------------------


def contour_refit_corner_aware(
    contour: Contour,
    tolerance: float,
    flatness: float,
    corner_angle_deg: float,
    lock_extrema: bool,
) -> Optional[List[str]]:
    anchors, edges, closed = build_topology(contour)
    if not anchors or not edges:
        return None

    # detect geometric corners
    corners = detect_corners(anchors, edges, closed, corner_angle_deg)

    # optionally add axis-extrema as "locks"
    if lock_extrema:
        for i in range(1, len(anchors) - 1):
            prev = anchors[i - 1].pt
            cur = anchors[i].pt
            nxt = anchors[i + 1].pt
            # very rough: local extremum in x or y w.r.t neighbors
            if (
                (cur[0] > prev[0] and cur[0] > nxt[0])
                or (cur[0] < prev[0] and cur[0] < nxt[0])
                or (cur[1] > prev[1] and cur[1] > nxt[1])
                or (cur[1] < prev[1] and cur[1] < nxt[1])
            ):
                corners.add(i)

    # choose breaks (must be indices in [0..len(anchors)-1])
    breaks = sorted(corners)
    # Build piece polylines between breaks
    pieces = polylines_from_breaks(anchors, edges, closed, breaks, flatness)
    if not pieces:
        return None

    # Emit
    return emit_fitted_from_pieces(pieces, tolerance, closed, corners, anchors, breaks)


def contour_total_len(pts: List[Point]) -> float:
    return sum(v_norm(v_sub(b, a)) for a, b in zip(pts, pts[1:]))


def contour_to_points_simple(
    contour: Contour, flatness: float
) -> Tuple[List[Point], bool]:
    # used only for length/degeneracy checks
    anchors, edges, closed = build_topology(contour)
    if not anchors or not edges:
        return ([], False)
    # simple traversal into one polyline
    pts = [anchors[0].pt]
    for e in edges:
        a = anchors[e.start_idx].pt
        b = anchors[e.end_idx].pt
        if e.op == "l":
            if not almost_eq(b, pts[-1]):
                pts.append(b)
        else:
            seg = flatten_cubic_adaptive(a, e.c1, e.c2, b, flatness)
            for p in seg:
                if not almost_eq(p, pts[-1]):
                    pts.append(p)
    return pts, closed


def format_passthrough(raw: List[str]) -> List[str]:
    return [ln if ln.endswith("\n") else ln + "\n" for ln in raw]


def process_sfd(
    in_lines: List[str],
    tolerance: float,
    flatness: float,
    layers_mode: str,
    min_stroke: float,
    corner_angle_deg: float,
    lock_extrema: bool,
    verbose: bool,
) -> List[str]:
    out_lines: List[str] = []
    in_char = False
    in_splineset = False
    current_splineset_lines: List[str] = []
    current_layer_ctx: Optional[str] = None

    def should_process() -> bool:
        return (
            layers_mode == "all"
            and current_layer_ctx not in ("Back",)
            or current_layer_ctx == "Fore"
        )

    i = 0
    n = len(in_lines)
    while i < n:
        line = in_lines[i]
        s = line.strip()

        # glyph scope
        if s.startswith("StartChar:"):
            in_char = True
            out_lines.append(line)
            i += 1
            continue
        if s.startswith("EndChar"):
            if in_splineset:
                out_lines.append(" SplineSet\n")
                out_lines.extend(current_splineset_lines)
                out_lines.append(" EndSplineSet\n")
                in_splineset = False
                current_splineset_lines = []
            in_char = False
            current_layer_ctx = None
            out_lines.append(line)
            i += 1
            continue

        # layer markers
        if s == "Fore":
            current_layer_ctx = "Fore"
            out_lines.append(line)
            i += 1
            continue
        if s == "Back":
            current_layer_ctx = "Back"
            out_lines.append(line)
            i += 1
            continue
        if s.startswith("Layer:"):
            current_layer_ctx = s
            out_lines.append(line)
            i += 1
            continue

        # SplineSet
        if s == "SplineSet":
            in_splineset = True
            current_splineset_lines = []
            i += 1
            while i < n and in_lines[i].strip() != "EndSplineSet":
                current_splineset_lines.append(in_lines[i])
                i += 1

            processed = False
            if should_process():
                contours = parse_splineset_block(
                    [ln.rstrip("\n") for ln in current_splineset_lines]
                )
                new_block: List[str] = []
                any_change = False
                for contour in contours:
                    raw = contour.raw_lines if contour.raw_lines else []
                    if (
                        contour.has_spiro
                        or not contour.parse_ok
                        or not contour.segments
                        or contour.segments[0].op != "m"
                    ):
                        new_block.extend(format_passthrough(raw))
                        continue

                    # Degeneracy & length check using a simple flattened polyline
                    pts_simple, _closed = contour_to_points_simple(contour, flatness)
                    if len(pts_simple) < 2 or contour_total_len(pts_simple) < max(
                        min_stroke, EPS
                    ):
                        new_block.extend(format_passthrough(raw))
                        continue

                    fitted = contour_refit_corner_aware(
                        contour, tolerance, flatness, corner_angle_deg, lock_extrema
                    )
                    if fitted:
                        new_block.append(
                            "".join([])
                        )  # placeholder, no-op to keep structure
                        new_block.extend(
                            [
                                ln + ("\n" if not ln.endswith("\n") else "")
                                for ln in fitted
                            ]
                        )
                        any_change = True
                    else:
                        new_block.extend(format_passthrough(raw))

                if new_block:
                    out_lines.append(" SplineSet\n")
                    out_lines.extend(new_block)
                    out_lines.append(" EndSplineSet\n")
                    processed = True
                    if verbose and any_change:
                        out_lines.append(
                            f" # Refitted (corner-aware) tol={tolerance}, flat={flatness}, angle={corner_angle_deg}\n"
                        )

            if not processed:
                out_lines.append(" SplineSet\n")
                out_lines.extend(current_splineset_lines)
                out_lines.append(" EndSplineSet\n")

            if i < n and in_lines[i].strip() == "EndSplineSet":
                i += 1
            in_splineset = False
            current_splineset_lines = []
            continue

        # default passthrough
        out_lines.append(line)
        i += 1

    return out_lines


# ---------------------------------- CLI ------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Refit SFD SplineSets using Schneider curve fitting (corner-aware)."
    )
    ap.add_argument("input", help="Input .sfd file")
    ap.add_argument(
        "-o", "--output", help="Output .sfd file (default: <input>-fitted.sfd)"
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Max fitting error in font units. Lower preserves shape better. Default: 0.6",
    )
    ap.add_argument(
        "--flatness",
        type=float,
        default=0.25,
        help="Adaptive flattening tolerance for existing cubics (font units). Default: 0.25",
    )
    ap.add_argument(
        "--corner-angle",
        type=float,
        default=35.0,
        help="Minimum tangent angle (degrees) to treat a node as a corner/cusp. Default: 35",
    )
    ap.add_argument(
        "--layers",
        choices=["fore", "all"],
        default="fore",
        help="Process only Fore layer (default) or all SplineSets.",
    )
    ap.add_argument(
        "--min-stroke",
        type=float,
        default=0.0,
        help="Min total contour length to fit; shorter contours are copied. Default: 0",
    )
    ap.add_argument(
        "--lock-extrema",
        action="store_true",
        help="Lock simple axis extrema (adds breaks there).",
    )
    ap.add_argument("--verbose", action="store_true", help="Emit refit comments.")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or os.path.splitext(in_path)[0] + "-fitted.sfd"

    try:
        with open(in_path, "r", encoding="utf-8", errors="replace") as f:
            in_lines = f.readlines()
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)

    out_lines = process_sfd(
        in_lines,
        tolerance=args.tolerance,
        flatness=args.flatness,
        layers_mode=("all" if args.layers == "all" else "fore"),
        min_stroke=args.min_stroke,
        corner_angle_deg=args.corner_angle,
        lock_extrema=args.lock_extrema,
        verbose=args.verbose,
    )

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
