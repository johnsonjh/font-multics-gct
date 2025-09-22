#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sfd_fit_schneider.py — Refit SFD SplineSets with Schneider-style cubic curves.

- Reads a FontForge SFD file.
- For each glyph's Fore-layer SplineSet (default), converts any polyline strokes
  (and sampled existing cubic segments) into a sequence of cubic Beziers using
  the Schneider/Graphics Gems algorithm (adaptive subdivision + least squares).
- Degenerate or unparsable contours are copied back verbatim.
- If a SplineSet contour contains a 'Spiro' block, that contour is copied as-is.
- Existing cubic segments are re-calculated by sampling to points, then refitting.

Usage:
  python3 sfd_fit_schneider.py input.sfd [-o output.sfd]
                                         [--tolerance 1.5]
                                         [--resample 12]
                                         [--layers fore|all]
                                         [--min-stroke 0.0]
                                         [--verbose]

Notes:
  * SFD SplineSet syntax is "postscript-like":
      x y m flags
      x y l flags
      x1 y1 x2 y2 x3 y3 c flags
    (There may be optional hint masks after flags; if a contour can't be
     mapped 1:1 due to such extras we copy that contour unchanged.)
  * Newly generated cubic endpoints are flagged as "curve" (0) and 'm' anchors
    as "corner" (1). See FontForge SFD docs for flag bits.
  * Schneider algorithm refs:
      - P. J. Schneider, "An Algorithm for Automatically Fitting Digitized
        Curves", in Graphics Gems (1990).
      - FitCurves.c reference implementation (Graphics Gems).
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ----------------------------- Geometry Utils -----------------------------

Point = Tuple[float, float]

EPS = 1e-9


def v_add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def v_scale(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def v_dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def v_norm2(a: Point) -> float:
    return v_dot(a, a)


def v_norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def v_unit(a: Point) -> Point:
    n = v_norm(a)
    if n < EPS:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


def almost_eq(a: Point, b: Point, tol: float = 1e-7) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# Evaluate a cubic Bezier at t (0..1)
def bezier_eval(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3 * u * u * t
    b2 = 3 * u * t * t
    b3 = t * t * t
    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    return (x, y)


def bezier_eval_dt(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    # First derivative
    u = 1.0 - t
    b0 = -3 * u * u
    b1 = 3 * u * u - 6 * u * t
    b2 = 6 * u * t - 3 * t * t
    b3 = 3 * t * t
    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    return (x, y)


def bezier_eval_ddt(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    # Second derivative
    u = 1.0 - t
    b0 = 6 * u
    b1 = -12 * u + 6 * t
    b2 = 6 * u - 12 * t
    b3 = 6 * t
    x = (
        b0 * (p1[0] - p0[0])
        + b1 * (p2[0] - p1[0])
        + b2 * (p3[0] - p2[0])
        + b3 * (p3[0] - p2[0])
        - b3 * (p2[0] - p1[0])
    )  # Correcting derivation inline
    # A simpler consistent form:
    x = 6 * ((1 - t) * (p2[0] - 2 * p1[0] + p0[0]) + t * (p3[0] - 2 * p2[0] + p1[0]))
    y = 6 * ((1 - t) * (p2[1] - 2 * p1[1] + p0[1]) + t * (p3[1] - 2 * p2[1] + p1[1]))
    return (x, y)


# ---------------------- Schneider Fit (Graphics Gems) ----------------------


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
        numerator = v_dot(diff, q1)
        denominator = v_dot(q1, q1) + v_dot(diff, q2)
        if abs(denominator) < 1e-12:
            return u_k
        return u_k - numerator / denominator

    return [
        (
            0.0
            if i == 0
            else (1.0 if i == len(u) - 1 else max(0.0, min(1.0, newton(u[i], pts[i]))))
        )
        for i in range(len(u))
    ]


def bezier_from_endpoints(
    pts: List[Point], u: List[float], tHat1: Point, tHat2: Point
) -> Tuple[Point, Point, Point, Point]:
    # Solve for alpha values using least-squares (as in Graphics Gems FitCurves.c)
    p0 = pts[0]
    p3 = pts[-1]
    n = len(pts)

    # Basis functions for interior points
    A = []
    for i in range(n):
        ui = u[i]
        b0 = (1 - ui) ** 3
        b1 = 3 * ((1 - ui) ** 2) * ui
        b2 = 3 * (1 - ui) * (ui**2)
        b3 = ui**3
        A.append((v_scale(tHat1, b1), v_scale(tHat2, b2)))

    # Build C and X matrices
    C00 = C01 = C11 = 0.0
    X0 = X1 = 0.0
    for i in range(n):
        ai0, ai1 = A[i]
        pi = pts[i]
        # tmp = P_i - [b0*P0 + b3*P3]
        ui = u[i]
        b0 = (1 - ui) ** 3
        b1 = 3 * ((1 - ui) ** 2) * ui
        b2 = 3 * (1 - ui) * (ui**2)
        b3 = ui**3
        tmp = v_sub(pi, v_add(v_scale(p0, b0), v_scale(p3, b3)))
        C00 += v_dot(ai0, ai0)
        C01 += v_dot(ai0, ai1)
        C11 += v_dot(ai1, ai1)
        X0 += v_dot(ai0, tmp)
        X1 += v_dot(ai1, tmp)

    # Solve [C00 C01; C01 C11] * [alpha_l; alpha_r] = [X0; X1]
    det = C00 * C11 - C01 * C01
    alpha_l = alpha_r = 0.0
    if abs(det) > 1e-12:
        alpha_l = (X0 * C11 - X1 * C01) / det
        alpha_r = (C00 * X1 - C01 * X0) / det
    else:
        # Fallback heuristic: use distance between endpoints
        seglen = v_norm(v_sub(p3, p0))
        alpha_l = alpha_r = seglen / 3.0

    # Clamp alphas to avoid wild control points
    seglen = v_norm(v_sub(p3, p0))
    max_alpha = 1e3 * (seglen if seglen > 0 else 1.0)
    if not math.isfinite(alpha_l) or abs(alpha_l) > max_alpha:
        alpha_l = seglen / 3.0
    if not math.isfinite(alpha_r) or abs(alpha_r) > max_alpha:
        alpha_r = seglen / 3.0

    # NOTE: Right tangent points from Pn to Pn-1 (backwards). Control at end is P3 + alpha_r * tHat2.
    p1 = v_add(p0, v_scale(tHat1, alpha_l))
    p2 = v_add(p3, v_scale(tHat2, alpha_r))
    return (p0, p1, p2, p3)


def compute_left_tangent(pts: List[Point], idx: int) -> Point:
    return v_unit(v_sub(pts[idx + 1], pts[idx]))


def compute_right_tangent(pts: List[Point], idx: int) -> Point:
    # Backwards from end point
    return v_unit(v_sub(pts[idx - 1], pts[idx]))


def compute_center_tangent(pts: List[Point], idx: int) -> Point:
    v1 = v_unit(v_sub(pts[idx], pts[idx - 1]))
    v2 = v_unit(v_sub(pts[idx + 1], pts[idx]))
    t = v_add(v1, v2)
    if v_norm(t) < EPS:
        # If opposing, pick a normal-ish average (fall back to v1)
        return v1
    return v_unit(t)


def find_max_error(
    pts: List[Point], bezi: Tuple[Point, Point, Point, Point], u: List[float]
) -> Tuple[int, float]:
    # Returns (index, squared_error)
    p0, p1, p2, p3 = bezi
    max_err = -1.0
    split_idx = len(pts) // 2
    for i in range(1, len(pts) - 1):
        qi = bezier_eval(p0, p1, p2, p3, u[i])
        d2 = v_norm2(v_sub(qi, pts[i]))
        if d2 > max_err:
            max_err = d2
            split_idx = i
    if max_err < 0.0:
        max_err = 0.0
    return split_idx, max_err


def fit_cubic_recursive(
    pts: List[Point], tHat1: Point, tHat2: Point, max_err2: float, max_iter: int = 10
) -> List[Tuple[Point, Point, Point, Point]]:
    n = len(pts)
    if n == 2:
        # Straight segment
        p0, p3 = pts[0], pts[1]
        seglen = v_norm(v_sub(p3, p0))
        p1 = v_add(p0, v_scale(tHat1, seglen / 3.0))
        p2 = v_add(p3, v_scale(tHat2, seglen / 3.0))
        return [(p0, p1, p2, p3)]

    # Initial parameterization
    u = chord_length_parameterize(pts)
    # Initial fit
    bezi = bezier_from_endpoints(pts, u, tHat1, tHat2)

    # Iteratively try to improve the fit
    for _ in range(max_iter):
        split_idx, err2 = find_max_error(pts, bezi, u)
        if err2 <= max_err2:
            return [bezi]
        # Reparameterize (Newton) and try again
        u_prime = reparameterize(pts, bezi, u)
        bezi_prime = bezier_from_endpoints(pts, u_prime, tHat1, tHat2)
        split_idx2, err2_prime = find_max_error(pts, bezi_prime, u_prime)
        if err2_prime < err2:
            u = u_prime
            bezi = bezi_prime
        else:
            break

    # If we get here, we need to split at the point of max error
    split_idx, _ = find_max_error(pts, bezi, u)
    # To avoid pathological splits
    split_idx = max(1, min(len(pts) - 2, split_idx))
    t_center = compute_center_tangent(pts, split_idx)
    left = fit_cubic_recursive(
        pts[: split_idx + 1], tHat1, t_center, max_err2, max_iter
    )
    right = fit_cubic_recursive(
        pts[split_idx:], v_scale(t_center, -1.0), tHat2, max_err2, max_iter
    )
    return left + right


def fit_cubic(
    pts: List[Point], max_error: float
) -> List[Tuple[Point, Point, Point, Point]]:
    # Filter duplicates
    dedup = []
    for p in pts:
        if not dedup or not almost_eq(dedup[-1], p):
            dedup.append(p)
    pts = dedup
    if len(pts) < 2:
        return []

    # End tangents
    tHat1 = compute_left_tangent(pts, 0)
    tHat2 = compute_right_tangent(pts, len(pts) - 1)

    # If either tangent is zero, attempt to infer from adjacent segment(s)
    if v_norm(tHat1) < EPS and len(pts) > 2:
        tHat1 = v_unit(v_sub(pts[2], pts[0]))
    if v_norm(tHat2) < EPS and len(pts) > 2:
        tHat2 = v_unit(v_sub(pts[-3], pts[-1]))

    return fit_cubic_recursive(pts, tHat1, tHat2, max_error * max_error)


# ----------------------------- SFD Parsing ---------------------------------


@dataclass
class Segment:
    op: str  # 'm', 'l', 'c'
    # For 'm' and 'l'
    pt: Optional[Point] = None
    flags: Optional[int] = None
    # For 'c'
    c1: Optional[Point] = None
    c2: Optional[Point] = None
    # Trailing tokens after flags (e.g., hint masks). Preserved only if contour copied verbatim.
    extras: List[str] = field(default_factory=list)


@dataclass
class Contour:
    raw_lines: List[str] = field(
        default_factory=list
    )  # original text lines for this contour (including Spiro block, etc.)
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
    """
    Parse a SplineSet block into contours. Each contour begins with an 'm' line.
    If a Spiro block is encountered for a contour, mark it and leave raw_lines filled.
    """
    contours: List[Contour] = []
    current: Optional[Contour] = None
    in_spiro = False

    def start_new_contour(line: str):
        nonlocal current
        if current is not None:
            contours.append(current)
        current = Contour(raw_lines=[line])

    for line in lines:
        stripped = line.strip()
        # Spiro passthrough
        if stripped.startswith("Spiro"):
            in_spiro = True
            if current is None:
                current = Contour()
            current.has_spiro = True
            current.raw_lines.append(line)
            continue
        if stripped.startswith("EndSpiro"):
            in_spiro = False
            if current is None:
                current = Contour()
            current.raw_lines.append(line)
            continue
        if in_spiro:
            if current is None:
                current = Contour()
            current.raw_lines.append(line)
            continue

        # Empty or comment/pass-through
        if not stripped:
            if current is None:
                current = Contour()
            current.raw_lines.append(line)
            continue

        # Tokenize
        toks = stripped.split()
        # Identify op token position ('m','l','c')
        op_idx = None
        for i, tk in enumerate(toks):
            if tk in ("m", "l", "c"):
                op_idx = i
                break
        if op_idx is None:
            # Unknown line inside SplineSet -> copy back inside this contour
            if current is None:
                current = Contour()
            current.parse_ok = False
            current.raw_lines.append(line)
            continue

        op = toks[op_idx]
        nums = toks[:op_idx]
        after = toks[op_idx + 1 :]

        # Start new contour when we see 'm'
        if op == "m":
            start_new_contour(line)
            # Expect 2 coords then flags
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

        # For 'l' and 'c', we need an existing contour
        if current is None:
            # Orphan command; treat as parse failure but keep
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
            if len(nums) != 6 or not after:
                current.parse_ok = False
                continue
            if (
                not all(is_float_token(x) for x in nums)
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


# ----------------------------- Conversion ----------------------------------


def sample_cubic(
    prev: Point, c1: Point, c2: Point, p: Point, samples: int
) -> List[Point]:
    pts = []
    # Skip t=0 (prev), include t in (1/s..1), include t=1 end
    for i in range(1, samples + 1):
        t = i / float(samples)
        pts.append(bezier_eval(prev, c1, c2, p, t))
    return pts


def contour_to_points(contour: Contour, resample: int) -> Tuple[List[Point], bool]:
    """
    Build a polyline from a parsed contour by collecting:
      - The 'm' point
      - Endpoints of 'l' segments
      - Sampled points of 'c' segments
    Returns (points, closed_flag).
    """
    if not contour.segments or contour.segments[0].op != "m":
        return [], False
    pts: List[Point] = []
    closed = False
    curr = contour.segments[0].pt
    if curr is None:
        return [], False
    pts.append(curr)
    for seg in contour.segments[1:]:
        if seg.op == "l":
            if seg.pt is None:
                continue
            pts.append(seg.pt)
            curr = seg.pt
        elif seg.op == "c":
            # Sample the cubic from current anchor
            if None in (seg.c1, seg.c2, seg.pt):
                continue
            samples = max(3, resample)
            sampled = sample_cubic(curr, seg.c1, seg.c2, seg.pt, samples)
            pts.extend(sampled)
            curr = seg.pt
        else:
            # unexpected op
            pass

    if pts and almost_eq(pts[0], pts[-1], tol=1e-7):
        closed = True

    # Deduplicate consecutive equal points
    cleaned = []
    for p in pts:
        if not cleaned or not almost_eq(cleaned[-1], p):
            cleaned.append(p)
    return cleaned, closed


def format_num(x: float) -> str:
    # Keep a compact but precise decimal representation
    s = f"{x:.6f}"
    # Trim trailing zeros and dot
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    if s == "":
        s = "0"
    return s


def emit_fitted_contour(pts: List[Point], closed: bool, tolerance: float) -> List[str]:
    """
    Fit the given polyline to cubic Beziers and emit SFD SplineSet lines
    for this single contour: an 'm' followed by 'c' segments.
    """
    out: List[str] = []
    if len(pts) < 2:
        return out

    # If closed, ensure final point equals first. If not, append the start.
    if closed and not almost_eq(pts[0], pts[-1]):
        pts = pts + [pts[0]]

    segments = fit_cubic(pts, tolerance)
    if not segments:
        return out

    # Start: 'm' at first segment's p0
    p0 = segments[0][0]
    out.append(f" {format_num(p0[0])} {format_num(p0[1])} m 1")

    last_end = p0
    for q0, c1, c2, q3 in segments:
        # Sanity: q0 should equal last_end (or very close)
        last_end = q3
        out.append(
            f" {format_num(c1[0])} {format_num(c1[1])} {format_num(c2[0])} {format_num(c2[1])} {format_num(q3[0])} {format_num(q3[1])} c 0"
        )

    return out


# ----------------------------- Main Pipeline -------------------------------


def process_sfd(
    in_lines: List[str],
    tolerance: float,
    resample: int,
    layers_mode: str,
    min_stroke: float,
    verbose: bool,
) -> List[str]:
    """
    Stream-transform the SFD:
      - Copy everything by default
      - When inside a glyph's Fore (or all) SplineSet, refit each contour
        unless it has spiro or can't be parsed, or is degenerate.
    """
    out_lines: List[str] = []

    in_char = False
    in_splineset = False
    current_splineset_lines: List[str] = []
    current_layer_ctx: Optional[str] = None  # 'Fore', 'Back', 'Layer:<n>' etc.

    def should_process_this_splineset() -> bool:
        if layers_mode == "all":
            # Don't process Back images or bitmap-only blocks; we only see SplineSet after Fore/Back/Layer
            return current_layer_ctx not in ("Back",)
        else:
            return current_layer_ctx == "Fore"

    i = 0
    n = len(in_lines)
    while i < n:
        line = in_lines[i]
        stripped = line.strip()

        # Glyph delimiters
        if stripped.startswith("StartChar:"):
            in_char = True
            out_lines.append(line)
            i += 1
            continue
        if stripped.startswith("EndChar"):
            # Flush any pending SplineSet (shouldn't be pending)
            if in_splineset:
                # Safety: copy as-is
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

        # Layer context markers
        if stripped == "Fore":
            current_layer_ctx = "Fore"
            out_lines.append(line)
            i += 1
            continue
        if stripped == "Back":
            current_layer_ctx = "Back"
            out_lines.append(line)
            i += 1
            continue
        if stripped.startswith("Layer:"):
            current_layer_ctx = (
                line.strip()
            )  # keep full text; we'll process SplineSet that follows if layers_mode=all
            out_lines.append(line)
            i += 1
            continue

        # SplineSet handling
        if stripped == "SplineSet":
            in_splineset = True
            current_splineset_lines = []
            i += 1
            # Accumulate until EndSplineSet
            while i < n:
                l2 = in_lines[i]
                if l2.strip() == "EndSplineSet":
                    break
                current_splineset_lines.append(l2)
                i += 1

            # Now i points at EndSplineSet or EOF
            # Decide whether to process this block
            processed = False
            if should_process_this_splineset():
                # Parse contours
                contours = parse_splineset_block(
                    [ln.rstrip("\n") for ln in current_splineset_lines]
                )

                # Build new SplineSet content
                new_block: List[str] = []
                any_change = False
                cursor = 0  # we keep order; for each contour we either emit refit or original
                # To map raw lines per contour, we reconstruct from each contour.raw_lines
                for contour in contours:
                    raw = contour.raw_lines if contour.raw_lines else []
                    # If contour has spiro or failed to parse, or missing 'm', copy raw
                    if (
                        contour.has_spiro
                        or not contour.parse_ok
                        or not contour.segments
                        or contour.segments[0].op != "m"
                    ):
                        new_block.extend(
                            [ln if ln.endswith("\n") else ln + "\n" for ln in raw]
                        )
                        continue

                    # Build polyline from contour
                    pts, closed = contour_to_points(contour, resample=resample)
                    # Degenerate checks: too few points or too short
                    if len(pts) < 2:
                        new_block.extend(
                            [ln if ln.endswith("\n") else ln + "\n" for ln in raw]
                        )
                        continue
                    total_len = 0.0
                    for a, b in zip(pts, pts[1:]):
                        total_len += v_norm(v_sub(b, a))
                    if total_len < max(min_stroke, EPS):
                        new_block.extend(
                            [ln if ln.endswith("\n") else ln + "\n" for ln in raw]
                        )
                        continue

                    # Fit with Schneider algorithm
                    fitted_lines = emit_fitted_contour(pts, closed, tolerance)
                    if fitted_lines:
                        new_block.extend(
                            [
                                ln + ("\n" if not ln.endswith("\n") else "")
                                for ln in fitted_lines
                            ]
                        )
                        any_change = True
                    else:
                        # Fallback: copy original
                        new_block.extend(
                            [ln if ln.endswith("\n") else ln + "\n" for ln in raw]
                        )

                # Only if we produced something; otherwise copy original
                if new_block:
                    out_lines.append(" SplineSet\n")
                    out_lines.extend(new_block)
                    out_lines.append(" EndSplineSet\n")
                    processed = True
                    if verbose and any_change:
                        out_lines.append(
                            f" # Refitted SplineSet with tolerance={tolerance}\n"
                        )
            if not processed:
                out_lines.append(" SplineSet\n")
                out_lines.extend(current_splineset_lines)
                out_lines.append(" EndSplineSet\n")

            # Consume the "EndSplineSet" line
            if i < n and in_lines[i].strip() == "EndSplineSet":
                i += 1
            in_splineset = False
            current_splineset_lines = []
            continue

        # Default: copy through
        out_lines.append(line)
        i += 1

    return out_lines


# --------------------------------- CLI -------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Refit SFD SplineSets using Schneider-style cubic curve fitting."
    )
    ap.add_argument("input", help="Input .sfd file")
    ap.add_argument(
        "-o", "--output", help="Output .sfd file (default: <input>-fitted.sfd)"
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1.5,
        help="Max fitting error (font units). Default: 1.5",
    )
    ap.add_argument(
        "--resample",
        type=int,
        default=12,
        help="Samples per existing cubic segment when re-calculating. Default: 12",
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
        help="Minimum total contour length to fit; shorter contours are copied. Default: 0",
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
        resample=args.resample,
        layers_mode=("all" if args.layers == "all" else "fore"),
        min_stroke=args.min_stroke,
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
