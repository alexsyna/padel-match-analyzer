"""Single-parameter radial (division-model) lens distortion: self-calibrated
correction for wide/fisheye action-cam footage.

The division model (Fitzgibbon 2001, the standard "plumb-line" barrel
correction) defines a raw/distorted pixel point's corrected position through
one scalar `k`, normalized by the image's half-diagonal so `k` stays roughly
resolution-independent:

    corrected = raw / (1 + k * |raw|^2)            (`apply_division`)

This direct form is what corrects a *point* (used by `estimate_k`'s
self-calibration search, applied straight to raw line-sample points). To
correct a whole *image* via `cv2.remap`, the relationship has to run the
other way: for each corrected-space output pixel, `cv2.remap` needs to know
which raw-space pixel to sample from, which is the algebraic inverse of
`apply_division` (`build_maps`, via `_inverse_radius`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from .court import horizontal_line_segments, line_mask


def apply_division(pts: np.ndarray, k: float, w: int, h: int) -> np.ndarray:
    """Direct division-model correction: maps a raw/distorted pixel point to
    its corrected position."""
    pts = np.asarray(pts, dtype=np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r0 = 0.5 * np.hypot(w, h)
    xn = (pts[..., 0] - cx) / r0
    yn = (pts[..., 1] - cy) / r0
    denom = 1.0 + k * (xn * xn + yn * yn)
    xo = xn / denom * r0 + cx
    yo = yn / denom * r0 + cy
    return np.stack([xo, yo], axis=-1)


def _inverse_radius(ri: np.ndarray, k: float) -> np.ndarray:
    """Solve `ri = rc / (1 + k * rc^2)` for `rc >= 0`, vectorized. Closed-form
    (quadratic in `rc`): of the two roots, the non-negative one is the
    physically valid inverse radius."""
    if k == 0.0:
        return ri.copy()
    d = np.maximum(1.0 - 4.0 * k * ri * ri, 0.0)
    sq = np.sqrt(d)
    denom = 2.0 * k * ri
    with np.errstate(divide="ignore", invalid="ignore"):
        r_minus = (1.0 - sq) / denom
        r_plus = (1.0 + sq) / denom
    valid_minus = (r_minus >= 0) & np.isfinite(r_minus)
    out = np.where(valid_minus, r_minus, r_plus)
    return np.where(ri < 1e-12, 0.0, out)


def apply_division_inverse(pts: np.ndarray, k: float, w: int, h: int) -> np.ndarray:
    """Algebraic inverse of `apply_division`: maps a corrected-space point to
    the raw/distorted point that would correct to it."""
    pts = np.asarray(pts, dtype=np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r0 = 0.5 * np.hypot(w, h)
    xn, yn = (pts[..., 0] - cx) / r0, (pts[..., 1] - cy) / r0
    ri = np.hypot(xn, yn)
    rc = _inverse_radius(ri, k)
    scale = np.where(ri < 1e-12, 1.0, rc / np.maximum(ri, 1e-12))
    return np.stack([xn * scale * r0 + cx, yn * scale * r0 + cy], axis=-1)


def build_maps(w: int, h: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    """`cv2.remap`-ready maps that correct a raw/distorted frame: for each
    corrected-space output pixel, where in the raw frame to sample from."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    grid = np.stack([xs, ys], axis=-1)
    sampled_from = apply_division_inverse(grid, k, w, h)
    return sampled_from[..., 0].astype(np.float32), sampled_from[..., 1].astype(np.float32)


@dataclass
class Distortion:
    """Carries a video's estimated lens-distortion coefficient and lazily
    builds/caches the remap maps for its frame size on first use."""

    k: float = 0.0
    _map_x: np.ndarray | None = field(default=None, repr=False, compare=False)
    _map_y: np.ndarray | None = field(default=None, repr=False, compare=False)

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        if self.k == 0.0:
            return frame
        if self._map_x is None:
            h, w = frame.shape[:2]
            self._map_x, self._map_y = build_maps(w, h, self.k)
        return cv2.remap(frame, self._map_x, self._map_y, cv2.INTER_LINEAR)


def simulate_capture_maps(w: int, h: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    """`cv2.remap`-ready maps for synthesizing test fixtures that emulate what
    a lens with distortion `k` would have captured, given an ideal (pinhole)
    rendering: `cv2.remap(ideal, *simulate_capture_maps(w, h, k))` produces an
    image on which `estimate_k`/`Distortion` should recover `k`.

    This is the direct counterpart of `build_maps`'s inverse — for each
    raw-space output pixel, which ideal-space pixel it shows — and is not
    used anywhere in the correction path itself (`Distortion.rectify` only
    ever needs `build_maps`); it exists purely to build synthetic fisheye
    test fixtures (`synth.generate(..., distortion_k=...)`)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    grid = np.stack([xs, ys], axis=-1)
    sampled_from = apply_division(grid, k, w, h)
    return sampled_from[..., 0].astype(np.float32), sampled_from[..., 1].astype(np.float32)


def fit_k_from_point_groups(point_groups: list[np.ndarray], w: int, h: int,
                            bounds: tuple[float, float] = (-0.6, 0.05)) -> tuple[float, float]:
    """Mean perpendicular-pixel residual of each group after applying a
    candidate `k` (direct `apply_division`, correcting the raw points),
    minimized over `k`. Needs at least 2 groups to be meaningfully
    constrained."""

    def residual(k: float) -> float:
        total = 0.0
        for pts in point_groups:
            warped = apply_division(pts, k, w, h)
            vx, vy, x0, y0 = cv2.fitLine(warped.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            d = np.abs((warped[:, 0] - x0) * vy - (warped[:, 1] - y0) * vx)
            total += float(np.mean(d))
        return total / len(point_groups)

    result = minimize_scalar(residual, bounds=bounds, method="bounded")
    return float(result.x), float(result.fun)


def _line_point_groups(frame: np.ndarray) -> list[np.ndarray]:
    """Raw pixel points from the longest near-horizontal line clusters, for
    distortion estimation. Distinct from `court._cluster_lines`, which fits a
    single straight line per cluster — here we need the *raw* points, since
    fitting a line to already-curved data is exactly what we're testing."""
    h, w = frame.shape[:2]
    segs = horizontal_line_segments(line_mask(frame))
    if len(segs) < 2:
        return []
    groups: list[dict] = []
    for s in sorted(segs, key=lambda s: (s[1] + s[3]) / 2):
        key = (s[1] + s[3]) / 2
        if groups and abs(key - groups[-1]["key"]) <= h * 0.02:
            groups[-1]["segs"].append(s)
            groups[-1]["key"] = float(np.mean([(a[1] + a[3]) / 2 for a in groups[-1]["segs"]]))
        else:
            groups.append({"key": key, "segs": [s]})
    groups = [g for g in groups
             if sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in g["segs"]) > 0.15 * w]
    groups = sorted(groups, key=lambda g: sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in g["segs"]),
                    reverse=True)[:3]
    point_groups = []
    for g in groups:
        pts = []
        for x1, y1, x2, y2 in g["segs"]:
            n = max(2, int(np.hypot(x2 - x1, y2 - y1) / 10))
            pts.extend(zip(np.linspace(x1, x2, n), np.linspace(y1, y2, n)))
        point_groups.append(np.array(pts, dtype=np.float64))
    return point_groups


def estimate_k(frame: np.ndarray, max_residual_px: float) -> float:
    """Self-calibrate a single distortion coefficient from the court's own
    (should-be-straight) painted lines. Returns 0.0 — no correction, today's
    exact pre-fix behavior — when fewer than 2 usable line groups are found
    or the best achievable residual doesn't clear `max_residual_px`."""
    h, w = frame.shape[:2]
    groups = _line_point_groups(frame)
    if len(groups) < 2:
        return 0.0
    k, residual = fit_k_from_point_groups(groups, w, h)
    return k if residual <= max_residual_px else 0.0
