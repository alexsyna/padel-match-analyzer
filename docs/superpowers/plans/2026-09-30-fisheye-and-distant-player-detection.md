# Fisheye correction + distant-player detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover accurate court detection and player tracking on fisheye/wide-lens footage, and stop missing genuinely-in-frame distant players, while leaving every existing (non-fisheye) video's behavior byte-for-byte unchanged.

**Architecture:** A new `distortion.py` module implements a single-parameter radial ("division model") lens-distortion estimate-and-correct step, self-calibrated from the court's own painted lines, that slots into `pipeline.resolve_court`/`track_players` and `webapp.probe_run` as a frame-preprocessing stage ahead of the existing homography-based court/tracking code (which is otherwise untouched). Separately, `detection.py` gains a court-aware two-pass (full-frame + upscaled far-court crop) detection path for YOLO and a depth-aware area/height threshold for the background-subtraction backend, both driven by the (now-correct) `CourtModel`.

**Tech Stack:** Python 3.10+, OpenCV (`cv2`), NumPy, SciPy (`scipy.optimize.minimize_scalar`, already a dependency), pytest. No new dependencies.

## Global Constraints

- Every existing test in `tests/test_units.py` and `tests/test_end_to_end.py` must keep passing unchanged — `k = 0.0` (no distortion detected) and `court=None`/`far_bbox=None` (no perspective info) must be true no-ops, reproducing today's exact code paths.
- No new third-party dependency. `scipy.optimize` is already available (`scipy>=1.10` in `pyproject.toml`).
- Config additions are additive only (new fields with defaults); no existing `PipelineConfig`/`DetectionConfig`/`CourtConfig` field is removed or repurposed.
- Follow the existing code style in this repo: plain dataclasses, no docstring bloat, comments only where the *why* isn't obvious from the code (see `CLAUDE.md`-equivalent conventions already visible throughout `court.py`/`detection.py`).
- Part 1B from the design spec (manual multi-point calibration UI, reached only when auto self-calibration fails) is explicitly **out of scope** for this plan — see "Deferred" at the end.

---

### Task 1: Extract `horizontal_line_segments` in `court.py` for reuse

**Files:**
- Modify: `src/padelfusion/court.py:206-226` (inside `detect_court`)
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `horizontal_line_segments(mask: np.ndarray) -> list[tuple[float, float, float, float]]` — near-horizontal Hough segments `(x1, y1, x2, y2)` from a line mask. Used by `detect_court` (this task) and by `distortion.estimate_k` (Task 4).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_units.py` (near the other court tests):

```python
from padelfusion.court import CourtModel, detect_court, horizontal_line_segments, parse_corners, zone


def test_horizontal_line_segments_finds_service_lines():
    spec = SynthSpec(width=1280, height=720)
    truth = CourtModel(image_corners(spec.width, spec.height))
    frame = render_background(spec, truth)
    from padelfusion.court import line_mask
    segs = horizontal_line_segments(line_mask(frame))
    assert len(segs) >= 2
    for x1, y1, x2, y2 in segs:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        assert ang < 10 or ang > 170
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_horizontal_line_segments_finds_service_lines -v`
Expected: FAIL with `ImportError: cannot import name 'horizontal_line_segments'`

- [ ] **Step 3: Extract the function**

In `src/padelfusion/court.py`, replace the Hough-extraction block inside `detect_court` (currently lines ~213-224):

```python
def detect_court(frame: np.ndarray, min_score: float = 0.55) -> CourtModel | None:
    """Find the court from its painted service lines.

    Returns None when no candidate reaches `min_score`. The caller then falls
    back to manual corners.
    """
    h, w = frame.shape[:2]
    mask = line_mask(frame)
    fat = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    horiz = horizontal_line_segments(mask)
    hlines = _cluster_lines(horiz, lambda s: (s[1] + s[3]) / 2, h * 0.012)
```

and add the new function just above `detect_court`:

```python
def horizontal_line_segments(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Near-horizontal Hough segments from a line mask, as (x1, y1, x2, y2)."""
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60,
                           minLineLength=min(h, w) // 10, maxLineGap=min(h, w) // 40)
    if segs is None:
        return []
    out = []
    for s in np.asarray(segs).reshape(-1, 4):
        s = tuple(map(float, s))
        ang = np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180
        if ang < 10 or ang > 170:
            out.append(s)
    return out
```

Remove the now-duplicated `segs = cv2.HoughLinesP(...)` / `if segs is None: return None` / `horiz = []` loop that used to live inline in `detect_court` (replaced by the two lines above). The rest of `detect_court` (from `hlines = sorted(...)` onward) is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_horizontal_line_segments_finds_service_lines tests/test_units.py::test_auto_court_detection_on_rendered_court tests/test_units.py::test_auto_court_detection_fails_cleanly_on_blank_frame -v`
Expected: all 3 PASS (the latter two confirm the refactor didn't change `detect_court`'s behavior)

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/court.py tests/test_units.py
git commit -m "refactor: extract horizontal_line_segments from detect_court for reuse"
```

---

### Task 2: Add `CourtConfig.distortion_max_residual_px`

**Files:**
- Modify: `src/padelfusion/config.py:22-33`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `CourtConfig.distortion_max_residual_px: float` (new field, default `1.5`)

- [ ] **Step 1: Write the failing test**

```python
from padelfusion.config import CourtConfig


def test_court_config_has_distortion_threshold():
    cfg = CourtConfig()
    assert cfg.distortion_max_residual_px == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_court_config_has_distortion_threshold -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add the field**

In `src/padelfusion/config.py`:

```python
@dataclass
class CourtConfig:
    net_zone_depth_m: float = 5.0
    out_of_court_margin_m: float = 0.75
    auto_min_score: float = 0.55
    # Mean perpendicular-pixel residual, after applying a candidate distortion
    # correction to the court's own line samples, below which auto
    # self-calibration (distortion.py) is accepted. Above this, k = 0.0 (no
    # correction) — the exact pre-fisheye-fix behavior.
    distortion_max_residual_px: float = 1.5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_court_config_has_distortion_threshold -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/config.py tests/test_units.py
git commit -m "feat: add CourtConfig.distortion_max_residual_px"
```

---

### Task 3: `distortion.py` — the division model and `Distortion`

**Files:**
- Create: `src/padelfusion/distortion.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces:
  - `apply_division(pts: np.ndarray, k: float, w: int, h: int) -> np.ndarray` — maps `(..., 2)` pixel points through the single-parameter division model.
  - `build_maps(w: int, h: int, k: float) -> tuple[np.ndarray, np.ndarray]` — `cv2.remap`-ready `(map_x, map_y)`, both `float32`.
  - `class Distortion` with `k: float = 0.0` and `.rectify(frame: np.ndarray) -> np.ndarray` (no-op when `k == 0.0`, else lazily builds and caches maps for that frame's `(h, w)`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_units.py`:

```python
from padelfusion.distortion import Distortion, apply_division, build_maps


def test_apply_division_identity_at_k_zero():
    pts = np.array([[10.0, 20.0], [640.0, 360.0], [1270.0, 710.0]])
    out = apply_division(pts, 0.0, 1280, 720)
    assert np.allclose(out, pts, atol=1e-9)


def test_apply_division_straightens_a_known_forward_distortion():
    """`_forward_distort` (test-only) builds the raw point that `apply_division`
    would map back to a given straight-line point, by inverting the model's
    purely-radial scale factor with a 1-D root search."""
    from scipy.optimize import brentq

    def _forward_distort(pts, k, w, h):
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r0 = 0.5 * np.hypot(w, h)
        out = np.empty_like(pts, dtype=np.float64)
        for i, (x, y) in enumerate(pts):
            xn, yn = (x - cx) / r0, (y - cy) / r0
            ro = np.hypot(xn, yn)
            if ro < 1e-9:
                out[i] = [x, y]
                continue
            theta = np.arctan2(yn, xn)
            rn = brentq(lambda r: r / (1 + k * r * r) - ro, 0, ro * 3 + 1.0)
            out[i] = [cx + r0 * rn * np.cos(theta), cy + r0 * rn * np.sin(theta)]
        return out

    w, h = 1280, 720
    k_true = -0.18
    straight = np.array([[x, 600.0] for x in np.linspace(80, 1200, 40)])
    distorted = _forward_distort(straight, k_true, w, h)

    def residual(pts):
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        return float(np.mean(np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)))

    assert residual(distorted) > 2.0
    corrected = apply_division(distorted, k_true, w, h)
    assert residual(corrected) < 0.5


def test_distortion_rectify_is_noop_at_k_zero():
    frame = np.random.default_rng(0).integers(0, 255, (100, 120, 3), dtype=np.uint8)
    out = Distortion(0.0).rectify(frame)
    assert out is frame


def test_distortion_rectify_caches_maps():
    frame = np.zeros((100, 120, 3), np.uint8)
    d = Distortion(-0.1)
    out1 = d.rectify(frame)
    assert out1.shape == frame.shape
    assert d._map_x is not None
    maps_id = id(d._map_x)
    d.rectify(frame)
    assert id(d._map_x) == maps_id  # reused, not rebuilt
```

Add `import cv2` at the top of `tests/test_units.py` if not already present (it currently isn't — check the existing import block and add it alongside the other imports).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "apply_division or distortion_rectify" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'padelfusion.distortion'`

- [ ] **Step 3: Implement `distortion.py`**

Create `src/padelfusion/distortion.py`:

```python
"""Single-parameter radial (division-model) lens distortion: self-calibrated
correction for wide/fisheye action-cam footage.

The division model (Fitzgibbon 2001, the standard "plumb-line" barrel
correction) maps a pixel point through one scalar `k`, normalized by the
image's half-diagonal so `k` stays roughly resolution-independent:

    r_o = r_i / (1 + k * r_i^2)

This same function is reused two ways in this module: applied to a full
output pixel grid it gives `cv2.remap`-ready correction maps (`build_maps`);
applied directly to a handful of line-sample points it is the objective
`court.py`'s self-calibration (`estimate_k`, in court.py) searches over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def apply_division(pts: np.ndarray, k: float, w: int, h: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r0 = 0.5 * np.hypot(w, h)
    xn = (pts[..., 0] - cx) / r0
    yn = (pts[..., 1] - cy) / r0
    denom = 1.0 + k * (xn * xn + yn * yn)
    xo = xn / denom * r0 + cx
    yo = yn / denom * r0 + cy
    return np.stack([xo, yo], axis=-1)


def build_maps(w: int, h: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    grid = np.stack([xs, ys], axis=-1)
    warped = apply_division(grid, k, w, h)
    return warped[..., 0].astype(np.float32), warped[..., 1].astype(np.float32)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "apply_division or distortion_rectify" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/distortion.py tests/test_units.py
git commit -m "feat: add division-model distortion correction (distortion.py)"
```

---

### Task 4: `estimate_k` — self-calibration from the court's own lines

**Files:**
- Modify: `src/padelfusion/distortion.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Consumes: `court.line_mask(frame)`, `court.horizontal_line_segments(mask)` (Task 1), `apply_division` (Task 3)
- Produces:
  - `fit_k_from_point_groups(point_groups: list[np.ndarray], w: int, h: int, bounds: tuple[float, float] = (-0.6, 0.05)) -> tuple[float, float]` — `(k, residual_px)`.
  - `estimate_k(frame: np.ndarray, max_residual_px: float) -> float` — `0.0` if rejected or too few usable line groups.

- [ ] **Step 1: Write the failing tests**

```python
from padelfusion.distortion import estimate_k, fit_k_from_point_groups


def test_fit_k_from_point_groups_recovers_known_k():
    from scipy.optimize import brentq

    def _forward_distort(pts, k, w, h):
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r0 = 0.5 * np.hypot(w, h)
        out = np.empty_like(pts, dtype=np.float64)
        for i, (x, y) in enumerate(pts):
            xn, yn = (x - cx) / r0, (y - cy) / r0
            ro = np.hypot(xn, yn)
            if ro < 1e-9:
                out[i] = [x, y]
                continue
            theta = np.arctan2(yn, xn)
            rn = brentq(lambda r: r / (1 + k * r * r) - ro, 0, ro * 3 + 1.0)
            out[i] = [cx + r0 * rn * np.cos(theta), cy + r0 * rn * np.sin(theta)]
        return out

    w, h = 1280, 720
    k_true = -0.18
    lines = [np.array([[x, y0] for x in np.linspace(80, 1200, 40)])
             for y0 in (200.0, 500.0, 650.0)]
    groups = [_forward_distort(line, k_true, w, h) for line in lines]
    k_fit, residual = fit_k_from_point_groups(groups, w, h)
    assert abs(k_fit - k_true) < 0.03
    assert residual < 0.5


def test_estimate_k_on_distorted_synthetic_frame():
    spec = SynthSpec(width=1280, height=720)
    court = CourtModel(image_corners(spec.width, spec.height))
    ideal = render_background(spec, court)
    k_true = -0.15
    map_x, map_y = build_maps(spec.width, spec.height, k_true)
    distorted = cv2.remap(ideal, map_x, map_y, cv2.INTER_LINEAR)

    assert detect_court(distorted, CourtConfig().auto_min_score) is None  # fails before correction

    k_fit = estimate_k(distorted, CourtConfig().distortion_max_residual_px)
    assert abs(k_fit - k_true) < 0.03
    corrected = Distortion(k_fit).rectify(distorted)
    assert detect_court(corrected, CourtConfig().auto_min_score) is not None


def test_estimate_k_returns_zero_on_blank_frame():
    assert estimate_k(np.zeros((720, 1280, 3), np.uint8), 1.5) == 0.0
```

These need `from padelfusion.config import CourtConfig` and `from padelfusion.distortion import build_maps` added to the test file's import block (alongside the existing imports).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "fit_k_from_point_groups or estimate_k" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `fit_k_from_point_groups` and `estimate_k`**

Append to `src/padelfusion/distortion.py` (add `from scipy.optimize import minimize_scalar` to the imports, and `from .court import horizontal_line_segments, line_mask`):

```python
from scipy.optimize import minimize_scalar

from .court import horizontal_line_segments, line_mask


def fit_k_from_point_groups(point_groups: list[np.ndarray], w: int, h: int,
                            bounds: tuple[float, float] = (-0.6, 0.05)) -> tuple[float, float]:
    """Mean perpendicular-pixel residual of each group after applying a
    candidate `k`, minimized over `k`. Needs at least 2 groups to be
    meaningfully constrained."""

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "fit_k_from_point_groups or estimate_k" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/distortion.py tests/test_units.py
git commit -m "feat: self-calibrate distortion k from the court's own lines"
```

---

### Task 5: `CourtModel.k` field + serialization

**Files:**
- Modify: `src/padelfusion/court.py:39-70` (`CourtModel` dataclass)
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `CourtModel.k: float = 0.0` (new field); `to_dict()["distortion_k"]`; `from_dict` reads it back (default `0.0` for old timelines without the key).

- [ ] **Step 1: Write the failing test**

```python
def test_court_model_distortion_k_roundtrip():
    m = CourtModel(image_corners(1280, 720), source="auto", score=0.9, k=-0.12)
    d = m.to_dict()
    assert d["distortion_k"] == -0.12
    back = CourtModel.from_dict(d)
    assert back.k == -0.12

    old_format = {"corners_img": image_corners(1280, 720).tolist(), "source": "manual"}
    assert CourtModel.from_dict(old_format).k == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_court_model_distortion_k_roundtrip -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'k'`

- [ ] **Step 3: Add the field**

In `src/padelfusion/court.py`, update `CourtModel`:

```python
@dataclass
class CourtModel:
    corners_img: np.ndarray  # (4, 2) pixel coords, order near-left, near-right, far-right, far-left
    source: str = "manual"  # "auto" | "manual"
    score: float | None = None  # line-support score for auto detection
    k: float = 0.0  # distortion.py's self-calibrated division-model coefficient; 0.0 = no correction

    def __post_init__(self) -> None:
        self.corners_img = np.asarray(self.corners_img, dtype=np.float64).reshape(4, 2)
        self.H = cv2.getPerspectiveTransform(
            self.corners_img.astype(np.float32), COURT_CORNERS.astype(np.float32)
        )
        self.H_inv = np.linalg.inv(self.H)

    def image_to_court(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def court_to_image(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "corners_img": self.corners_img.tolist(),
            "corner_order": ["near_left", "near_right", "far_right", "far_left"],
            "source": self.source,
            "score": self.score,
            "distortion_k": self.k,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CourtModel":
        return cls(np.array(d["corners_img"]), d.get("source", "manual"), d.get("score"),
                   d.get("distortion_k", 0.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_court_model_distortion_k_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/court.py tests/test_units.py
git commit -m "feat: carry the distortion coefficient on CourtModel"
```

---

### Task 6: Wire self-calibration into `pipeline.resolve_court`

**Files:**
- Modify: `src/padelfusion/pipeline.py:62-72`
- Test: `tests/test_units.py`

**Interfaces:**
- Consumes: `distortion.estimate_k`, `distortion.Distortion` (Task 3/4), `CourtModel.k` (Task 5)
- Produces: `resolve_court(frame, cfg, corners=None) -> CourtModel` — unchanged signature/return type, now transparently corrects fisheye footage.

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_court_corrects_fisheye_where_detect_court_fails():
    from padelfusion.config import PipelineConfig
    from padelfusion.distortion import build_maps
    from padelfusion.pipeline import resolve_court

    spec = SynthSpec(width=1280, height=720)
    truth = CourtModel(image_corners(spec.width, spec.height))
    ideal = render_background(spec, truth)
    map_x, map_y = build_maps(spec.width, spec.height, -0.15)
    distorted = cv2.remap(ideal, map_x, map_y, cv2.INTER_LINEAR)

    assert detect_court(distorted, PipelineConfig().court.auto_min_score) is None

    court = resolve_court(distorted, PipelineConfig())
    assert court.source == "auto"
    assert court.k != 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_resolve_court_corrects_fisheye_where_detect_court_fails -v`
Expected: FAIL (`resolve_court` raises `CourtNotFoundError`)

- [ ] **Step 3: Update `resolve_court`**

In `src/padelfusion/pipeline.py`, add the import (`from .distortion import Distortion, estimate_k`) and change:

```python
def resolve_court(frame: np.ndarray, cfg: PipelineConfig, corners=None) -> CourtModel:
    k = estimate_k(frame, cfg.court.distortion_max_residual_px)
    rectified = Distortion(k).rectify(frame) if k else frame
    if corners is not None:
        return CourtModel(np.asarray(corners, dtype=float), source="manual", k=k)
    court = detect_court(rectified, cfg.court.auto_min_score)
    if court is None:
        raise CourtNotFoundError(
            "automatic court-line detection failed. Export the first frame with "
            "`padelfusion frame VIDEO frame.png`, read off the four court corners "
            "(near-left, near-right, far-right, far-left) and pass them with "
            "--corners 'x,y;x,y;x,y;x,y'")
    court.k = k
    return court
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_resolve_court_corrects_fisheye_where_detect_court_fails -v`
Expected: PASS

Also run the full unit suite to confirm no regressions:

Run: `pytest tests/test_units.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/pipeline.py tests/test_units.py
git commit -m "feat: self-calibrate and correct fisheye distortion in resolve_court"
```

---

### Task 7: Rectify frames throughout `track_players`

**Files:**
- Modify: `src/padelfusion/pipeline.py:75-108` (`track_players`)

**Interfaces:**
- Consumes: `court.k` (Task 5), `distortion.Distortion` (Task 3)
- No new public interface — internal consistency fix so every frame the detector/tracker sees is in the same rectified pixel space the resolved `CourtModel`'s homography assumes.

- [ ] **Step 1: Update `track_players`**

In `src/padelfusion/pipeline.py`:

```python
def track_players(video_path, court: CourtModel, cfg: PipelineConfig, me_px=None,
                  play_times: list[float] | None = None, on_frame=None):
    """`play_times` (audio onset times) tell the background detector when players
    are spread out and moving. Frames sampled then give a background without
    players baked into it."""
    tcfg = cfg.tracking
    dist = Distortion(court.k)
    first = vid.read_frame(video_path, 0.0)
    mask = court_region_mask(court, first.shape, cfg.court.out_of_court_margin_m + 0.5)
    times = None
    if play_times and len(play_times) >= 10:
        times = sorted(t + d for t in play_times for d in (-0.5, 0.0, 0.5) if t + d >= 0)
    detector = make_detector(
        cfg.detection,
        lambda: [dist.rectify(f) for f in vid.sample_frames(video_path, cfg.detection.bg_samples, times)],
        court_mask=mask)
    tracker = Tracker(tcfg)
    initialized = False
    n_samples = 0
    for _, t, frame in vid.iter_frames(video_path, tcfg.analysis_fps):
        frame = dist.rectify(frame)
        n_samples += 1
        if on_frame is not None:
            on_frame(t)
        cdets = to_court_detections(detector.detect(frame), court, cfg.court.out_of_court_margin_m)
        if not initialized:
            if t > tcfg.init_search_s:
                raise TrackingInitError(
                    f"could not find two players on each side of the net in the first "
                    f"{tcfg.init_search_s:.0f}s; check the capture framing")
            initialized = tracker.initialize(t, cdets, me_px)
            continue
        tracker.update(t, cdets)
    if not initialized:
        raise TrackingInitError("no frames with four players found")
    points, segments = tracker.finalize()
    return tracker, points, segments, detector.name, n_samples
```

(Only the two additions — `dist = Distortion(court.k)`, and rectifying inside the sample-frames lambda and the frame loop — are new; everything else is unchanged.)

At `k == 0.0` (the overwhelming majority of existing/non-fisheye videos), `Distortion.rectify` is a no-op passthrough (Task 3), so this is behavior-preserving for every video that isn't fisheye-distorted.

- [ ] **Step 2: Run the full unit and end-to-end suites to confirm no regression**

Run: `pytest tests/ -v`
Expected: all PASS (this task has no new dedicated test — Task 10's end-to-end test is what exercises this code path with `k != 0.0`; this step just guards the `k == 0.0` path stays identical)

- [ ] **Step 3: Commit**

```bash
git add src/padelfusion/pipeline.py
git commit -m "feat: rectify analyzed frames through track_players when distorted"
```

---

### Task 8: Add distortion support to `synth.generate`

**Files:**
- Modify: `src/padelfusion/synth.py` (`generate`)
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `distortion.build_maps` (Task 3)
- Produces: `synth.generate(out_dir, spec=None, distortion_k: float = 0.0)` — new optional kwarg, default preserves today's exact output byte-for-byte.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_end_to_end.py`:

```python
def test_generate_accepts_distortion_k(tmp_path):
    paths = generate(tmp_path, SynthSpec(width=1280, height=720, duration_s=3.0, seed=1),
                     distortion_k=-0.2)
    meta = json.loads(paths["meta"].read_text())
    assert meta["distortion_k"] == -0.2
    assert len(meta["me_px"]) == 2
    assert paths["video"].exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end.py::test_generate_accepts_distortion_k -v`
Expected: FAIL with `TypeError: generate() got an unexpected keyword argument 'distortion_k'`

- [ ] **Step 3: Add distortion support to `synth.generate`**

In `src/padelfusion/synth.py`, add the import (`from .distortion import build_maps`) and change the `generate` function:

```python
def generate(out_dir: str | Path, spec: SynthSpec | None = None,
            distortion_k: float = 0.0) -> dict[str, Path]:
    spec = spec or SynthSpec()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    corners = image_corners(spec.width, spec.height)
    court = CourtModel(corners)
    rallies = make_script(spec, rng)
    bg = render_background(spec, court)
    maps = build_maps(spec.width, spec.height, distortion_k) if distortion_k else None

    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "video.mp4"
        wav = Path(tmp) / "audio.wav"
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), spec.fps,
                                 (spec.width, spec.height))
        frame0 = None
        for i in range(int(spec.duration_s * spec.fps)):
            t = i / spec.fps
            frame = bg.copy()
            order = sorted(TEAM, key=lambda p: -player_pos(p, t, rallies)[1])  # far first
            far_drawn = False
            for pid in order:
                if TEAM[pid] == "near" and not far_drawn:
                    draw_net(frame, court)
                    far_drawn = True
                draw_player(frame, court, pid, player_pos(pid, t, rallies), swinging(pid, t, rallies))
            if maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            if i == 0:
                frame0 = frame.copy()
            writer.write(frame)
        writer.release()
        wavfile.write(wav, spec.sample_rate, (render_audio(spec, rallies, rng) * 32767).astype(np.int16))
        video = out / "synthetic_match.mp4"
        subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent),
                        "-i", str(wav), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(video)], check=True)

    labels = out / "labels.json"
    labels.write_text(json.dumps(labels_from_script(rallies), indent=1))
    if maps is not None:
        # Locate P1 in the actual (distorted) first frame by color, rather
        # than forward-distorting the ideal pixel point.
        color = np.array(PLAYER_COLORS["P1"])
        blob = cv2.inRange(frame0, color, color)
        ys, xs = np.nonzero(blob)
        me = [float(xs.mean()), float(ys.min() + 2)]
    else:
        p = court.court_to_image([player_pos("P1", 0.0, rallies)])[0]
        me = [float(p[0]), float(p[1] - 20)]
    meta = out / "synthetic_meta.json"
    meta.write_text(json.dumps({"corners_img": corners.tolist(), "me_px": me,
                                "spec": spec.__dict__, "distortion_k": distortion_k}, indent=1))
    return {"video": video, "labels": labels, "meta": meta}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_end_to_end.py::test_generate_accepts_distortion_k -v`
Expected: PASS

Also run: `pytest tests/test_end_to_end.py -v` to confirm the pre-existing `test_pipeline_end_to_end` (which calls `generate()` without `distortion_k`) is unaffected.
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/synth.py tests/test_end_to_end.py
git commit -m "feat: support forward-distorting the synthetic match for fisheye tests"
```

---

### Task 9: Rectify `frame0.jpg` in the webapp

**Files:**
- Modify: `src/padelfusion/webapp.py:122-128` (`probe_run`)
- Test: `tests/test_webapp.py`

**Interfaces:**
- Consumes: `synth.generate(..., distortion_k=...)` (Task 8)
- Produces: `probe_run` now writes an already-rectified `frame0.jpg`, and the run dict gains a `distortion_k` field. No endpoint signatures change — `/court/detect` and manual-corner clicks continue to operate on whatever `frame0.jpg` contains, which is now pre-corrected.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_webapp.py`, reusing the file's existing `server` fixture and `call` helper exactly as `test_upload_detect_labels_and_files` does, with its own distorted video built via `generate(..., distortion_k=...)`:

```python
def test_upload_rectifies_fisheye_and_still_detects_court(server, tmp_path_factory):
    base, _ = server
    video = generate(tmp_path_factory.mktemp("v_fisheye"),
                     SynthSpec(width=1280, height=720, duration_s=3.0), distortion_k=-0.2)["video"]
    status, body = call(base, "/api/upload?filename=fisheye.mp4", "PUT", raw=video.read_bytes())
    run = json.loads(body)
    assert run["state"] == "ready"
    assert run["distortion_k"] != 0.0

    court = json.loads(call(base, f"/api/runs/{run['id']}/court/detect", "POST", body={})[1])
    assert court["found"]
```

`SynthSpec` and `generate` are already imported at the top of this file (used by the `server` fixture) — no new import needed.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webapp.py::test_upload_rectifies_fisheye_and_still_detects_court -v`
Expected: FAIL with `KeyError: 'distortion_k'`

- [ ] **Step 3: Update `probe_run`**

In `src/padelfusion/webapp.py`, add the import (`from .distortion import Distortion, estimate_k`) and change:

```python
def probe_run(run: dict) -> dict:
    info = vid.probe(video_path(run))
    errors, warnings = vid.check_capture(info, PipelineConfig().capture)
    frame = vid.read_frame(video_path(run), 0.0)
    k = estimate_k(frame, PipelineConfig().court.distortion_max_residual_px)
    frame = Distortion(k).rectify(frame)
    cv2.imwrite(str(run_dir(run["id"]) / "frame0.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return update_run(run["id"], state="ready", info=info.to_dict(), capture_errors=errors,
                      capture_warnings=warnings, distortion_k=k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_webapp.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/webapp.py tests/test_webapp.py
git commit -m "feat: rectify frame0.jpg and record distortion_k in the webapp"
```

---

### Task 10: End-to-end regression — full synthetic match through fisheye distortion

**Files:**
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `synth.generate(..., distortion_k=...)` (Task 8), `resolve_court`/`track_players` fisheye wiring (Tasks 6-7)

- [ ] **Step 1: Write the test**

Add to `tests/test_end_to_end.py`:

```python
@pytest.fixture(scope="module")
def synthetic_fisheye(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_fisheye")
    return generate(out, SynthSpec(width=1280, height=720, duration_s=45.0, seed=3),
                    distortion_k=-0.22)


def test_pipeline_recovers_from_fisheye_distortion(synthetic_fisheye, tmp_path):
    meta = json.loads(synthetic_fisheye["meta"].read_text())
    cfg = PipelineConfig()
    cfg.detection.backend = "background"
    res = analyze(synthetic_fisheye["video"], tmp_path, cfg, me_px=tuple(meta["me_px"]),
                  use_llm=False, skip_capture_check=True)

    assert res.timeline.meta["court"]["source"] == "auto"
    assert res.timeline.meta["court"]["distortion_k"] != 0.0

    checks = {c.metric: c for c in validate_dir(tmp_path, synthetic_fisheye["labels"])}
    for metric in ("rally_count_err_pct", "shot_count_err_pct", "unflagged_swaps_per_5min",
                   "stats_without_confidence"):
        assert checks[metric].passed, checks[metric]
    assert checks["net_share_err_pp"].value <= 15.0
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_end_to_end.py::test_pipeline_recovers_from_fisheye_distortion -v`
Expected: PASS (this is the key regression proof: the analogous non-distorted assertions already pass in `test_pipeline_end_to_end`; this confirms the same holds once fisheye distortion is introduced and corrected)

This test is slow (renders and encodes a second ~45s synthetic video). That's consistent with the existing end-to-end test's cost.

- [ ] **Step 3: Run the full suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: end-to-end regression for fisheye correction on a synthetic match"
```

---

### Task 11: Move `px_per_m` to `court.py`; add `expected_player_height_px`

**Files:**
- Modify: `src/padelfusion/synth.py:130-133` (remove `px_per_m`, import from court)
- Modify: `src/padelfusion/court.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces:
  - `court.px_per_m(court: CourtModel, y: float) -> float` (moved, unchanged behavior)
  - `court.expected_player_height_px(court: CourtModel, y: float, real_height_m: float = 1.7) -> float`

- [ ] **Step 1: Write the failing test**

```python
def test_expected_player_height_decreases_with_depth():
    from padelfusion.court import expected_player_height_px

    m = CourtModel(image_corners(1280, 720))
    near = expected_player_height_px(m, 1.0)
    mid = expected_player_height_px(m, 10.0)
    far = expected_player_height_px(m, 19.0)
    assert near > mid > far > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_expected_player_height_decreases_with_depth -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Move `px_per_m` and add the new helper**

In `src/padelfusion/court.py`, add near the bottom (after `parse_corners`, before the "Automatic detection" section):

```python
def px_per_m(court: CourtModel, y: float) -> float:
    """In-plane pixel scale at court depth `y`: how many pixels wide 1 metre
    is at that depth. Used as a stand-in for vertical scale too (the camera
    isn't looking straight down), the same approximation `synth.py` already
    uses to draw players of a roughly-correct size."""
    p = court.court_to_image([[0, y], [10, y]])
    return float(np.linalg.norm(p[1] - p[0]) / 10.0)


def expected_player_height_px(court: CourtModel, y: float, real_height_m: float = 1.7) -> float:
    return real_height_m * px_per_m(court, y)
```

In `src/padelfusion/synth.py`, remove the now-duplicated `def px_per_m(...)` (currently lines 130-133) and add `px_per_m` to the existing `from .court import ...` line:

```python
from .court import COURT_CORNERS, COURT_LINES, CourtModel, NET_Y, px_per_m
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -v`
Expected: all PASS (including the existing synth-dependent tests, confirming the move didn't break `render_background`/`draw_player`)

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/court.py src/padelfusion/synth.py tests/test_units.py
git commit -m "refactor: move px_per_m to court.py, add expected_player_height_px"
```

---

### Task 12: `far_field_bbox`

**Files:**
- Modify: `src/padelfusion/court.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `far_field_bbox(court: CourtModel, frame_shape, margin_m: float) -> tuple[int, int, int, int] | None`

- [ ] **Step 1: Write the failing test**

```python
def test_far_field_bbox_covers_far_half_only():
    from padelfusion.court import far_field_bbox

    m = CourtModel(image_corners(1280, 720))
    bbox = far_field_bbox(m, (720, 1280, 3), margin_m=1.0)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 < x2 <= 1280
    assert 0 <= y1 < y2 <= 720
    # The far half (beyond the net) should sit in the upper part of the frame,
    # since the camera looks from the near baseline towards the far one.
    assert y2 < 720 * 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_far_field_bbox_covers_far_half_only -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement it**

In `src/padelfusion/court.py`, add near `in_court`:

```python
def far_field_bbox(court: CourtModel, frame_shape, margin_m: float) -> tuple[int, int, int, int] | None:
    """Pixel bounding box of the far half of the court (beyond the net),
    clipped to the frame. None if the projected region is degenerate."""
    h, w = frame_shape[:2]
    poly = court.court_to_image([
        [-margin_m, NET_Y], [COURT_WIDTH + margin_m, NET_Y],
        [COURT_WIDTH + margin_m, COURT_LENGTH + margin_m], [-margin_m, COURT_LENGTH + margin_m],
    ])
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_far_field_bbox_covers_far_half_only -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/court.py tests/test_units.py
git commit -m "feat: add far_field_bbox for court-aware far-region cropping"
```

---

### Task 13: `merge_detections` and `detect_two_pass`

**Files:**
- Modify: `src/padelfusion/detection.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces:
  - `merge_detections(dets_a: list[Detection], dets_b: list[Detection], iou_thresh: float = 0.3) -> list[Detection]`
  - `detect_two_pass(detect_fn, frame: np.ndarray, far_bbox: tuple[int, int, int, int] | None, upscale: float, iou_thresh: float = 0.3) -> list[Detection]` — `detect_fn: Callable[[np.ndarray], list[Detection]]`, decoupled from any specific backend so it's testable without YOLO installed.

- [ ] **Step 1: Write the failing tests**

```python
from padelfusion.detection import Detection, detect_two_pass, merge_detections


def _det(x1, y1, x2, y2, score=0.9):
    return Detection((x1, y1, x2, y2), score)


def test_merge_detections_collapses_overlapping_keeps_best_score():
    a = [_det(0, 0, 10, 30, 0.5)]
    b = [_det(1, 1, 11, 31, 0.9)]
    merged = merge_detections(a, b, iou_thresh=0.3)
    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_merge_detections_keeps_non_overlapping():
    a = [_det(0, 0, 10, 30)]
    b = [_det(500, 500, 510, 530)]
    merged = merge_detections(a, b, iou_thresh=0.3)
    assert len(merged) == 2


def test_detect_two_pass_runs_detector_on_full_and_upscaled_far_crop():
    calls = []

    def fake_detect(frame):
        calls.append(frame.shape)
        if frame.shape[0] == 200:  # the far-crop pass (cropped+upscaled to 200 tall)
            return [_det(10, 10, 20, 40)]
        return [_det(300, 300, 320, 360)]  # the full-frame pass

    frame = np.zeros((720, 1280, 3), np.uint8)
    far_bbox = (100, 100, 300, 200)  # 200 wide x 100 tall -> upscaled 2x -> 400x200
    dets = detect_two_pass(fake_detect, frame, far_bbox, upscale=2.0)

    assert len(calls) == 2
    assert len(dets) == 2
    far_det = next(d for d in dets if d.bbox[0] > 100)
    # (10,10)-(20,40) in the 2x-upscaled crop maps back to (100+5,100+5)-(100+10,100+20)
    assert abs(far_det.bbox[0] - 105.0) < 1e-6
    assert abs(far_det.bbox[1] - 105.0) < 1e-6


def test_detect_two_pass_without_far_bbox_runs_once():
    calls = []

    def fake_detect(frame):
        calls.append(frame.shape)
        return []

    frame = np.zeros((100, 100, 3), np.uint8)
    detect_two_pass(fake_detect, frame, None, upscale=2.0)
    assert len(calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "merge_detections or detect_two_pass" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement both functions**

In `src/padelfusion/detection.py`, add after the `Detection` dataclass:

```python
def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def merge_detections(dets_a: list[Detection], dets_b: list[Detection],
                     iou_thresh: float = 0.3) -> list[Detection]:
    """Pairs across the two lists whose IoU exceeds `iou_thresh` collapse to
    the higher-confidence detection; everything else from both lists survives."""
    used_b = set()
    merged = []
    for a in dets_a:
        best_j, best_iou = None, 0.0
        for j, b in enumerate(dets_b):
            if j in used_b:
                continue
            i = _iou(a, b)
            if i > best_iou:
                best_j, best_iou = j, i
        if best_j is not None and best_iou > iou_thresh:
            used_b.add(best_j)
            merged.append(a if a.score >= dets_b[best_j].score else dets_b[best_j])
        else:
            merged.append(a)
    merged.extend(b for j, b in enumerate(dets_b) if j not in used_b)
    return merged


def detect_two_pass(detect_fn, frame: np.ndarray, far_bbox: tuple[int, int, int, int] | None,
                    upscale: float, iou_thresh: float = 0.3) -> list[Detection]:
    """Run `detect_fn` on the full frame, and again on an upscaled crop of
    `far_bbox` (if given and `upscale > 1`), mapping the far pass's boxes
    back to full-frame coordinates before merging."""
    dets = detect_fn(frame)
    if far_bbox is None or upscale <= 1.0:
        return dets
    x1, y1, x2, y2 = far_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return dets
    up = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    far_dets = detect_fn(up)
    mapped = [Detection((d.bbox[0] / upscale + x1, d.bbox[1] / upscale + y1,
                         d.bbox[2] / upscale + x1, d.bbox[3] / upscale + y1), d.score, d.hist)
             for d in far_dets]
    return merge_detections(dets, mapped, iou_thresh)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "merge_detections or detect_two_pass" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/detection.py tests/test_units.py
git commit -m "feat: add merge_detections and detect_two_pass for far-court recall"
```

---

### Task 14: New `DetectionConfig` fields and default changes

**Files:**
- Modify: `src/padelfusion/config.py:36-43`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `DetectionConfig.far_field_upscale: float = 2.0`, `far_field_enabled: bool = True`, `bg_min_height_frac: float = 0.35`; `yolo_model` default becomes `"yolov8s.pt"`; `yolo_min_conf` default becomes `0.25`.

- [ ] **Step 1: Write the failing test**

```python
def test_detection_config_defaults():
    cfg = DetectionConfig()
    assert cfg.yolo_model == "yolov8s.pt"
    assert cfg.yolo_min_conf == 0.25
    assert cfg.far_field_upscale == 2.0
    assert cfg.far_field_enabled is True
    assert cfg.bg_min_height_frac == 0.35
```

Add `DetectionConfig` to the existing `from padelfusion.config import ...` line in `tests/test_units.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_detection_config_defaults -v`
Expected: FAIL (`AttributeError` and/or wrong default values)

- [ ] **Step 3: Update `DetectionConfig`**

In `src/padelfusion/config.py`:

```python
@dataclass
class DetectionConfig:
    backend: str = "auto"  # auto | yolo | background
    yolo_model: str = "yolov8s.pt"
    yolo_min_conf: float = 0.25
    bg_samples: int = 60
    bg_diff_threshold: int = 35
    bg_min_area_frac: float = 0.00025
    # Depth-aware background-detector gate: a component's bbox height must
    # clear this fraction of the expected on-screen player height at its
    # court depth (court.expected_player_height_px). Used only when a
    # CourtModel is available; otherwise bg_min_area_frac is the sole gate.
    bg_min_height_frac: float = 0.35
    # Court-aware two-pass detection: also run detection on an upscaled crop
    # of the far half of the court, to recover small/distant players.
    far_field_upscale: float = 2.0
    far_field_enabled: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_detection_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/config.py tests/test_units.py
git commit -m "feat: bump YOLO defaults, add far-field and depth-aware detection config"
```

---

### Task 15: Wire two-pass detection into `YoloDetector`

**Files:**
- Modify: `src/padelfusion/detection.py` (`YoloDetector`, `make_detector`)

**Interfaces:**
- Consumes: `detect_two_pass` (Task 13)
- Produces: `YoloDetector(cfg, far_bbox: tuple[int, int, int, int] | None = None)`; `make_detector(cfg, sample_frames_fn, court_mask=None, court=None, far_bbox=None)` — new optional kwargs, all defaulting to today's exact behavior.

- [ ] **Step 1: Update `YoloDetector` and `make_detector`**

No dedicated new test here — `ultralytics` isn't a core dependency, and the two-pass *logic* is already covered by Task 13's `detect_two_pass` tests against a fake `detect_fn`. This task is a thin, low-risk wiring change; Task 18's end-to-end test (background backend) is what exercises the wiring live.

In `src/padelfusion/detection.py`:

```python
class YoloDetector:
    name = "yolo"

    def __init__(self, cfg: DetectionConfig, far_bbox: tuple[int, int, int, int] | None = None):
        from ultralytics import YOLO  # optional dependency

        self.cfg = cfg
        self.far_bbox = far_bbox if cfg.far_field_enabled else None
        self.model = YOLO(cfg.yolo_model)

    def _raw_detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, classes=[0], conf=self.cfg.yolo_min_conf, verbose=False)[0]
        dets = []
        for box, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            bbox = tuple(float(v) for v in box)
            dets.append(Detection(bbox, float(conf), appearance_hist(frame, bbox)))
        return dets

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return detect_two_pass(self._raw_detect, frame, self.far_bbox, self.cfg.far_field_upscale)
```

```python
def make_detector(cfg: DetectionConfig, sample_frames_fn, court_mask=None, court=None,
                  far_bbox: tuple[int, int, int, int] | None = None):
    backend = cfg.backend
    if backend == "auto":
        backend = "yolo" if yolo_available() else "background"
    if backend == "yolo":
        return YoloDetector(cfg, far_bbox=far_bbox)
    if backend == "background":
        return BackgroundDetector(cfg, sample_frames_fn(), court_mask, court=court)
    raise ValueError(f"unknown detector backend: {cfg.backend}")
```

(`BackgroundDetector`'s new `court=` kwarg is added in Task 16; this task's edit to `make_detector` and `BackgroundDetector(...)` call is written in anticipation of it — if executing tasks out of order, add `court=None` as an accepted-but-unused kwarg to `BackgroundDetector.__init__` in this task instead, then flesh it out in Task 16.)

- [ ] **Step 2: Run the full unit suite to confirm no regression**

Run: `pytest tests/test_units.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/padelfusion/detection.py
git commit -m "feat: wire court-aware two-pass detection into YoloDetector"
```

---

### Task 16: Depth-aware threshold in `BackgroundDetector`

**Files:**
- Modify: `src/padelfusion/detection.py` (`BackgroundDetector`)
- Test: `tests/test_units.py`

**Interfaces:**
- Consumes: `court.expected_player_height_px` (Task 11)
- Produces: `BackgroundDetector(cfg, sample_frames, court_mask=None, court: CourtModel | None = None)` — depth-aware gate when `court` is given, else today's exact flat `bg_min_area_frac` gate.

- [ ] **Step 1: Write the failing test**

```python
def test_background_detector_depth_aware_threshold_admits_small_far_blob():
    from padelfusion.court import expected_player_height_px
    from padelfusion.detection import BackgroundDetector

    court = CourtModel(image_corners(1280, 720))
    cfg = DetectionConfig(backend="background", bg_min_area_frac=0.01)  # deliberately strict flat gate
    bg = np.full((720, 1280, 3), 40, np.uint8)
    far_h = expected_player_height_px(court, 18.0)
    blob_h = int(0.5 * far_h)  # clears the depth-aware gate (0.35 * far_h) ...
    blob_w = max(2, int(blob_h * 0.9))
    foot = court.court_to_image([[5.0, 18.0]])[0]
    x0, y0 = int(foot[0] - blob_w / 2), int(foot[1] - blob_h)

    frame_flat = bg.copy()
    frame_flat[y0:y0 + blob_h, x0:x0 + blob_w] = (200, 200, 200)
    detector_flat = BackgroundDetector(cfg, [bg])
    assert detector_flat.detect(frame_flat) == []  # ... but not the old flat one

    detector_depth = BackgroundDetector(cfg, [bg], court=court)
    dets = detector_depth.detect(frame_flat)
    assert len(dets) == 1
```

Add `DetectionConfig` and `BackgroundDetector` to the relevant import lines if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_background_detector_depth_aware_threshold_admits_small_far_blob -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'court'`)

- [ ] **Step 3: Update `BackgroundDetector`**

In `src/padelfusion/detection.py` (add `from .court import CourtModel, expected_player_height_px` to the imports):

```python
class BackgroundDetector:
    name = "background"

    def __init__(self, cfg: DetectionConfig, sample_frames: list[np.ndarray],
                 court_mask: np.ndarray | None = None, court: "CourtModel | None" = None):
        if not sample_frames:
            raise ValueError("background detector needs sample frames")
        self.cfg = cfg
        self.background = np.median(np.stack(sample_frames), axis=0).astype(np.uint8)
        self.court_mask = court_mask
        self.court = court
        h, w = self.background.shape[:2]
        self.min_area = cfg.bg_min_area_frac * h * w

    def detect(self, frame: np.ndarray) -> list[Detection]:
        diff = cv2.absdiff(frame, self.background)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(diff, self.cfg.bg_diff_threshold, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
        if self.court_mask is not None:
            fg = cv2.bitwise_and(fg, self.court_mask)
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg)
        edges = _edge_energy(frame)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if h < w * 0.8:
                continue
            if self.court is not None:
                foot = self.court.image_to_court([[x + w / 2.0, y + h]])[0]
                if h < self.cfg.bg_min_height_frac * expected_player_height_px(self.court, foot[1]):
                    continue
            elif area < self.min_area:
                continue
            if self._is_ghost(edges, x, y, w, h):
                continue
            bbox = (float(x), float(y), float(x + w), float(y + h))
            dets.append(Detection(bbox, 1.0, appearance_hist(frame, bbox)))
        return dets

    def _is_ghost(self, edges: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        p = 3
        ys, xs = slice(max(y - p, 0), y + h + p), slice(max(x - p, 0), x + w + p)
        now, bg = float(edges[ys, xs].mean()), float(self.bg_edges[ys, xs].mean())
        return now < 0.5 * bg

    @property
    def bg_edges(self) -> np.ndarray:
        if not hasattr(self, "_bg_edges"):
            self._bg_edges = _edge_energy(self.background)
        return self._bg_edges
```

(Only `court` storage, the reordered/extended per-component gate, and the docstring-comment are new; `_is_ghost`/`bg_edges`/everything else is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/detection.py tests/test_units.py
git commit -m "feat: depth-aware area/height gate in BackgroundDetector"
```

---

### Task 17: Thread `court`/`far_bbox` through `track_players`

**Files:**
- Modify: `src/padelfusion/pipeline.py:75-108` (`track_players`, continuing from Task 7)

**Interfaces:**
- Consumes: `far_field_bbox` (Task 12), updated `make_detector` (Tasks 15/16)

- [ ] **Step 1: Update `track_players`**

In `src/padelfusion/pipeline.py`, add the import (`from .court import far_field_bbox`) and change the `make_detector` call (everything else in the function is exactly as Task 7 left it):

```python
def track_players(video_path, court: CourtModel, cfg: PipelineConfig, me_px=None,
                  play_times: list[float] | None = None, on_frame=None):
    tcfg = cfg.tracking
    dist = Distortion(court.k)
    first = vid.read_frame(video_path, 0.0)
    mask = court_region_mask(court, first.shape, cfg.court.out_of_court_margin_m + 0.5)
    far_bbox = far_field_bbox(court, first.shape, cfg.court.out_of_court_margin_m + 0.5)
    times = None
    if play_times and len(play_times) >= 10:
        times = sorted(t + d for t in play_times for d in (-0.5, 0.0, 0.5) if t + d >= 0)
    detector = make_detector(
        cfg.detection,
        lambda: [dist.rectify(f) for f in vid.sample_frames(video_path, cfg.detection.bg_samples, times)],
        court_mask=mask, court=court, far_bbox=far_bbox)
    tracker = Tracker(tcfg)
    initialized = False
    n_samples = 0
    for _, t, frame in vid.iter_frames(video_path, tcfg.analysis_fps):
        frame = dist.rectify(frame)
        n_samples += 1
        if on_frame is not None:
            on_frame(t)
        cdets = to_court_detections(detector.detect(frame), court, cfg.court.out_of_court_margin_m)
        if not initialized:
            if t > tcfg.init_search_s:
                raise TrackingInitError(
                    f"could not find two players on each side of the net in the first "
                    f"{tcfg.init_search_s:.0f}s; check the capture framing")
            initialized = tracker.initialize(t, cdets, me_px)
            continue
        tracker.update(t, cdets)
    if not initialized:
        raise TrackingInitError("no frames with four players found")
    points, segments = tracker.finalize()
    return tracker, points, segments, detector.name, n_samples
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all PASS (including `test_pipeline_end_to_end`, confirming `court`/`far_bbox` threading doesn't change the background backend's output on the existing synthetic match — its player sizes already clear both the old flat gate and the new depth-aware one)

- [ ] **Step 3: Commit**

```bash
git add src/padelfusion/pipeline.py
git commit -m "feat: thread court and far-field bbox into the detector"
```

---

### Task 18: End-to-end sanity check with depth-aware detection enabled

**Files:**
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- No new production interface — this is a non-regression check, not the primary proof the fix works (that's Task 16's targeted unit test, which constructs a blob precisely sized to fail the old flat gate and pass the new depth-aware one).

- [ ] **Step 1: Write the test**

Add to `tests/test_end_to_end.py`:

```python
def test_pipeline_end_to_end_unaffected_by_depth_aware_detection(synthetic, tmp_path):
    """Non-regression: the existing synthetic match, run through the same
    background backend now that it's depth-aware (court + far_bbox wired
    through track_players in pipeline.py), still validates the same way.
    The targeted proof that the depth-aware gate itself admits small far
    blobs the old flat gate rejected is
    test_background_detector_depth_aware_threshold_admits_small_far_blob in
    test_units.py."""
    meta = json.loads(synthetic["meta"].read_text())
    cfg = PipelineConfig()
    cfg.detection.backend = "background"
    res = analyze(synthetic["video"], tmp_path, cfg, me_px=tuple(meta["me_px"]),
                  use_llm=False, skip_capture_check=True)

    checks = {c.metric: c for c in validate_dir(tmp_path, synthetic["labels"])}
    for metric in ("rally_count_err_pct", "shot_count_err_pct", "unflagged_swaps_per_5min",
                   "stats_without_confidence"):
        assert checks[metric].passed, checks[metric]
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_end_to_end.py::test_pipeline_end_to_end_unaffected_by_depth_aware_detection -v`
Expected: PASS (this exercises code already in place from Tasks 14-16; no production code changes in this task)

- [ ] **Step 3: Run the entire test suite one last time**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: confirm depth-aware background detection doesn't regress the existing synthetic match"
```

---

## Deferred (not in this plan)

- **Part 1B — manual multi-point calibration UI.** Only reachable when auto self-calibration (Task 4) rejects a video. Needs a new webapp endpoint plus a frontend step in `web/index.html`'s corner-picking flow. Scoped out per the design spec's own framing of it as a fallback for the (expected to be rare, after Tasks 1-10) residual failure case. Revisit if real fisheye footage turns out to need it in practice.
- **Tuning `distortion_max_residual_px` / the `k` search bounds `(-0.6, 0.05)` against real footage.** The synthetic tests validate the *mechanism*; the specific thresholds should be checked against the user's actual fisheye clip once this lands, per the design spec's "Risks" section.
