# Ball-based shot detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect shots from the ball's visual trajectory (primary signal), with audio (when present) only filling in gaps the ball detector misses — giving usable shot/rally stats on footage with no audio track at all.

**Architecture:** A new `ball.py` module does color-threshold ball-candidate detection (excluding anything inside a tracked player's bounding box, which is what makes this robust against same-colored shoes/gear), a gap-tolerant single-object trajectory tracker, and hit detection from sharp trajectory-direction changes near a player. It's wired into `pipeline.track_players`'s existing per-frame loop (reusing the same per-frame player detections, no extra inference pass) and `pipeline.analyze()` merges its output with audio-detected shots, recording which source(s) contributed via a new `timeline.meta["shots_source"]` field that `stats.py` uses to downgrade confidence appropriately.

**Tech Stack:** Python 3.10+, NumPy, OpenCV (`cv2.inRange`, `cv2.connectedComponentsWithStats`), pytest. No new dependency.

## Global Constraints

- Every existing test must keep passing unchanged. `generate()`'s new ball-rendering is opt-in (`render_ball: bool = False`), so all existing synthetic fixtures are untouched.
- A video with no rendered/real ball must behave exactly as today: `ball_shots` ends up empty, `merge_with_audio_gaps([], audio_shots, tol)` returns all of `audio_shots` unchanged (the "far from any ball shot" condition is vacuously true when there are no ball shots), and `shots_source` becomes `"audio"`.
- The trajectory-kink thresholds (`kink_angle_deg`, `max_speed_mps`, `max_gap_s`, `close_to_player_m`) are expected to need real-footage tuning after this lands — this plan validates the *mechanism* against synthetic data with exact known kink points, not final real-world threshold values.

---

### Task 1: `BallConfig`

**Files:**
- Modify: `src/padelfusion/config.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `BallConfig` dataclass; `PipelineConfig.ball: BallConfig`.

- [ ] **Step 1: Write the failing test**

```python
def test_ball_config_defaults():
    cfg = BallConfig()
    assert cfg.kink_angle_deg == 55.0
    assert cfg.max_speed_mps == 35.0
    assert cfg.audio_merge_tolerance_s == 0.25
```

Add `BallConfig` to the existing `from padelfusion.config import (...)` block.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_ball_config_defaults -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add the config**

In `src/padelfusion/config.py`, add after `AudioConfig`:

```python
@dataclass
class BallConfig:
    # HSV bounds for the ball's yellow-green color (OpenCV hue range 0-179).
    hsv_lower: tuple[int, int, int] = (25, 80, 100)
    hsv_upper: tuple[int, int, int] = (45, 255, 255)
    min_size_px: int = 3
    max_size_px: int = 30
    min_aspect: float = 0.4
    max_aspect: float = 2.5
    # A candidate blob inside any player's bbox (+ this margin) is dropped —
    # this is what excludes same-colored shoes/gear.
    player_exclude_margin_px: float = 6.0
    max_speed_mps: float = 35.0
    max_gap_s: float = 0.5
    # Minimum incoming/outgoing direction change (degrees) to call a
    # trajectory point a hit, rather than smooth gravity-curved flight.
    kink_angle_deg: float = 55.0
    close_to_player_m: float = 2.5
    min_interval_s: float = 0.15
    # An audio onset within this of an existing ball-detected shot is treated
    # as the same event and not added again.
    audio_merge_tolerance_s: float = 0.25
```

And in `PipelineConfig`:

```python
@dataclass
class PipelineConfig:
    capture: CaptureSpec = field(default_factory=CaptureSpec)
    court: CourtConfig = field(default_factory=CourtConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    shots: ShotConfig = field(default_factory=ShotConfig)
    rallies: RallyConfig = field(default_factory=RallyConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_ball_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/config.py tests/test_units.py
git commit -m "feat: add BallConfig"
```

---

### Task 2: `ball.detect_ball_candidates`

**Files:**
- Create: `src/padelfusion/ball.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `detect_ball_candidates(frame, court_mask, player_bboxes, cfg) -> list[tuple[float, float]]`

- [ ] **Step 1: Write the failing test**

```python
def test_detect_ball_candidates_excludes_blobs_inside_player_bboxes():
    from padelfusion.ball import detect_ball_candidates

    cfg = BallConfig()
    hsv_mid = np.uint8([[[35, 200, 220]]])
    ball_bgr = tuple(int(c) for c in cv2.cvtColor(hsv_mid, cv2.COLOR_HSV2BGR)[0, 0])
    frame = np.full((200, 300, 3), (60, 60, 60), np.uint8)
    cv2.circle(frame, (50, 50), 6, ball_bgr, -1)    # free ball
    cv2.circle(frame, (150, 150), 6, ball_bgr, -1)  # same color, inside a player bbox
    player_bboxes = [(120.0, 120.0, 180.0, 180.0)]
    cands = detect_ball_candidates(frame, None, player_bboxes, cfg)
    assert len(cands) == 1
    assert abs(cands[0][0] - 50) < 3 and abs(cands[0][1] - 50) < 3
```

Add `BallConfig` import if not already present from Task 1.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_detect_ball_candidates_excludes_blobs_inside_player_bboxes -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'padelfusion.ball'`

- [ ] **Step 3: Create `ball.py`**

```python
"""Ball-based shot detection: color-threshold candidate detection, a
gap-tolerant single-object trajectory tracker, and hit detection from sharp
trajectory direction changes near a tracked player.

Shot detection here is the primary signal (per the user's real footage
having no audio at all); `pipeline.analyze` merges this module's output
with audio-detected shots, using audio only to fill gaps this module missed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import BallConfig


def detect_ball_candidates(frame: np.ndarray, court_mask: np.ndarray | None,
                           player_bboxes: list[tuple[float, float, float, float]],
                           cfg: BallConfig) -> list[tuple[float, float]]:
    """Ball-colored blob centroids (pixel coordinates), excluding anything
    inside a player's bounding box (kills same-colored shoes/gear)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))
    if court_mask is not None:
        mask = cv2.bitwise_and(mask, court_mask)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    m = cfg.player_exclude_margin_px
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (cfg.min_size_px <= w <= cfg.max_size_px and cfg.min_size_px <= h <= cfg.max_size_px):
            continue
        if not (cfg.min_aspect < w / h < cfg.max_aspect):
            continue
        cx, cy = centroids[i]
        if any(x1 - m <= cx <= x2 + m and y1 - m <= cy <= y2 + m for x1, y1, x2, y2 in player_bboxes):
            continue
        cands.append((float(cx), float(cy)))
    return cands
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_detect_ball_candidates_excludes_blobs_inside_player_bboxes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/ball.py tests/test_units.py
git commit -m "feat: ball-candidate detection with player-bbox exclusion"
```

---

### Task 3: `ball.BallTracker`

**Files:**
- Modify: `src/padelfusion/ball.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `BallPoint` dataclass (`t`, `x`, `y`); `BallTracker(cfg)` with `.update(t, candidates_xy)` and `.points: list[BallPoint]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ball_tracker_links_a_smooth_trajectory():
    from padelfusion.ball import BallTracker

    cfg = BallConfig(max_speed_mps=10.0, max_gap_s=0.5)
    tr = BallTracker(cfg)
    for i in range(5):
        tr.update(i * 0.1, [(1.0 + 0.2 * i, 2.0)])
    assert len(tr.points) == 5


def test_ball_tracker_starts_new_segment_outside_speed_gate():
    from padelfusion.ball import BallTracker

    cfg = BallConfig(max_speed_mps=5.0, max_gap_s=0.5)
    tr = BallTracker(cfg)
    tr.update(0.0, [(1.0, 2.0)])
    tr.update(0.1, [(50.0, 2.0)])  # ~490 m/s -> far outside a 5 m/s gate
    assert len(tr.points) == 1


def test_ball_tracker_links_across_a_short_gap():
    from padelfusion.ball import BallTracker

    cfg = BallConfig(max_speed_mps=20.0, max_gap_s=0.5)
    tr = BallTracker(cfg)
    tr.update(0.0, [(1.0, 2.0)])
    tr.update(0.4, [(2.0, 2.0)])  # 0.4s gap (under max_gap_s), 2.5 m/s (under the speed gate)
    assert len(tr.points) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "ball_tracker" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `BallPoint` and `BallTracker`**

Append to `src/padelfusion/ball.py`:

```python
@dataclass
class BallPoint:
    t: float
    x: float
    y: float


class BallTracker:
    """Single-object, gap-tolerant tracker in court-coordinate space."""

    def __init__(self, cfg: BallConfig):
        self.cfg = cfg
        self.points: list[BallPoint] = []

    def update(self, t: float, candidates_xy: list[tuple[float, float]]) -> None:
        if not candidates_xy:
            return
        if not self.points:
            x, y = candidates_xy[0]
            self.points.append(BallPoint(t, x, y))
            return
        last = self.points[-1]
        dt = t - last.t
        if dt > self.cfg.max_gap_s:
            x, y = candidates_xy[0]
            self.points.append(BallPoint(t, x, y))
            return
        gate = self.cfg.max_speed_mps * dt
        best = min(candidates_xy, key=lambda c: np.hypot(c[0] - last.x, c[1] - last.y))
        d = float(np.hypot(best[0] - last.x, best[1] - last.y))
        if d <= gate:
            self.points.append(BallPoint(t, best[0], best[1]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "ball_tracker" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/ball.py tests/test_units.py
git commit -m "feat: gap-tolerant single-object ball trajectory tracker"
```

---

### Task 4: `ball.detect_ball_shots`

**Files:**
- Modify: `src/padelfusion/ball.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Consumes: `BallPoint` (Task 3), `tracking.TrackPoint`, `events.Shot`
- Produces: `detect_ball_shots(points: list[BallPoint], player_tracks: dict[str, list[TrackPoint]], cfg: BallConfig) -> list[Shot]`

- [ ] **Step 1: Write the failing tests**

```python
def test_detect_ball_shots_finds_a_hit_near_a_player():
    from padelfusion.ball import BallPoint, detect_ball_shots
    from padelfusion.tracking import TrackPoint

    points = [
        BallPoint(0.0, 0.0, 5.0),
        BallPoint(0.1, 1.0, 5.0),
        BallPoint(0.2, 2.0, 5.0),  # kink here: incoming +x, outgoing -x
        BallPoint(0.3, 1.0, 5.0),
        BallPoint(0.4, 0.0, 5.0),
    ]
    player_tracks = {"P1": [TrackPoint(0.2, 2.0, 5.0, (0, 0, 10, 30))]}
    cfg = BallConfig(kink_angle_deg=90.0, close_to_player_m=1.0)
    shots = detect_ball_shots(points, player_tracks, cfg)
    assert len(shots) == 1
    assert shots[0].attributed_player == "P1"
    assert abs(shots[0].t - 0.2) < 1e-6


def test_detect_ball_shots_ignores_a_kink_far_from_any_player():
    from padelfusion.ball import BallPoint, detect_ball_shots
    from padelfusion.tracking import TrackPoint

    points = [
        BallPoint(0.0, 0.0, 5.0),
        BallPoint(0.1, 1.0, 5.0),
        BallPoint(0.2, 2.0, 5.0),
        BallPoint(0.3, 1.0, 5.0),
        BallPoint(0.4, 0.0, 5.0),
    ]
    player_tracks = {"P1": [TrackPoint(0.2, 20.0, 15.0, (0, 0, 10, 30))]}  # far away
    cfg = BallConfig(kink_angle_deg=90.0, close_to_player_m=1.0)
    shots = detect_ball_shots(points, player_tracks, cfg)
    assert shots == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "detect_ball_shots" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement it**

Append to `src/padelfusion/ball.py` (add `from .events import Shot` and `from .tracking import TrackPoint` to the imports):

```python
from .events import Shot
from .tracking import TrackPoint


def detect_ball_shots(points: list[BallPoint], player_tracks: dict[str, list[TrackPoint]],
                      cfg: BallConfig) -> list[Shot]:
    """A shot is a trajectory point where the incoming/outgoing direction
    changes sharply (a racket hit, unlike gravity-curved flight) and some
    tracked player is nearby at that instant (excludes floor/glass bounces
    away from play)."""
    if len(points) < 3:
        return []
    shots: list[Shot] = []
    last_t = -1e9
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        if b.t - a.t > cfg.max_gap_s or c.t - b.t > cfg.max_gap_s:
            continue
        v_in = np.array([b.x - a.x, b.y - a.y])
        v_out = np.array([c.x - b.x, c.y - b.y])
        n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if n_in < 1e-6 or n_out < 1e-6:
            continue
        cos_ang = np.dot(v_in, v_out) / (n_in * n_out)
        angle = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))
        if angle < cfg.kink_angle_deg:
            continue
        if b.t - last_t < cfg.min_interval_s:
            continue
        best_pid, best_d = None, cfg.close_to_player_m
        for pid, pts in player_tracks.items():
            near = min(pts, key=lambda p: abs(p.t - b.t), default=None)
            if near is None or abs(near.t - b.t) > 0.3:
                continue
            d = float(np.hypot(near.x - b.x, near.y - b.y))
            if d < best_d:
                best_pid, best_d = pid, d
        if best_pid is None:
            continue
        shots.append(Shot(b.t, angle, 0.0, "ball", best_pid, 1.0, attributed_player=best_pid))
        last_t = b.t
    return shots
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "detect_ball_shots" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/ball.py tests/test_units.py
git commit -m "feat: detect ball-trajectory shots from sharp direction changes"
```

---

### Task 5: `ball.merge_with_audio_gaps`

**Files:**
- Modify: `src/padelfusion/ball.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `merge_with_audio_gaps(ball_shots: list[Shot], audio_shots: list[Shot], tolerance_s: float) -> list[Shot]`

- [ ] **Step 1: Write the failing test**

```python
def test_merge_with_audio_gaps_adds_far_shot_and_drops_close_one():
    from padelfusion.ball import merge_with_audio_gaps

    ball_shots = [Shot(1.0, 10.0, 0.0, "ball", "P1", 1.0, attributed_player="P1")]
    audio_far = Shot(5.0, 1.0, -10.0, "motion", "P2", 1.0, attributed_player="P2")
    audio_close = Shot(1.1, 1.0, -10.0, "motion", "P2", 1.0, attributed_player="P2")
    merged = merge_with_audio_gaps(ball_shots, [audio_far, audio_close], tolerance_s=0.25)
    assert [round(s.t, 2) for s in merged] == [1.0, 5.0]


def test_merge_with_audio_gaps_adds_everything_when_no_ball_shots():
    from padelfusion.ball import merge_with_audio_gaps

    audio_shots = [Shot(1.0, 1.0, -10.0, "motion", "P1", 1.0, attributed_player="P1"),
                  Shot(2.0, 1.0, -10.0, "motion", "P2", 1.0, attributed_player="P2")]
    merged = merge_with_audio_gaps([], audio_shots, tolerance_s=0.25)
    assert [round(s.t, 2) for s in merged] == [1.0, 2.0]
```

The second test is the exact backward-compatibility guarantee this plan relies on: a video with no ball rendered/detected must get all of its audio shots back unchanged.

Add `from padelfusion.events import Shot` to the test file's imports if not already present (check first — `Shot` is already imported per the existing rally-segmentation tests).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "merge_with_audio_gaps" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement it**

Append to `src/padelfusion/ball.py`:

```python
def merge_with_audio_gaps(ball_shots: list[Shot], audio_shots: list[Shot],
                          tolerance_s: float) -> list[Shot]:
    """Ball shots plus any audio shot not within `tolerance_s` of an existing
    ball shot — audio only fills gaps the ball detector missed."""
    out = list(ball_shots)
    ball_times = [s.t for s in ball_shots]
    for s in audio_shots:
        if all(abs(s.t - bt) > tolerance_s for bt in ball_times):
            out.append(s)
    return sorted(out, key=lambda s: s.t)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "merge_with_audio_gaps" -v`
Expected: all PASS

Also run the full unit suite:

Run: `pytest tests/test_units.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/ball.py tests/test_units.py
git commit -m "feat: merge ball shots with audio-filled gaps"
```

---

### Task 6: Wire ball detection into `pipeline.track_players`

**Files:**
- Modify: `src/padelfusion/pipeline.py`

**Interfaces:**
- Consumes: `ball.BallTracker`, `ball.detect_ball_candidates` (Tasks 2-3)
- Produces: `track_players(...) -> (tracker, points, segments, detector_name, n_samples, ball_points)` — new 6th return value, `list[BallPoint]`.

- [ ] **Step 1: Update `track_players`**

In `src/padelfusion/pipeline.py`, add the import (`from .ball import BallTracker, detect_ball_candidates`) and change:

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
    far_bbox = far_field_bbox(court, first.shape, cfg.court.out_of_court_margin_m + 0.5)
    times = None
    if play_times and len(play_times) >= 10:
        times = sorted(t + d for t in play_times for d in (-0.5, 0.0, 0.5) if t + d >= 0)
    detector = make_detector(
        cfg.detection,
        lambda: [dist.rectify(f) for f in vid.sample_frames(video_path, cfg.detection.bg_samples, times)],
        court_mask=mask, court=court, far_bbox=far_bbox)
    tracker = Tracker(tcfg)
    ball_tracker = BallTracker(cfg.ball)
    initialized = False
    n_samples = 0
    for _, t, frame in vid.iter_frames(video_path, tcfg.analysis_fps):
        frame = dist.rectify(frame)
        n_samples += 1
        if on_frame is not None:
            on_frame(t)
        raw_dets = detector.detect(frame)
        cdets = to_court_detections(raw_dets, court, cfg.court.out_of_court_margin_m)
        ball_px = detect_ball_candidates(frame, mask, [d.bbox for d in raw_dets], cfg.ball)
        if ball_px:
            ball_tracker.update(t, [tuple(p) for p in court.image_to_court(ball_px)])
        if not initialized:
            if t > tcfg.init_search_s:
                raise TrackingInitError(
                    f"could not find your tagged player in the first "
                    f"{tcfg.init_search_s:.0f}s; check the capture framing")
            initialized = tracker.initialize(t, cdets, me_px)
            continue
        tracker.update(t, cdets)
    if not initialized:
        raise TrackingInitError("no frames with any player found")
    points, segments = tracker.finalize()
    return tracker, points, segments, detector.name, n_samples, ball_tracker.points
```

(Only `detector.detect(frame)` being captured once as `raw_dets` and reused, the `ball_tracker`/`detect_ball_candidates` additions, and the new 6th return value are changes — everything else is unchanged.)

- [ ] **Step 2: Run the full test suite**

`track_players` has one caller (`analyze`, updated in Task 7) — running the suite now will show `analyze` failing to unpack 5 values from a 6-tuple. That's expected and fixed in Task 7; this step exists to confirm the *rest* of the suite (anything not touching `analyze`) is unaffected.

Run: `pytest tests/test_units.py -v`
Expected: all PASS (nothing in test_units.py calls `track_players` or `analyze` directly)

- [ ] **Step 3: Commit**

```bash
git add src/padelfusion/pipeline.py
git commit -m "feat: track ball candidates alongside players in track_players"
```

---

### Task 7: Wire ball shots into `pipeline.analyze` and `build_timeline`

**Files:**
- Modify: `src/padelfusion/pipeline.py`

**Interfaces:**
- Consumes: `ball.detect_ball_shots`, `ball.merge_with_audio_gaps` (Tasks 4-5), `track_players`'s new 6th return value (Task 6)
- Produces: `build_timeline(..., shots_source: str)` — new required parameter; `timeline.meta["shots_source"]`.

- [ ] **Step 1: Update `build_timeline`**

In `src/padelfusion/pipeline.py`:

```python
def build_timeline(info: vid.VideoInfo, court: CourtModel, cfg: PipelineConfig, tracker: Tracker,
                   points, segments, shots, rallies, rejected, isolated, detector_name: str,
                   n_samples: int, warnings: list[str], shots_source: str) -> Timeline:
    positions = []
    for pid, pts in points.items():
        for p in pts:
            positions.append({
                "t": round(p.t, 3), "player": pid, "team": team_side(p.y),
                "x": round(p.x, 3), "y": round(p.y, 3),
                "zone": zone(p.y, cfg.court.net_zone_depth_m),
                "interpolated": p.interpolated, "flags": list(p.flags),
                "bbox_w": round(p.bbox[2] - p.bbox[0], 1), "bbox_h": round(p.bbox[3] - p.bbox[1], 1),
            })
    positions.sort(key=lambda r: (r["t"], r["player"]))
    meta = {
        "video": str(info.path),
        "video_info": info.to_dict(),
        "duration_s": info.duration_s,
        "analysis_fps": cfg.tracking.analysis_fps,
        "n_samples": n_samples,
        "court": court.to_dict(),
        "detector": detector_name,
        "shots_source": shots_source,
        "config": cfg.to_dict(),
        "warnings": warnings,
    }
    players = {pid: {"team": tr.team, "is_user": tr.is_user} for pid, tr in tracker.tracks.items()}
    return Timeline(meta, players, positions, [s.to_dict() for s in shots],
                    [r.to_dict() for r in rallies], segments, rejected, [round(t, 3) for t in isolated])
```

- [ ] **Step 2: Update `analyze`**

In `src/padelfusion/pipeline.py`, add the import (`from .ball import detect_ball_shots, merge_with_audio_gaps`) and change the body from `step("Listening for ball impacts", 0.05)` through the `build_timeline(...)` call:

```python
    step("Listening for ball impacts", 0.05)
    if info.has_audio:
        audio = vid.extract_audio(video_path, cfg.audio.sample_rate)
        onsets, quiet = detect_onsets(audio, cfg.audio)
    else:
        onsets, quiet = [], []

    step("Tracking players", 0.10)
    duration = max(info.duration_s, 1e-6)
    last_report = [0.0]

    def on_frame(t: float) -> None:
        if progress is not None and t - last_report[0] >= 2.0:
            last_report[0] = t
            progress("Tracking players", 0.10 + 0.75 * min(t / duration, 1.0))

    tracker, points, segments, det_name, n_samples, ball_points = track_players(
        video_path, court, cfg, me_px, play_times=[o.t for o in onsets], on_frame=on_frame)
    if me_px is None:
        warnings.append("no self-tag given: P1 defaults to the near-left player")

    step("Finding shots and rallies", 0.87)
    ball_shots = detect_ball_shots(ball_points, points, cfg.ball)
    audio_shots, rejected = fuse_shots(onsets, points, cfg.shots)
    rejected += [{**o.to_dict(), "reason": "too_quiet"} for o in quiet]
    shots = merge_with_audio_gaps(ball_shots, audio_shots, cfg.ball.audio_merge_tolerance_s) \
        if info.has_audio else ball_shots
    if ball_shots and audio_shots:
        shots_source = "ball+audio"
    elif ball_shots:
        shots_source = "ball"
    elif audio_shots:
        shots_source = "audio"
    else:
        shots_source = "none"
        warnings.append("no shots detected from audio or the ball trajectory: shot and rally stats are empty")
    rallies, isolated = segment_rallies(shots, points, cfg.rallies)

    tl = build_timeline(info, court, cfg, tracker, points, segments, shots, rallies,
                        sorted(rejected, key=lambda r: r["t"]), isolated, det_name, n_samples, warnings,
                        shots_source)
    tl.save(out)
```

Note the old unconditional `warnings.append("no audio: shot and rally stats are empty")` inside the `else: onsets, quiet = [], []` branch is removed — it's no longer accurate (ball detection may well produce shots with no audio at all), replaced by the single `shots_source == "none"` warning above, which covers the actually-empty case regardless of why.

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all PASS. This is the key regression check: the existing synthetic match fixtures render no ball, so `ball_shots` is always `[]` for them, and `merge_with_audio_gaps([], audio_shots, tol)` returns `audio_shots` unchanged (the "far from any ball shot" condition is vacuously true with zero ball shots) — so every existing test's shot/rally output is bit-for-bit what it was before this task.

- [ ] **Step 4: Commit**

```bash
git add src/padelfusion/pipeline.py
git commit -m "feat: merge ball and audio shots in analyze(), record shots_source"
```

---

### Task 8: Confidence downgrade in `stats.py` when ball detection was used

**Files:**
- Modify: `src/padelfusion/stats.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Consumes: `meta["shots_source"]` (Task 7)
- Produces: `shot_count` (and the other shot-tier stats sharing `shot_tier`) drop one confidence level whenever ball detection contributed any shots.

- [ ] **Step 1: Write the failing tests**

```python
def test_shot_count_downgraded_when_ball_detection_used():
    tl = _tiny_timeline()
    tl.meta["shots_source"] = "ball"
    res = compute_stats(tl, StatsConfig())
    assert res.by_key("shot_count").confidence == "Low"


def test_shot_count_stays_medium_for_audio_only():
    tl = _tiny_timeline()
    tl.meta["shots_source"] = "audio"
    res = compute_stats(tl, StatsConfig())
    assert res.by_key("shot_count").confidence == "Medium"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "shot_count_downgraded or shot_count_stays_medium" -v`
Expected: `test_shot_count_downgraded_when_ball_detection_used` FAILS (currently always `MEDIUM` regardless of source); `test_shot_count_stays_medium_for_audio_only` PASSES already (confirm this — it's the existing default behavior, a regression guard for this task)

- [ ] **Step 3: Update the shots/rally block**

In `src/padelfusion/stats.py`, in `compute_stats`, change:

```python
    # -- shots and rallies -------------------------------------------------
    shots = tl.shots
    rally_shots = [s for s in shots if s.get("rally_id") is not None]
    ungated = sum(1 for s in rally_shots if s["gate"] == "ungated")
    shots_source = meta.get("shots_source", "audio")
    shot_tier = MEDIUM
    shot_notes = ["neighbouring courts and ball bounces can add false impacts"]
    if "ball" in shots_source:
        shot_tier = downgrade(shot_tier)
        shot_notes.append("shots detected from ball trajectory are less validated than audio impacts")
    if rally_shots and ungated / len(rally_shots) > 0.2:
        shot_tier = downgrade(shot_tier)
        shot_notes.append(f"{ungated} shots had no tracking data to confirm them")
```

(Only the `shots_source` line and the `if "ball" in shots_source:` block are new; the `ungated` check now calls `downgrade(shot_tier)` instead of the literal assignment `shot_tier = LOW` it used before — equivalent when starting from `MEDIUM`, and composes correctly when both conditions fire.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "shot_count_downgraded or shot_count_stays_medium" -v`
Expected: both PASS

Also run the full unit suite:

Run: `pytest tests/test_units.py -v`
Expected: all PASS (in particular `test_compute_stats_labels_every_stat`, which doesn't set `shots_source` on its fixture timeline — `meta.get("shots_source", "audio")` defaults to `"audio"`, preserving today's exact tiering)

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/stats.py tests/test_units.py
git commit -m "feat: downgrade shot-count confidence when ball detection was used"
```

---

### Task 9: Synthetic ball rendering and end-to-end regression

**Files:**
- Modify: `src/padelfusion/synth.py`
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Produces: `synth.generate(out_dir, spec=None, distortion_k=0.0, render_ball: bool = False)` — new optional kwarg, default preserves today's exact output (no ball drawn, matching every existing fixture).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_end_to_end.py`:

```python
@pytest.fixture(scope="module")
def synthetic_with_ball(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_ball")
    return generate(out, SynthSpec(width=1280, height=720, duration_s=45.0, seed=3), render_ball=True)


def test_pipeline_detects_shots_from_ball_trajectory(synthetic_with_ball, tmp_path):
    meta = json.loads(synthetic_with_ball["meta"].read_text())
    cfg = PipelineConfig()
    cfg.detection.backend = "background"
    res = analyze(synthetic_with_ball["video"], tmp_path, cfg, me_px=tuple(meta["me_px"]),
                  use_llm=False, skip_capture_check=True)

    assert "ball" in res.timeline.meta["shots_source"]

    checks = {c.metric: c for c in validate_dir(tmp_path, synthetic_with_ball["labels"])}
    for metric in ("rally_count_err_pct", "shot_count_err_pct", "unflagged_swaps_per_5min",
                   "stats_without_confidence"):
        assert checks[metric].passed, checks[metric]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end.py::test_pipeline_detects_shots_from_ball_trajectory -v`
Expected: FAIL with `TypeError: generate() got an unexpected keyword argument 'render_ball'`

- [ ] **Step 3: Add ball rendering to `synth.generate`**

In `src/padelfusion/synth.py`, add `import cv2` (already imported) and the HSV-derived ball color near the top (after `PLAYER_COLORS`):

```python
_BALL_HSV = np.uint8([[[35, 200, 220]]])
BALL_BGR = tuple(int(c) for c in cv2.cvtColor(_BALL_HSV, cv2.COLOR_HSV2BGR)[0, 0])
```

Add these two functions near `draw_player`:

```python
def ball_pos(t: float, rallies: list[dict]) -> np.ndarray | None:
    """Piecewise-linear path between consecutive shot positions within the
    rally covering `t`; None when no rally covers `t` (ball not in play)."""
    for r in rallies:
        shots = r["shots"]
        t0, t1 = shots[0][0], shots[-1][0]
        if t0 - 0.05 <= t <= t1 + 0.05:
            idx = next((i for i in range(len(shots) - 1) if shots[i][0] <= t <= shots[i + 1][0]),
                      len(shots) - 2) if t > t0 else 0
            ta, ha = shots[idx]
            tb, hb = shots[min(idx + 1, len(shots) - 1)]
            pa, pb = player_pos(ha, ta, rallies), player_pos(hb, tb, rallies)
            f = 0.0 if tb <= ta else float(np.clip((t - ta) / (tb - ta), 0, 1))
            return pa + f * (pb - pa)
    return None


def draw_ball(img: np.ndarray, court: CourtModel, xy: np.ndarray) -> None:
    s = px_per_m(court, xy[1])
    p = court.court_to_image([xy])[0]
    r = max(2, int(0.10 * s))
    cv2.circle(img, (int(p[0]), int(p[1])), r, BALL_BGR, -1)
```

Update `generate`'s signature and frame loop:

```python
def generate(out_dir: str | Path, spec: SynthSpec | None = None,
            distortion_k: float = 0.0, render_ball: bool = False) -> dict[str, Path]:
```

and inside the per-frame loop, after the existing player-drawing loop and before the `if maps is not None:` distortion block:

```python
            if render_ball:
                bp = ball_pos(t, rallies)
                if bp is not None:
                    draw_ball(frame, court, bp)
            if maps is not None:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_end_to_end.py::test_pipeline_detects_shots_from_ball_trajectory -v`
Expected: PASS

This test is slow (renders and encodes another ~45s synthetic video with per-frame ball position computation). Consistent with the other synthetic end-to-end tests' cost.

If the test fails on the validation checks (not the `shots_source` assertion) rather than passing outright, that is real, useful signal about the kink-detection thresholds needing adjustment against this synthetic data before considering the mechanism validated — per this plan's stated goal of proving the mechanism, not finalizing real-world thresholds. Do not loosen the synthetic test's tolerances to force a pass; instead adjust `BallConfig`'s defaults (`kink_angle_deg`, `close_to_player_m`, `max_gap_s`) since the synthetic ball path's kinks are exact and known — if detection doesn't recover them, the thresholds are the problem, not the test.

- [ ] **Step 5: Run the full suite one last time**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/padelfusion/synth.py tests/test_end_to_end.py
git commit -m "test: end-to-end regression for ball-based shot detection"
```
