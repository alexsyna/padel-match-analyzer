"""Ball-based shot detection: color-threshold candidate detection, a
gap-tolerant single-object trajectory tracker, and hit detection from sharp
trajectory direction changes near a tracked player.

Everything here operates in pixel space, not court coordinates. Players'
court positions come from their *feet* — a point actually touching the
ground, so the ground-plane homography is valid for them. The ball is
usually airborne mid-rally; projecting an elevated pixel through that same
homography produces a wildly wrong "ground" position (confirmed directly
against real footage: static background clutter like signage, which stays
at a fixed pixel position, was landing tens of metres outside the actual
court in the naive court-space version of this module). There is no way to
recover the ball's true ground position from a single camera, so trajectory
linking, the speed gate, and the "near a player" check are all done in pixel
space and pixel/bbox terms instead.

Shot detection here is the primary signal (per the user's real footage
having no audio at all); `pipeline.analyze` merges this module's output
with audio-detected shots, using audio only to fill gaps this module missed.

Real-footage validation (2026-09-30/10-01, indoor court with reflective
glass, overhead lighting, and a static yellow-green water bottle on a bench
at the court's edge, hand-labeled ground truth): the pixel-space rewrite
fixed a real bug (ball candidates were being run through the ground-plane
homography like players, producing nonsense positions for an airborne
object). A second real bug was found the same way — visually, from the
live detection overlay (see `web/index.html`'s "Show detected players and
ball" toggle) — showing a persistent false "ball" locked onto that static
bottle for the whole clip: 97.8% of all raw candidates were that one static
object, not the ball. `static_ball_mask` fixes this by excluding any
ball-colored region that's also present in a temporal median background
(the same idea `detection.BackgroundDetector` already uses, applied here to
ball candidates specifically). Combined, recall went from ~4% to ~30% and
the underlying candidate data dropped from ~1300 points (mostly noise) to
~290 (mostly signal) on the same clip. Still far from reliable — color-
threshold detection alone still picks up real noise on this kind of footage
even after these two fixes. Treat this module's output as a genuinely
experimental, best-effort signal, not a validated one; the "Low confidence"
tiering this feeds into (stats.py) reflects that. Further recall
improvement would likely need a better candidate detector (e.g.
motion-consistency across frames, not color alone), not just more threshold
tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import BallConfig
from .events import Shot
from .tracking import TrackPoint


def static_ball_mask(sample_frames: list[np.ndarray], cfg: BallConfig) -> np.ndarray:
    """A mask of ball-colored regions present in a temporal median background
    (a water bottle, branded gear, anything else that just sits there for
    the whole clip) — a persistent object can't be the ball, confirmed
    directly against real footage where a single static bottle accounted
    for ~98% of all raw candidates before this filter existed."""
    bg = np.median(np.stack(sample_frames), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))
    return cv2.dilate(mask, np.ones((9, 9), np.uint8))


def detect_ball_candidates(frame: np.ndarray, court_mask: np.ndarray | None,
                           player_bboxes: list[tuple[float, float, float, float]],
                           cfg: BallConfig, static_mask: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Ball-colored blob centroids (pixel coordinates), excluding anything
    inside a player's bounding box (kills same-colored shoes/gear) or inside
    `static_mask` (kills persistent same-colored background objects)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))
    if court_mask is not None:
        mask = cv2.bitwise_and(mask, court_mask)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    m = cfg.player_exclude_margin_px
    h_mask, w_mask = mask.shape
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
        if static_mask is not None:
            iy, ix = min(int(cy), h_mask - 1), min(int(cx), w_mask - 1)
            if static_mask[iy, ix]:
                continue
        cands.append((float(cx), float(cy)))
    return cands


@dataclass
class BallPoint:
    t: float
    x: float  # pixel coordinates
    y: float


class BallTracker:
    """Single-object, gap-tolerant tracker in pixel space."""

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
        gate = self.cfg.max_speed_px_s * dt
        best = min(candidates_xy, key=lambda c: np.hypot(c[0] - last.x, c[1] - last.y))
        d = float(np.hypot(best[0] - last.x, best[1] - last.y))
        if d <= gate:
            self.points.append(BallPoint(t, best[0], best[1]))


def _dist_to_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> float:
    """0 if (x, y) is inside bbox, else the distance to its nearest edge."""
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def detect_ball_shots(points: list[BallPoint], player_tracks: dict[str, list[TrackPoint]],
                      cfg: BallConfig) -> list[Shot]:
    """A shot is a trajectory point where the incoming/outgoing direction
    changes sharply (a racket hit, unlike gravity-curved flight) and some
    tracked player's bounding box is nearby at that instant (excludes
    floor/glass bounces away from play). The angle check is scale-invariant
    so it's unaffected by working in pixel space; the proximity check uses
    each player's pixel bbox directly rather than a ground-plane distance,
    since the ball's own pixel position has no valid ground-plane
    equivalent while airborne."""
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
        best_pid, best_d = None, cfg.close_to_player_px
        for pid, pts in player_tracks.items():
            near = min(pts, key=lambda p: abs(p.t - b.t), default=None)
            if near is None or abs(near.t - b.t) > 0.3:
                continue
            d = _dist_to_bbox(b.x, b.y, near.bbox)
            if d < best_d:
                best_pid, best_d = pid, d
        if best_pid is None:
            continue
        shots.append(Shot(b.t, angle, 0.0, "ball", best_pid, 1.0, attributed_player=best_pid))
        last_t = b.t
    return shots


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
