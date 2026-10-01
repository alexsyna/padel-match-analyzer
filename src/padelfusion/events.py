"""Shot fusion (audio onsets x player motion) and rally segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .audio import Onset
from .config import RallyConfig, ShotConfig
from .tracking import TrackPoint


@dataclass
class Shot:
    t: float
    strength: float
    level_db: float
    gate: str  # "motion" (confirmed by player motion) | "ungated" (no tracking data at that time)
    active_player: str | None  # most active player around the impact
    activity: float
    attributed_player: str | None = None  # stretch goal, LOW confidence
    rally_id: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("t", "strength", "level_db", "activity"):
            d[k] = round(d[k], 4)
        return d


@dataclass
class Rally:
    rally_id: int
    start: float  # first shot
    end: float  # last shot
    n_shots: int
    clip_start: float
    clip_end: float
    shot_times: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d


def player_activity(points: list[TrackPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample (times, speed m/s, relative bbox change per sample)."""
    if len(points) < 2:
        return np.array([]), np.array([]), np.array([])
    t = np.array([p.t for p in points])
    xy = np.array([[p.x, p.y] for p in points])
    wh = np.array([[p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]] for p in points])
    dt = np.maximum(np.diff(t), 1e-3)
    speed = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt])
    rel = np.abs(np.diff(wh, axis=0)) / np.maximum(wh[:-1], 1.0)
    change = np.concatenate([[0.0], rel.max(axis=1)])
    # Drop steps that bridge a tracking gap.
    gap = np.concatenate([[False], np.diff(t) > 0.5])
    speed[gap] = 0.0
    change[gap] = 0.0
    return t, speed, change


def fuse_shots(onsets: list[Onset], tracks: dict[str, list[TrackPoint]],
               cfg: ShotConfig) -> tuple[list[Shot], list[dict]]:
    """Keep onsets that coincide with player motion. Return (shots, rejected)."""
    acts = {pid: player_activity(pts) for pid, pts in tracks.items()}
    shots, rejected = [], []
    for o in onsets:
        lo, hi = o.t - cfg.motion_window_before_s, o.t + cfg.motion_window_after_s
        best_pid, best_score, any_data = None, 0.0, False
        for pid, (t, speed, change) in acts.items():
            m = (t >= lo) & (t <= hi)
            if not m.any():
                continue
            any_data = True
            swing = float(change[m].max()) / cfg.min_bbox_change
            move = float(speed[m].max()) / cfg.min_player_speed_mps
            score = swing if cfg.require_swing else max(swing, move)
            if score > best_score:
                best_pid, best_score = pid, score
        if not any_data:
            shots.append(Shot(o.t, o.strength, o.level_db, "ungated", None, 0.0))
        elif best_score >= 1.0:
            shots.append(Shot(o.t, o.strength, o.level_db, "motion", best_pid, best_score,
                              attributed_player=best_pid))
        else:
            rejected.append({**o.to_dict(), "reason": "no_player_motion"})
    return shots, rejected


def _all_still(tracks, t0: float, t1: float, cfg: RallyConfig) -> bool:
    """True if every tracked player moved slower than the reset speed for the whole window."""
    for pts in tracks.values():
        seg = [p for p in pts if t0 <= p.t <= t1]
        if len(seg) < 2:
            return False
        t = np.array([p.t for p in seg])
        xy = np.array([[p.x, p.y] for p in seg])
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)
        k = max(1, int(round(0.5 / max(np.median(np.diff(t)), 1e-3))))
        smooth = np.convolve(speed, np.ones(k) / k, mode="same")
        if smooth.max() > cfg.reset_speed_mps:
            return False
    return True


def segment_rallies(shots: list[Shot], tracks: dict[str, list[TrackPoint]],
                    cfg: RallyConfig) -> tuple[list[Rally], list[float]]:
    """Split shots into rallies by shot gaps and player resets.

    Returns (rallies, isolated shot times dropped for having < min_shots).
    """
    shots = sorted(shots, key=lambda s: s.t)
    groups: list[list[Shot]] = []
    for s in shots:
        if groups:
            prev = groups[-1][-1]
            gap = s.t - prev.t
            reset = gap >= cfg.reset_still_s and _all_still(
                tracks, prev.t + 0.5, prev.t + 0.5 + cfg.reset_still_s, cfg)
            if gap <= cfg.max_shot_gap_s and not reset:
                groups[-1].append(s)
                continue
        groups.append([s])

    rallies, isolated = [], []
    for g in groups:
        if len(g) < cfg.min_shots:
            isolated.extend(s.t for s in g)
            for s in g:
                s.rally_id = None
            continue
        rid = len(rallies) + 1
        for s in g:
            s.rally_id = rid
        rallies.append(Rally(rid, g[0].t, g[-1].t, len(g),
                             max(0.0, g[0].t - cfg.clip_pad_before_s), g[-1].t + cfg.clip_pad_after_s,
                             [round(s.t, 3) for s in g]))
    return rallies, isolated
