"""Four-player tracker with a team-half constraint.

Each team stays on its own side of the net. A detection on the near half can
only ever be assigned to a near-team track, so an identity swap is only
possible between partners. The tracker doesn't guess through those cases:
ambiguous frames are flagged and end up as low-confidence segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackingConfig
from .court import CourtModel, in_court, team_side
from .detection import Detection

BIG = 1e6


@dataclass
class TrackPoint:
    t: float
    x: float
    y: float
    bbox: tuple[float, float, float, float]
    flags: list[str] = field(default_factory=list)
    interpolated: bool = False


@dataclass
class Track:
    pid: str
    team: str  # near | far
    is_user: bool
    hist: np.ndarray
    points: list[TrackPoint] = field(default_factory=list)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    def predict(self, t: float) -> np.ndarray:
        dt = min(t - self.last.t, 0.5)
        return np.array([self.last.x, self.last.y]) + self.vel * dt


def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    bc = float(np.sum(np.sqrt(np.clip(h1, 0, None) * np.clip(h2, 0, None))))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


@dataclass
class CourtDetection:
    det: Detection
    xy: np.ndarray  # feet in court metres


def to_court_detections(dets: list[Detection], court: CourtModel, margin: float) -> list[CourtDetection]:
    """Map feet to court coordinates and drop anything outside the court.

    Reflections in the glass map outside the court area and are discarded here.
    """
    if not dets:
        return []
    xy = court.image_to_court([d.foot for d in dets])
    return [CourtDetection(d, p) for d, p in zip(dets, xy) if in_court(p[0], p[1], margin)]


def _side_candidates(cdets: list[CourtDetection], side: str) -> list[CourtDetection]:
    cands = [c for c in cdets if team_side(c.xy[1]) == side]
    return sorted(cands, key=lambda c: (c.det.score, c.det.size[1]), reverse=True)[:2]


class Tracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.tracks: dict[str, Track] = {}
        self.sample_times: list[float] = []
        self.my_team: str | None = None
        self.pending: dict[str, list[dict]] = {"near": [], "far": []}

    # -- initialisation ---------------------------------------------------
    def initialize(self, t: float, cdets: list[CourtDetection], me_px: tuple[float, float] | None) -> bool:
        near = _side_candidates(cdets, "near")
        far = _side_candidates(cdets, "far")
        all_c = near + far
        if not all_c:
            return False
        if me_px is not None:
            def dist_to_me(c: CourtDetection) -> float:
                x1, y1, x2, y2 = c.det.bbox
                if x1 <= me_px[0] <= x2 and y1 <= me_px[1] <= y2:
                    return 0.0
                return float(np.hypot(c.det.foot[0] - me_px[0], c.det.foot[1] - me_px[1]))

            me = min(all_c, key=dist_to_me)
        else:
            if not near:
                return False
            me = min(near, key=lambda c: c.xy[0])  # default: near-left player
        my_team = team_side(me.xy[1])
        self.my_team = my_team
        opp_team = "far" if my_team == "near" else "near"
        sides = {"near": near, "far": far}
        partner = next((c for c in sides[my_team] if c is not me), None)
        opps = sorted(sides[opp_team], key=lambda c: c.xy[0])
        assignments = [("P1", me, my_team, True)]
        if partner is not None:
            assignments.append(("P2", partner, my_team, False))
        if len(opps) >= 1:
            assignments.append(("P3", opps[0], opp_team, False))
        if len(opps) >= 2:
            assignments.append(("P4", opps[1], opp_team, False))
        for pid, c, team, is_user in assignments:
            tr = Track(pid, team, is_user, c.det.hist.copy())
            tr.points.append(TrackPoint(t, float(c.xy[0]), float(c.xy[1]), c.det.bbox))
            self.tracks[pid] = tr
        self.sample_times.append(t)
        return True

    # -- per-frame update -------------------------------------------------
    def update(self, t: float, cdets: list[CourtDetection]) -> None:
        self.sample_times.append(t)
        for team in ("near", "far"):
            tracks = [tr for tr in self.tracks.values() if tr.team == team]
            tds = [c for c in cdets if team_side(c.xy[1]) == team]
            unmatched = self._assign(t, tracks, tds)
            if len(tracks) < 2:
                self._update_candidates(t, team, unmatched, len(tracks))

    def _next_pid(self, team: str) -> str | None:
        candidates = ("P1", "P2") if team == self.my_team else ("P3", "P4")
        for pid in candidates:
            if pid not in self.tracks:
                return pid
        return None

    def _update_candidates(self, t: float, team: str, unmatched: list[CourtDetection], n_tracks: int) -> None:
        pend = self.pending[team]
        matched_idx: set[int] = set()
        for cand in pend:
            best_i, best_d = None, self.cfg.gate_margin_m
            for i, c in enumerate(unmatched):
                if i in matched_idx:
                    continue
                d = float(np.linalg.norm(cand["xy"] - c.xy))
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                matched_idx.add(best_i)
                cand["xy"] = unmatched[best_i].xy
                cand["det"] = unmatched[best_i].det
                cand["streak"] += 1
            else:
                cand["streak"] = 0
        self.pending[team] = [c for c in pend if c["streak"] > 0]
        for i, c in enumerate(unmatched):
            if i not in matched_idx:
                self.pending[team].append({"xy": c.xy, "det": c.det, "streak": 1})

        for cand in list(self.pending[team]):
            if n_tracks >= 2:
                break
            if cand["streak"] >= self.cfg.late_bind_confirm_frames:
                pid = self._next_pid(team)
                if pid is None:
                    continue
                tr = Track(pid, team, False, cand["det"].hist.copy())
                tr.points.append(TrackPoint(t, float(cand["xy"][0]), float(cand["xy"][1]), cand["det"].bbox))
                self.tracks[pid] = tr
                self.pending[team].remove(cand)
                n_tracks += 1

    def _cost(self, tr: Track, c: CourtDetection, t: float) -> float:
        pred = tr.predict(t)
        d = float(np.linalg.norm(pred - c.xy))
        gate = self.cfg.max_speed_mps * (t - tr.last.t) + self.cfg.gate_margin_m
        if d > gate:
            return BIG
        return d + self.cfg.appearance_weight * bhattacharyya(tr.hist, c.det.hist)

    def _assign(self, t: float, tracks: list[Track], tds: list[CourtDetection]) -> list[CourtDetection]:
        if not tracks or not tds:
            return list(tds)
        cost = np.array([[self._cost(tr, c, t) for c in tds] for tr in tracks])
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < BIG]
        flags: dict[int, list[str]] = {r: [] for r, _ in matches}

        if len(matches) == 2:
            (r0, c0), (r1, c1) = matches
            best = cost[r0, c0] + cost[r1, c1]
            alt = cost[r0, c1] + cost[r1, c0]
            if alt < BIG and alt - best < self.cfg.ambiguity_margin:
                flags[r0].append("ambiguous_id")
                flags[r1].append("ambiguous_id")
            if np.linalg.norm(tds[c0].xy - tds[c1].xy) < self.cfg.close_partner_m:
                flags[r0].append("close_partners")
                flags[r1].append("close_partners")
        elif len(matches) == 1 and len(tracks) == 2:
            r, c = matches[0]
            other = tracks[1 - r]
            if np.linalg.norm(other.predict(t) - tds[c].xy) < 1.5 * self.cfg.close_partner_m:
                flags[r].append("merged_detection")

        for r, c in matches:
            tr, cd = tracks[r], tds[c]
            dt = max(t - tr.last.t, 1e-3)
            new_vel = (cd.xy - np.array([tr.last.x, tr.last.y])) / dt
            tr.vel = 0.6 * tr.vel + 0.4 * new_vel if dt < 0.5 else np.zeros(2)
            tr.points.append(TrackPoint(t, float(cd.xy[0]), float(cd.xy[1]), cd.det.bbox, flags[r]))
            if not flags[r]:
                tr.hist = 0.9 * tr.hist + 0.1 * cd.det.hist

        matched_cols = {c for _, c in matches}
        return [tds[c] for c in range(len(tds)) if c not in matched_cols]

    # -- post-processing --------------------------------------------------
    def finalize(self) -> tuple[dict[str, list[TrackPoint]], list[dict]]:
        """Interpolate short gaps and return (points per player, flagged segments)."""
        out: dict[str, list[TrackPoint]] = {}
        segments: list[dict] = []
        times = np.array(self.sample_times)
        for pid, tr in self.tracks.items():
            pts: list[TrackPoint] = []
            for a, b in zip(tr.points, tr.points[1:] + [None]):
                pts.append(a)
                if b is None:
                    continue
                gap = b.t - a.t
                inner = times[(times > a.t + 1e-9) & (times < b.t - 1e-9)]
                if len(inner) == 0:
                    continue
                if gap <= self.cfg.max_interp_gap_s:
                    for ti in inner:
                        f = (ti - a.t) / gap
                        bbox = tuple(av + f * (bv - av) for av, bv in zip(a.bbox, b.bbox))
                        pts.append(TrackPoint(float(ti), a.x + f * (b.x - a.x), a.y + f * (b.y - a.y),
                                              bbox, ["interpolated"], True))
                else:
                    segments.append({"player": pid, "start": round(a.t, 3), "end": round(b.t, 3),
                                     "reason": "lost"})
            if len(times) and tr.points and times[-1] - tr.points[-1].t > self.cfg.max_interp_gap_s:
                segments.append({"player": pid, "start": round(tr.points[-1].t, 3),
                                 "end": round(float(times[-1]), 3), "reason": "lost"})
            out[pid] = pts
            segments.extend(flag_segments(pid, pts))
        segments.sort(key=lambda s: (s["start"], s["player"]))
        return out, segments


def flag_segments(pid: str, pts: list[TrackPoint], join_gap_s: float = 0.5) -> list[dict]:
    """Merge consecutive flagged points (excluding plain interpolation) into segments."""
    segs: list[dict] = []
    cur = None
    for p in pts:
        reasons = [f for f in p.flags if f != "interpolated"]
        if not reasons:
            continue
        if cur and p.t - cur["end"] <= join_gap_s:
            cur["end"] = round(p.t, 3)
            cur["_reasons"].update(reasons)
        else:
            if cur:
                segs.append(cur)
            cur = {"player": pid, "start": round(p.t, 3), "end": round(p.t, 3), "_reasons": set(reasons)}
    if cur:
        segs.append(cur)
    for s in segs:
        s["reason"] = ",".join(sorted(s.pop("_reasons")))
    return segs
