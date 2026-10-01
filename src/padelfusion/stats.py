"""Stats engine: every figure is computed from the event timeline, and each one
carries a confidence tier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import StatsConfig
from .court import COURT_LENGTH, COURT_WIDTH
from .timeline import Timeline

HIGH, MEDIUM, LOW = "High", "Medium", "Low"
TIERS = (HIGH, MEDIUM, LOW)


def downgrade(tier: str) -> str:
    return TIERS[min(TIERS.index(tier) + 1, len(TIERS) - 1)]


@dataclass
class Stat:
    key: str
    label: str
    value: float | int | None
    display: str
    confidence: str
    method: str
    player: str | None = None
    unit: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.confidence not in TIERS:
            raise ValueError(f"stat {self.key} has invalid confidence {self.confidence!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StatsResult:
    stats: list[Stat]
    heatmaps: dict[str, dict]  # pid -> {"grid": 2D list (y bins x x bins), "bin_m": float}
    player_quality: dict[str, dict]

    def to_dict(self) -> dict:
        return {"stats": [s.to_dict() for s in self.stats], "heatmaps": self.heatmaps,
                "player_quality": self.player_quality}

    def by_key(self, key: str, player: str | None = None) -> Stat | None:
        return next((s for s in self.stats if s.key == key and s.player == player), None)


def _fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}"


def _runs(ts: np.ndarray, max_dt: float) -> list[slice]:
    if len(ts) == 0:
        return []
    breaks = np.flatnonzero(np.diff(ts) > max_dt) + 1
    edges = [0, *breaks.tolist(), len(ts)]
    return [slice(a, b) for a, b in zip(edges, edges[1:]) if b - a >= 2]


def distance_covered(pts: list[dict], cfg: StatsConfig, sample_dt: float) -> float:
    """Smoothed path length. Smoothing damps tracking jitter, which inflates
    raw distance. Steps across gaps or above a plausible speed are skipped."""
    if len(pts) < 2:
        return 0.0
    ts = np.array([p["t"] for p in pts])
    xy = np.array([[p["x"], p["y"]] for p in pts])
    k = max(1, int(round(cfg.distance_smooth_s / max(sample_dt, 1e-3))))
    total = 0.0
    for sl in _runs(ts, 1.0):
        seg, t = xy[sl], ts[sl]
        if k > 1 and len(seg) > k:
            kernel = np.ones(k) / k
            pad = k // 2
            padded = np.pad(seg, ((pad, k - 1 - pad), (0, 0)), mode="edge")
            seg = np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)])
        steps = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        speed = steps / np.maximum(np.diff(t), 1e-3)
        total += float(steps[speed <= cfg.max_plausible_speed_mps].sum())
    return total


def heatmap(pts: list[dict], bin_m: float) -> list[list[float]]:
    nx, ny = int(round(COURT_WIDTH / bin_m)), int(round(COURT_LENGTH / bin_m))
    if not pts:
        return np.zeros((ny, nx)).tolist()
    xs = np.clip([p["x"] for p in pts], 0, COURT_WIDTH - 1e-6)
    ys = np.clip([p["y"] for p in pts], 0, COURT_LENGTH - 1e-6)
    h, _, _ = np.histogram2d(ys, xs, bins=[ny, nx], range=[[0, COURT_LENGTH], [0, COURT_WIDTH]])
    return (h / h.sum()).round(5).tolist()


def compute_stats(tl: Timeline, cfg: StatsConfig) -> StatsResult:
    meta = tl.meta
    duration = float(meta["duration_s"])
    sample_dt = 1.0 / float(meta["analysis_fps"])
    n_samples = max(1, int(meta.get("n_samples", round(duration / sample_dt))))
    court = meta.get("court", {})
    court_weak = court.get("source") == "auto" and (court.get("score") or 0) < 0.7

    stats: list[Stat] = []
    heatmaps: dict[str, dict] = {}
    quality: dict[str, dict] = {}
    by_player = tl.positions_by_player()

    for pid in sorted(tl.players):
        pts = by_player.get(pid, [])
        observed = [p for p in pts if not p["interpolated"]]
        coverage = len(observed) / n_samples
        flagged_pts = sum(1 for p in pts if any(f != "interpolated" for f in p["flags"]))
        lost_s = sum(s["end"] - s["start"] for s in tl.flagged_segments
                     if s["player"] == pid and s["reason"] == "lost")
        flagged_frac = min(1.0, flagged_pts / n_samples + lost_s / max(duration, 1e-6))
        quality[pid] = {"coverage": round(coverage, 4), "flagged_frac": round(flagged_frac, 4)}

        notes: list[str] = []
        pos_penalty = 0
        if flagged_frac > cfg.flag_downgrade_frac:
            pos_penalty += 1
            notes.append(f"{flagged_frac:.0%} of this track is flagged (ID ambiguity or lost)")
        if coverage < 0.7:
            pos_penalty += 1
            notes.append(f"player detected in only {coverage:.0%} of sampled frames")
        if court_weak:
            pos_penalty += 1
            notes.append("court mapping auto-detected with weak line support")

        def tier(base: str) -> str:
            t = base
            for _ in range(pos_penalty):
                t = downgrade(t)
            return t

        heatmaps[pid] = {"grid": heatmap(pts, cfg.heatmap_bin_m), "bin_m": cfg.heatmap_bin_m}
        stats.append(Stat("heatmap", "Position heatmap", round(coverage * 100, 1),
                          f"tracked in {coverage * 100:.0f}% of frames", tier(HIGH),
                          "player detection + tracking mapped to court coordinates", pid, "%",
                          list(notes)))
        if pts:
            net = sum(1 for p in pts if p["zone"] == "net") / len(pts) * 100
            stats.append(Stat("net_time_pct", "Net vs. back time", round(net, 1),
                              f"{net:.0f}% net / {100 - net:.0f}% back", tier(HIGH),
                              "share of tracked samples within the net zone", pid, "%", list(notes)))
        dist = distance_covered(pts, cfg, sample_dt)
        stats.append(Stat("distance_m", "Distance covered", round(dist),
                          f"{dist:.0f} m", tier(MEDIUM),
                          "smoothed foot-position path length; tracking jitter inflates it", pid, "m",
                          list(notes)))

    # -- partner distance ---------------------------------------------------
    teams: dict[str, list[str]] = {}
    for pid, info in tl.players.items():
        teams.setdefault(info["team"], []).append(pid)
    for pids in teams.values():
        if len(pids) != 2:
            continue
        p1, p2 = sorted(pids)
        pts1, pts2 = by_player.get(p1, []), by_player.get(p2, [])
        if not pts1 or not pts2:
            continue
        t2 = np.array([p["t"] for p in pts2])
        xy2 = np.array([[p["x"], p["y"]] for p in pts2])
        dists = []
        for p in pts1:
            i = int(np.argmin(np.abs(t2 - p["t"])))
            if abs(t2[i] - p["t"]) <= 0.25:
                dists.append(float(np.hypot(p["x"] - xy2[i, 0], p["y"] - xy2[i, 1])))
        if not dists:
            continue
        mean_d = float(np.mean(dists))
        for pid in (p1, p2):
            base = next((st for st in stats if st.key == "distance_m" and st.player == pid), None)
            conf = base.confidence if base else MEDIUM
            stats.append(Stat("partner_distance_m", "Distance from partner (mean)", round(mean_d, 1),
                              f"{mean_d:.1f} m", conf,
                              "mean Euclidean distance to the nearest-time partner sample", pid, "m",
                              ["distance between the two players on the same team"]))

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
    if tl.isolated_shots:
        shot_notes.append(f"{len(tl.isolated_shots)} isolated impacts outside any rally were not counted")
    stats.append(Stat("shot_count", "Total shots", len(rally_shots), str(len(rally_shots)), shot_tier,
                      "audio impact onsets confirmed by player motion", None, "shots", shot_notes))

    rallies = tl.rallies
    durs = np.array([r["duration"] for r in rallies]) if rallies else np.array([])
    min_shots = meta.get("config", {}).get("rallies", {}).get("min_shots", 2)
    rally_notes = [f"a rally needs at least {min_shots} confirmed shots"]
    stats.append(Stat("rally_count", "Rallies", len(rallies), str(len(rallies)), shot_tier,
                      "audio onsets + player motion, split on shot gaps and resets", None, "rallies",
                      rally_notes))
    if len(durs):
        stats.append(Stat("rally_mean_s", "Mean rally duration", round(float(durs.mean()), 1),
                          f"{durs.mean():.1f} s", shot_tier, "first to last shot of each rally", None, "s"))
        stats.append(Stat("rally_median_s", "Median rally duration", round(float(np.median(durs)), 1),
                          f"{np.median(durs):.1f} s", shot_tier, "first to last shot of each rally", None, "s"))
        stats.append(Stat("rally_longest_s", "Longest rally", round(float(durs.max()), 1),
                          f"{durs.max():.1f} s", shot_tier, "first to last shot of each rally", None, "s"))
        spr = np.mean([r["n_shots"] for r in rallies])
        stats.append(Stat("shots_per_rally", "Shots per rally (mean)", round(float(spr), 1),
                          f"{spr:.1f}", shot_tier, "rally shot counts", None, "shots"))

    # -- stretch goal: shot attribution --------------------------------------
    for pid in sorted(tl.players):
        n = sum(1 for s in rally_shots if s.get("attributed_player") == pid)
        stats.append(Stat("shots_attributed", "Shots attributed", n, str(n), LOW,
                          "impact time + most active player (no pose model yet)", pid, "shots",
                          ["experimental: attribution is not validated"]))

    # -- positions at shots (matches the ground-truth protocol) -----------
    for pid in sorted(tl.players):
        pts = by_player.get(pid, [])
        if not pts or not rally_shots:
            continue
        ts = np.array([p["t"] for p in pts])
        zones = []
        for s in rally_shots:
            i = int(np.argmin(np.abs(ts - s["t"])))
            if abs(ts[i] - s["t"]) <= 0.25:
                zones.append(pts[i]["zone"])
        if zones:
            v = sum(z == "net" for z in zones) / len(zones) * 100
            base = next(st for st in stats if st.key == "net_time_pct" and st.player == pid)
            stats.append(Stat("net_at_shots_pct", "At net when a shot is struck", round(v, 1),
                              f"{v:.0f}%", base.confidence,
                              "player zone at each shot timestamp", pid, "%", list(base.notes)))

    return StatsResult(stats, heatmaps, quality)
