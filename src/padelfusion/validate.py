"""Compare a pipeline timeline with a hand-labelled ground truth.

Label file (JSON):

    {
      "window": {"start": 0.0, "end": 600.0},          # optional; defaults to the whole clip
      "rallies": [{"start": 12.3, "end": 19.8}, ...],   # first and last shot of each rally
      "shots": [12.3, 13.4, ...],                       # every shot timestamp (s)
      "positions_at_shots": [                           # per shot, per player
        {"t": 12.3, "player": "P1", "zone": "net", "side": "left"}, ...
      ]
    }

"zone" is "net" or "back" (front or back half of the player's side). The
optional "side" ("left"/"right", seen from the camera) is used to count
partner ID swaps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .timeline import Timeline

THRESHOLDS = {
    "rally_count_err_pct": 5.0,
    "shot_count_err_pct": 10.0,
    "net_share_err_pp": 5.0,
    "unflagged_swaps_per_5min": 1.0,
    "stats_without_confidence": 0,
}


@dataclass
class Check:
    metric: str
    value: float | None
    threshold: float
    passed: bool | None
    detail: str = ""


def _in_window(t: float, w) -> bool:
    return w[0] <= t <= w[1]


def match_events(pred: list[float], truth: list[float], tol: float) -> int:
    """Greedy one-to-one matching of timestamps within `tol`. Returns matches."""
    pred, truth = sorted(pred), sorted(truth)
    i = j = hits = 0
    while i < len(pred) and j < len(truth):
        d = pred[i] - truth[j]
        if abs(d) <= tol:
            hits += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return hits


def _flagged(tl: Timeline, pid: str, t: float) -> bool:
    return any(s["player"] == pid and s["start"] - 0.25 <= t <= s["end"] + 0.25 for s in tl.flagged_segments)


def validate(tl: Timeline, labels: dict, stats: dict | None = None, shot_tol_s: float = 0.2) -> list[Check]:
    w = labels.get("window") or {}
    win = (float(w.get("start", 0.0)), float(w.get("end", tl.meta["duration_s"])))
    minutes = (win[1] - win[0]) / 60
    checks: list[Check] = []

    pred_rallies = [r for r in tl.rallies if _in_window(r["start"], win)]
    true_rallies = [r for r in labels.get("rallies", []) if _in_window(r["start"], win)]
    if true_rallies:
        err = abs(len(pred_rallies) - len(true_rallies)) / len(true_rallies) * 100
        checks.append(Check("rally_count_err_pct", round(err, 2), THRESHOLDS["rally_count_err_pct"],
                            err <= THRESHOLDS["rally_count_err_pct"],
                            f"pipeline {len(pred_rallies)} vs hand {len(true_rallies)}"))

    pred_shots = [s["t"] for s in tl.shots if s.get("rally_id") is not None and _in_window(s["t"], win)]
    true_shots = [t for t in labels.get("shots", []) if _in_window(t, win)]
    if true_shots:
        err = abs(len(pred_shots) - len(true_shots)) / len(true_shots) * 100
        hits = match_events(pred_shots, true_shots, shot_tol_s)
        prec = hits / len(pred_shots) if pred_shots else 0.0
        rec = hits / len(true_shots)
        checks.append(Check("shot_count_err_pct", round(err, 2), THRESHOLDS["shot_count_err_pct"],
                            err <= THRESHOLDS["shot_count_err_pct"],
                            f"pipeline {len(pred_shots)} vs hand {len(true_shots)}; "
                            f"timing precision {prec:.2f}, recall {rec:.2f} (±{shot_tol_s}s)"))

    labs = [p for p in labels.get("positions_at_shots", []) if _in_window(p["t"], win)]
    if labs:
        by_player = tl.positions_by_player()
        worst, details = 0.0, []
        swaps = 0
        for pid in sorted({p["player"] for p in labs}):
            pts = by_player.get(pid, [])
            if not pts:
                continue
            ts = np.array([p["t"] for p in pts])
            pl = [p for p in labs if p["player"] == pid]
            pred_net = true_net = n = 0
            for lab in pl:
                i = int(np.argmin(np.abs(ts - lab["t"])))
                if abs(ts[i] - lab["t"]) > 0.25:
                    continue
                n += 1
                pred_net += pts[i]["zone"] == "net"
                true_net += lab["zone"] == "net"
            if n:
                diff = abs(pred_net - true_net) / n * 100
                worst = max(worst, diff)
                details.append(f"{pid} {pred_net / n * 100:.0f}% vs {true_net / n * 100:.0f}%")
        checks.append(Check("net_share_err_pp", round(worst, 2), THRESHOLDS["net_share_err_pp"],
                            worst <= THRESHOLDS["net_share_err_pp"], "; ".join(details)))

        # Partner swaps: compare left/right order of partners with the labels.
        side_labs = [p for p in labs if p.get("side") in ("left", "right")]
        if side_labs:
            teams: dict[str, list[str]] = {}
            for pid, info in tl.players.items():
                teams.setdefault(info["team"], []).append(pid)
            for team, pids in teams.items():
                if len(pids) != 2:
                    continue
                a, b = sorted(pids)
                prev_ok = True
                for t in sorted({p["t"] for p in side_labs if p["player"] == a}):
                    lab_a = next(p["side"] for p in side_labs if p["player"] == a and p["t"] == t)
                    xa = _x_at(by_player.get(a, []), t)
                    xb = _x_at(by_player.get(b, []), t)
                    if xa is None or xb is None:
                        continue
                    pred_a = "left" if xa < xb else "right"
                    ok = pred_a == lab_a
                    if prev_ok and not ok and not (_flagged(tl, a, t) or _flagged(tl, b, t)):
                        swaps += 1
                    prev_ok = ok
            per5 = swaps / max(minutes / 5, 1e-6)
            checks.append(Check("unflagged_swaps_per_5min", round(per5, 2),
                                THRESHOLDS["unflagged_swaps_per_5min"],
                                per5 <= THRESHOLDS["unflagged_swaps_per_5min"], f"{swaps} unflagged swaps"))

    if stats is not None:
        missing = sum(1 for s in stats["stats"] if s.get("confidence") not in ("High", "Medium", "Low"))
        checks.append(Check("stats_without_confidence", missing, 0, missing == 0))
    return checks


def _x_at(pts: list[dict], t: float) -> float | None:
    if not pts:
        return None
    ts = np.array([p["t"] for p in pts])
    i = int(np.argmin(np.abs(ts - t)))
    return pts[i]["x"] if abs(ts[i] - t) <= 0.25 else None


def validate_dir(run_dir: str | Path, labels_path: str | Path) -> list[Check]:
    run_dir = Path(run_dir)
    tl = Timeline.load(run_dir)
    labels = json.loads(Path(labels_path).read_text())
    stats_path = run_dir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    return validate(tl, labels, stats)


def format_checks(checks: list[Check]) -> str:
    lines = [f"{'metric':<28}{'value':>10}{'threshold':>11}  result  detail"]
    for c in checks:
        res = "PASS" if c.passed else "FAIL"
        lines.append(f"{c.metric:<28}{c.value!s:>10}{c.threshold!s:>11}  {res:<6}  {c.detail}")
    return "\n".join(lines)
