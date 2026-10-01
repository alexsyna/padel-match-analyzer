"""The event timeline: the one data layer every stat is computed from.

It is also the export format (JSON + CSV) used for validation. v1 ball
tracking and v2 sensor data will extend it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = "padelfusion.timeline/0.1"


@dataclass
class Timeline:
    meta: dict
    players: dict[str, dict]  # pid -> {team, is_user}
    positions: list[dict]  # {t, player, x, y, zone, team, flags, interpolated, bbox_w, bbox_h, bbox}
    shots: list[dict]
    rallies: list[dict]
    flagged_segments: list[dict]
    rejected_onsets: list[dict] = field(default_factory=list)
    isolated_shots: list[float] = field(default_factory=list)
    ball_positions: list[dict] = field(default_factory=list)  # {t, x, y} in pixel space

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "meta": self.meta,
            "players": self.players,
            "rallies": self.rallies,
            "shots": self.shots,
            "flagged_segments": self.flagged_segments,
            "rejected_onsets": self.rejected_onsets,
            "isolated_shots": self.isolated_shots,
            "positions": self.positions,
            "ball_positions": self.ball_positions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        if d.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported timeline schema: {d.get('schema')}")
        return cls(d["meta"], d["players"], d["positions"], d["shots"], d["rallies"],
                   d["flagged_segments"], d.get("rejected_onsets", []), d.get("isolated_shots", []),
                   d.get("ball_positions", []))

    # -- IO -----------------------------------------------------------------
    def save(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"json": out / "timeline.json"}
        with open(paths["json"], "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        paths["positions_csv"] = _write_csv(
            out / "positions.csv", self.positions,
            ["t", "player", "team", "x", "y", "zone", "interpolated", "flags", "bbox_w", "bbox_h"])
        paths["shots_csv"] = _write_csv(
            out / "shots.csv", self.shots,
            ["t", "rally_id", "strength", "level_db", "gate", "active_player", "activity", "attributed_player"])
        paths["rallies_csv"] = _write_csv(
            out / "rallies.csv", self.rallies,
            ["rally_id", "start", "end", "duration", "n_shots", "clip_start", "clip_end"])
        paths["flags_csv"] = _write_csv(out / "flagged_segments.csv", self.flagged_segments,
                                        ["player", "start", "end", "reason"])
        paths["events_csv"] = _write_csv(out / "events.csv", self.event_rows(),
                                         ["t", "event", "player", "rally_id", "detail"])
        return paths

    @classmethod
    def load(cls, path: str | Path) -> "Timeline":
        p = Path(path)
        if p.is_dir():
            p = p / "timeline.json"
        with open(p) as f:
            return cls.from_dict(json.load(f))

    def event_rows(self) -> list[dict]:
        """A single chronological event log (positions excluded)."""
        rows = []
        for r in self.rallies:
            rows.append({"t": r["start"], "event": "rally_start", "player": "", "rally_id": r["rally_id"],
                         "detail": f"{r['n_shots']} shots"})
            rows.append({"t": r["end"], "event": "rally_end", "player": "", "rally_id": r["rally_id"],
                         "detail": f"{r['duration']:.2f}s"})
        for s in self.shots:
            rows.append({"t": s["t"], "event": "shot", "player": s.get("attributed_player") or "",
                         "rally_id": s.get("rally_id") if s.get("rally_id") is not None else "",
                         "detail": s["gate"]})
        for g in self.flagged_segments:
            rows.append({"t": g["start"], "event": "flag_start", "player": g["player"], "rally_id": "",
                         "detail": g["reason"]})
            rows.append({"t": g["end"], "event": "flag_end", "player": g["player"], "rally_id": "",
                         "detail": g["reason"]})
        order = {"rally_start": 0, "flag_start": 1, "shot": 2, "flag_end": 3, "rally_end": 4}
        rows.sort(key=lambda r: (r["t"], order[r["event"]]))
        return rows

    # -- helpers -------------------------------------------------------------
    def positions_by_player(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {pid: [] for pid in self.players}
        for p in self.positions:
            out.setdefault(p["player"], []).append(p)
        for v in out.values():
            v.sort(key=lambda p: p["t"])
        return out


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            if isinstance(r.get("flags"), list):
                r["flags"] = "|".join(r["flags"])
            w.writerow(r)
    return path
