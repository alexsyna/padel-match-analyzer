"""Synthetic match generator with exact ground truth.

It renders a court seen from behind the near glass, four players with
scripted movement, and an audio track with racket impacts, ball bounces
and neighbouring-court noise. It then writes the hand-count label file the
real validation protocol uses. This exercises the whole pipeline (IO,
court detection, tracking, audio, fusion, stats, validation) without real
footage. It is not a stand-in for real-footage accuracy.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile

from .court import COURT_CORNERS, COURT_LINES, CourtModel, NET_Y, px_per_m
from .distortion import simulate_capture_maps
from .video import ffmpeg_exe

PLAYER_COLORS = {  # BGR
    "P1": (40, 40, 220),
    "P2": (40, 200, 230),
    "P3": (60, 180, 60),
    "P4": (200, 60, 200),
}
_BALL_HSV = np.uint8([[[35, 200, 220]]])
BALL_BGR = tuple(int(c) for c in cv2.cvtColor(_BALL_HSV, cv2.COLOR_HSV2BGR)[0, 0])
BASE = {"P1": (2.5, 1.5), "P2": (7.5, 5.0), "P3": (2.5, 18.0), "P4": (7.5, 15.5)}
# Direction of rally movement (towards the net for the back player of each team).
OFFSET = {"P1": (0.0, 2.5), "P2": (0.0, 1.0), "P3": (0.0, -2.5), "P4": (0.0, -1.0)}
TEAM = {"P1": "near", "P2": "near", "P3": "far", "P4": "far"}


@dataclass
class SynthSpec:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    duration_s: float = 90.0
    seed: int = 7
    sample_rate: int = 22050


def image_corners(w: int, h: int) -> np.ndarray:
    return np.array([[0.12 * w, 0.935 * h], [0.88 * w, 0.935 * h],
                     [0.66 * w, 0.30 * h], [0.34 * w, 0.30 * h]])


def make_script(spec: SynthSpec, rng: np.random.Generator) -> list[dict]:
    """Rallies as {shots: [(t, hitter)], jitter}, with 8-15 s between points."""
    rallies = []
    t = 3.0
    while True:
        n = int(rng.integers(3, 9))
        gaps = rng.uniform(0.9, 1.6, n - 1)
        if t + gaps.sum() + 2.0 > spec.duration_s:
            break
        shots = []
        for k in range(n):
            team = "near" if k % 2 == 0 else "far"
            hitter = rng.choice([p for p in TEAM if TEAM[p] == team])
            shots.append((round(t, 3), str(hitter)))
            if k < n - 1:
                t += gaps[k]
        jitter = {pid: np.array([rng.uniform(-1.2, 1.2), rng.uniform(-1.0, 1.0)]) for pid in TEAM}
        rallies.append({"shots": shots, "jitter": jitter})
        t += rng.uniform(8.0, 15.0)
    return rallies


def envelope(t: float, rallies: list[dict]) -> float:
    e = 0.0
    for r in rallies:
        a, b = r["shots"][0][0] - 0.6, r["shots"][-1][0] + 0.3
        if a <= t <= b:
            e = max(e, min(1.0, (t - a) / 0.6))
        elif b < t <= b + 1.6:
            e = max(e, 1.0 - (t - b) / 1.6)
    return e


def base_pos(pid: str, t: float, rallies: list[dict]) -> np.ndarray:
    """Rest position, which changes from point to point (as serve positions do).
    Players walk to the next point's position during the first seconds of a pause."""
    if not rallies:
        return np.array(BASE[pid])
    idx = next((i for i, r in enumerate(rallies) if t <= r["shots"][-1][0] + 1.9), len(rallies) - 1)
    cur = np.array(BASE[pid]) + rallies[idx]["jitter"][pid]
    if idx == 0:
        return cur
    prev = np.array(BASE[pid]) + rallies[idx - 1]["jitter"][pid]
    t0 = rallies[idx - 1]["shots"][-1][0] + 1.9
    f = float(np.clip((t - t0) / 2.5, 0.0, 1.0))
    return prev + f * (cur - prev)


def player_pos(pid: str, t: float, rallies: list[dict]) -> np.ndarray:
    e = envelope(t, rallies)
    bx, by = base_pos(pid, t, rallies)
    ox, oy = OFFSET[pid]
    ph = {"P1": 0.0, "P2": 1.3, "P3": 2.1, "P4": 0.7}[pid]
    dx = 1.3 * np.sin(1.7 * t + ph)
    dy = 1.1 * np.sin(1.1 * t + 2 * ph)
    return np.array([bx + e * (ox + dx), by + e * (oy + dy)])


def swinging(pid: str, t: float, rallies: list[dict]) -> bool:
    return any(abs(t - st) < 0.12 and h == pid for r in rallies for st, h in r["shots"])


def render_background(spec: SynthSpec, court: CourtModel) -> np.ndarray:
    w, h = spec.width, spec.height
    img = np.full((h, w, 3), (45, 40, 38), np.uint8)
    floor = court.court_to_image([[-1, -1], [11, -1], [11, 21], [-1, 21]]).astype(np.int32)
    cv2.fillPoly(img, [floor], (60, 70, 60))
    surf = court.court_to_image(COURT_CORNERS).astype(np.int32)
    cv2.fillPoly(img, [surf], (150, 95, 35))
    th = max(2, w // 480)
    for name in ("near_service", "far_service", "centre_service"):
        a, b = COURT_LINES[name]
        p = court.court_to_image([a, b]).astype(int)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (240, 240, 240), th, cv2.LINE_AA)
    return img


def draw_net(img: np.ndarray, court: CourtModel) -> None:
    p = court.court_to_image([[0, NET_Y], [10, NET_Y]]).astype(int)
    hh = int(0.88 * px_per_m(court, NET_Y))
    pts = np.array([p[0], p[1], p[1] - [0, hh], p[0] - [0, hh]], np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (30, 30, 30))
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.line(img, tuple(p[0] - [0, hh]), tuple(p[1] - [0, hh]), (235, 235, 235), max(2, img.shape[1] // 640))


def draw_player(img, court, pid, xy, swing: bool) -> None:
    s = px_per_m(court, xy[1])
    foot = court.court_to_image([xy])[0]
    bw, bh = 0.6 * s * (1.5 if swing else 1.0), 1.8 * s
    x1, x2 = int(foot[0] - bw / 2), int(foot[0] + bw / 2)
    y1, y2 = int(foot[1] - bh), int(foot[1])
    cv2.rectangle(img, (x1, y1 + int(0.15 * bh)), (x2, y2), PLAYER_COLORS[pid], -1)
    cv2.circle(img, (int(foot[0]), y1 + int(0.08 * bh)), max(2, int(0.08 * bh)), (150, 180, 225), -1)


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


def render_audio(spec: SynthSpec, rallies: list[dict], rng: np.random.Generator) -> np.ndarray:
    sr = spec.sample_rate
    n = int(spec.duration_s * sr)
    y = rng.normal(0, 0.003, n).astype(np.float64)

    def add(t, sig):
        i = int(t * sr)
        j = min(n, i + len(sig))
        if 0 <= i < n:
            y[i:j] += sig[: j - i]

    tt = np.arange(int(0.03 * sr)) / sr
    for r in rallies:
        for k, (t, hitter) in enumerate(r["shots"]):
            amp = 0.5 if TEAM[hitter] == "near" else 0.22
            click = (np.sin(2 * np.pi * 2600 * tt) + 0.6 * rng.normal(0, 1, tt.size)) * np.exp(-tt / 0.004)
            add(t, amp * click)
            if k < len(r["shots"]) - 1:  # bounce: soft low thud
                tb = np.arange(int(0.06 * sr)) / sr
                thud = np.sin(2 * np.pi * 350 * tb) * np.minimum(1, tb / 0.006) * np.exp(-tb / 0.02)
                add(t + 0.45, 0.25 * thud)
    # Neighbouring courts: impacts between our rallies, some loud enough to pass the level filter.
    gaps = [(a["shots"][-1][0] + 2.2, b["shots"][0][0] - 1.0) for a, b in zip(rallies, rallies[1:])]
    for a, b in gaps:
        for t in np.arange(a, b, 0.9):
            amp = 0.06 if rng.random() < 0.5 else 0.015
            click = np.sin(2 * np.pi * 2300 * tt) * np.exp(-tt / 0.005)
            add(float(t), amp * click)
    return np.clip(y, -1, 1)


def labels_from_script(rallies: list[dict]) -> dict:
    shots = [t for r in rallies for t, _ in r["shots"]]
    pos = []
    for t in shots:
        xy = {pid: player_pos(pid, t, rallies) for pid in TEAM}
        for pid in TEAM:
            partner = next(p for p in TEAM if TEAM[p] == TEAM[pid] and p != pid)
            pos.append({"t": t, "player": pid,
                        "zone": "net" if abs(xy[pid][1] - NET_Y) <= 5.0 else "back",
                        "side": "left" if xy[pid][0] < xy[partner][0] else "right"})
    return {"rallies": [{"start": r["shots"][0][0], "end": r["shots"][-1][0]} for r in rallies],
            "shots": shots, "positions_at_shots": pos}


def generate(out_dir: str | Path, spec: SynthSpec | None = None,
            distortion_k: float = 0.0, render_ball: bool = False) -> dict[str, Path]:
    spec = spec or SynthSpec()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    corners = image_corners(spec.width, spec.height)
    court = CourtModel(corners)
    rallies = make_script(spec, rng)
    bg = render_background(spec, court)
    maps = simulate_capture_maps(spec.width, spec.height, distortion_k) if distortion_k else None

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
            if render_ball:
                bp = ball_pos(t, rallies)
                if bp is not None:
                    draw_ball(frame, court, bp)
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
