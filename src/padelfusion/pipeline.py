"""End-to-end pipeline: video -> detectors -> event timeline -> stats -> report."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import video as vid
from .audio import detect_onsets
from .ball import BallTracker, detect_ball_candidates, detect_ball_shots, merge_with_audio_gaps, static_ball_mask
from .config import PipelineConfig
from .court import CourtModel, detect_court, far_field_bbox, team_side, zone
from .detection import make_detector
from .distortion import Distortion, estimate_k
from .events import fuse_shots, segment_rallies
from .llm import Summary, summarize
from .report import render_report
from .stats import StatsResult, compute_stats
from .timeline import Timeline
from .tracking import Tracker, to_court_detections

log = logging.getLogger("padelfusion")


class CaptureSpecError(ValueError):
    pass


class CourtNotFoundError(RuntimeError):
    pass


class TrackingInitError(RuntimeError):
    pass


@dataclass
class AnalysisResult:
    timeline: Timeline
    stats: StatsResult
    summary: Summary
    report_path: Path
    out_dir: Path


def court_region_mask(court: CourtModel, shape, margin_m: float) -> np.ndarray:
    """Court area (plus margin) extended upwards, so players' bodies are kept
    while the floor outside the court is masked out."""
    h, w = shape[:2]
    m = margin_m
    poly = court.court_to_image([[-m, -m], [10 + m, -m], [10 + m, 20 + m], [-m, 20 + m]])
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    up = max(3, int(0.3 * h))
    kernel = np.ones((up, 1), np.uint8)
    return cv2.dilate(mask, kernel, anchor=(0, 0))


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
    ball_static_mask = static_ball_mask(
        [dist.rectify(f) for f in vid.sample_frames(video_path, cfg.detection.bg_samples, times)], cfg.ball)
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
        ball_px = detect_ball_candidates(frame, mask, [d.bbox for d in raw_dets], cfg.ball,
                                         static_mask=ball_static_mask)
        if ball_px:
            ball_tracker.update(t, ball_px)
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


def build_timeline(info: vid.VideoInfo, court: CourtModel, cfg: PipelineConfig, tracker: Tracker,
                   points, segments, shots, rallies, rejected, isolated, detector_name: str,
                   n_samples: int, warnings: list[str], shots_source: str, ball_points) -> Timeline:
    positions = []
    for pid, pts in points.items():
        for p in pts:
            positions.append({
                "t": round(p.t, 3), "player": pid, "team": team_side(p.y),
                "x": round(p.x, 3), "y": round(p.y, 3),
                "zone": zone(p.y, cfg.court.net_zone_depth_m),
                "interpolated": p.interpolated, "flags": list(p.flags),
                "bbox_w": round(p.bbox[2] - p.bbox[0], 1), "bbox_h": round(p.bbox[3] - p.bbox[1], 1),
                "bbox": [round(v, 1) for v in p.bbox],
            })
    positions.sort(key=lambda r: (r["t"], r["player"]))
    ball_positions = [{"t": round(b.t, 3), "x": round(b.x, 1), "y": round(b.y, 1)} for b in ball_points]
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
                    [r.to_dict() for r in rallies], segments, rejected, [round(t, 3) for t in isolated],
                    ball_positions)


def analyze(video_path, out_dir, cfg: PipelineConfig | None = None, corners=None, me_px=None,
            use_llm: bool = True, cut_clips: bool = False, skip_capture_check: bool = False,
            llm_client=None, progress=None) -> AnalysisResult:
    """`progress(stage, fraction)` is called as the run advances (used by the web UI)."""
    cfg = cfg or PipelineConfig()

    def step(stage: str, frac: float) -> None:
        log.info(stage)
        if progress is not None:
            progress(stage, frac)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    step("Checking the video", 0.01)
    info = vid.probe(video_path)
    errors, warnings = vid.check_capture(info, cfg.capture)
    if errors and not skip_capture_check:
        raise CaptureSpecError("video rejected: " + "; ".join(errors))
    warnings = warnings + ([f"capture check skipped: {e}" for e in errors] if errors else [])

    step("Mapping the court", 0.03)
    court = resolve_court(vid.read_frame(video_path, 0.0), cfg, corners)
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
                        shots_source, ball_points)
    tl.save(out)

    step("Computing stats", 0.90)
    result = compute_stats(tl, cfg.stats)
    with open(out / "stats.json", "w") as f:
        json.dump(result.to_dict(), f, indent=1)

    if use_llm:
        step("Writing the coaching summary with Claude", 0.93)
    summary = summarize(result, client=llm_client) if use_llm else Summary(None, "skipped", "disabled with --no-llm")
    (out / "summary.json").write_text(json.dumps(summary.__dict__, indent=1))

    clips = None
    if cut_clips and tl.rallies:
        clip_dir = out / "clips"
        clip_dir.mkdir(exist_ok=True)
        clips = {}
        for r in tl.rallies:
            p = clip_dir / f"rally_{r['rally_id']:03d}.mp4"
            vid.cut_clip(video_path, p, r["clip_start"], r["clip_end"])
            clips[r["rally_id"]] = f"clips/{p.name}"
    step("Building the report", 0.98)
    report = render_report(tl, result, summary, out / "report.html", video_path, clips)
    step("Done", 1.0)
    return AnalysisResult(tl, result, summary, report, out)
