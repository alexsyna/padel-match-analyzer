"""Synthetic match through the whole pipeline: video file -> report -> validation."""

import json

import pytest

from padelfusion.config import PipelineConfig
from padelfusion.pipeline import CaptureSpecError, analyze
from padelfusion.synth import SynthSpec, generate
from padelfusion.timeline import Timeline
from padelfusion.validate import validate_dir


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    return generate(out, SynthSpec(width=1280, height=720, duration_s=45.0, seed=3))


def test_generate_accepts_distortion_k(tmp_path):
    paths = generate(tmp_path, SynthSpec(width=1280, height=720, duration_s=3.0, seed=1),
                     distortion_k=-0.2)
    meta = json.loads(paths["meta"].read_text())
    assert meta["distortion_k"] == -0.2
    assert len(meta["me_px"]) == 2
    assert paths["video"].exists()


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


def test_capture_spec_rejects_720p(synthetic, tmp_path):
    with pytest.raises(CaptureSpecError, match="1080p"):
        analyze(synthetic["video"], tmp_path, use_llm=False)


@pytest.fixture(scope="module")
def synthetic_fisheye(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_fisheye")
    return generate(out, SynthSpec(width=1280, height=720, duration_s=45.0, seed=3),
                    distortion_k=-0.3)


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


def test_pipeline_end_to_end(synthetic, tmp_path):
    meta = json.loads(synthetic["meta"].read_text())
    cfg = PipelineConfig()
    cfg.detection.backend = "background"
    res = analyze(synthetic["video"], tmp_path, cfg, me_px=tuple(meta["me_px"]),
                  use_llm=False, skip_capture_check=True)

    assert res.timeline.meta["court"]["source"] == "auto"
    for name in ("timeline.json", "positions.csv", "shots.csv", "rallies.csv", "events.csv",
                 "flagged_segments.csv", "stats.json", "report.html"):
        assert (tmp_path / name).exists(), name
    assert Timeline.load(tmp_path).rallies == res.timeline.rallies

    # The user tag lands on the right player: P1 starts on the near-left.
    p1 = next(p for p in res.timeline.positions if p["player"] == "P1")
    assert p1["team"] == "near" and p1["x"] < 5

    checks = {c.metric: c for c in validate_dir(tmp_path, synthetic["labels"])}
    for metric in ("rally_count_err_pct", "shot_count_err_pct", "unflagged_swaps_per_5min",
                   "stats_without_confidence"):
        assert checks[metric].passed, checks[metric]
    # Net share on a short clip is sensitive to players standing on the zone
    # boundary, so check it against a looser bound.
    assert checks["net_share_err_pp"].value <= 15.0

    html = (tmp_path / "report.html").read_text()
    assert "badge" in html and "#t=" in html
