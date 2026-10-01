from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from padelfusion.audio import detect_onsets
from padelfusion.config import (
    AudioConfig,
    BallConfig,
    CaptureSpec,
    CourtConfig,
    DetectionConfig,
    PipelineConfig,
    RallyConfig,
    StatsConfig,
    TrackingConfig,
)
from padelfusion.court import (
    CourtModel,
    detect_court,
    expected_player_height_px,
    horizontal_line_segments,
    parse_corners,
    zone,
)
from padelfusion.detection import Detection
from padelfusion.distortion import (
    Distortion,
    apply_division,
    build_maps,
    estimate_k,
    fit_k_from_point_groups,
    simulate_capture_maps,
)
from padelfusion.events import Shot, segment_rallies
from padelfusion.llm import allowed_numbers, summarize, untraceable_numbers
from padelfusion.stats import TIERS, Stat, compute_stats, distance_covered
from padelfusion.synth import SynthSpec, image_corners, render_background
from padelfusion.timeline import Timeline
from padelfusion.tracking import CourtDetection, Tracker, TrackPoint
from padelfusion.validate import validate
from padelfusion.video import VideoInfo, check_capture


# -- court ------------------------------------------------------------------
def test_homography_roundtrip():
    m = CourtModel(image_corners(1920, 1080))
    pts = np.array([[0, 0], [10, 20], [5, 10], [2.5, 6.95]])
    back = m.image_to_court(m.court_to_image(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_auto_court_detection_on_rendered_court():
    spec = SynthSpec(width=1280, height=720)
    truth = CourtModel(image_corners(spec.width, spec.height))
    m = detect_court(render_background(spec, truth))
    assert m is not None and m.source == "auto"
    assert np.abs(m.corners_img - truth.corners_img).max() < 12


def test_auto_court_detection_fails_cleanly_on_blank_frame():
    assert detect_court(np.zeros((720, 1280, 3), np.uint8)) is None


def test_horizontal_line_segments_finds_service_lines():
    spec = SynthSpec(width=1280, height=720)
    truth = CourtModel(image_corners(spec.width, spec.height))
    frame = render_background(spec, truth)
    from padelfusion.court import line_mask
    segs = horizontal_line_segments(line_mask(frame))
    assert len(segs) >= 2
    for x1, y1, x2, y2 in segs:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        assert ang < 10 or ang > 170


def test_detect_ball_candidates_excludes_persistent_static_objects():
    from padelfusion.ball import detect_ball_candidates, static_ball_mask

    cfg = BallConfig()
    hsv_mid = np.uint8([[[35, 200, 220]]])
    ball_bgr = tuple(int(c) for c in cv2.cvtColor(hsv_mid, cv2.COLOR_HSV2BGR)[0, 0])
    bg_color = (60, 60, 60)
    static_pos = (200, 100)
    real_ball_pos = (50, 50)

    samples = []
    for _ in range(5):
        f = np.full((200, 300, 3), bg_color, np.uint8)
        cv2.circle(f, static_pos, 6, ball_bgr, -1)  # a bottle: present in every sample
        samples.append(f)
    static_mask = static_ball_mask(samples, cfg)

    frame = np.full((200, 300, 3), bg_color, np.uint8)
    cv2.circle(frame, static_pos, 6, ball_bgr, -1)     # the persistent bottle, still there
    cv2.circle(frame, real_ball_pos, 6, ball_bgr, -1)  # the actual ball

    cands = detect_ball_candidates(frame, None, [], cfg, static_mask=static_mask)
    assert len(cands) == 1
    assert abs(cands[0][0] - 50) < 3 and abs(cands[0][1] - 50) < 3


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


def test_detect_ball_shots_finds_a_hit_near_a_player():
    from padelfusion.ball import BallPoint, detect_ball_shots

    points = [
        BallPoint(0.0, 0.0, 5.0),
        BallPoint(0.1, 1.0, 5.0),
        BallPoint(0.2, 2.0, 5.0),  # kink here: incoming +x, outgoing -x
        BallPoint(0.3, 1.0, 5.0),
        BallPoint(0.4, 0.0, 5.0),
    ]
    player_tracks = {"P1": [TrackPoint(0.2, 2.0, 5.0, (0, 0, 10, 30))]}
    cfg = BallConfig(kink_angle_deg=90.0, close_to_player_px=1.0)
    shots = detect_ball_shots(points, player_tracks, cfg)
    assert len(shots) == 1
    assert shots[0].attributed_player == "P1"
    assert abs(shots[0].t - 0.2) < 1e-6


def test_detect_ball_shots_ignores_a_kink_far_from_any_player():
    from padelfusion.ball import BallPoint, detect_ball_shots

    points = [
        BallPoint(0.0, 0.0, 5.0),
        BallPoint(0.1, 1.0, 5.0),
        BallPoint(0.2, 2.0, 5.0),
        BallPoint(0.3, 1.0, 5.0),
        BallPoint(0.4, 0.0, 5.0),
    ]
    # bbox far from the kink point (2.0, 5.0) in pixel space
    player_tracks = {"P1": [TrackPoint(0.2, 20.0, 15.0, (500, 500, 600, 600))]}
    cfg = BallConfig(kink_angle_deg=90.0, close_to_player_px=1.0)
    shots = detect_ball_shots(points, player_tracks, cfg)
    assert shots == []


def test_ball_tracker_links_a_smooth_trajectory():
    from padelfusion.ball import BallTracker

    cfg = BallConfig(max_speed_px_s=10.0, max_gap_s=0.5)
    tr = BallTracker(cfg)
    for i in range(5):
        tr.update(i * 0.1, [(1.0 + 0.2 * i, 2.0)])
    assert len(tr.points) == 5


def test_ball_tracker_starts_new_segment_outside_speed_gate():
    from padelfusion.ball import BallTracker

    cfg = BallConfig(max_speed_px_s=5.0, max_gap_s=0.5)
    tr = BallTracker(cfg)
    tr.update(0.0, [(1.0, 2.0)])
    tr.update(0.1, [(50.0, 2.0)])  # ~490 px/s -> far outside a 5 px/s gate
    assert len(tr.points) == 1


def test_ball_tracker_links_across_a_short_gap():
    from padelfusion.ball import BallTracker

    cfg = BallConfig(max_speed_px_s=20.0, max_gap_s=0.5)
    tr = BallTracker(cfg)
    tr.update(0.0, [(1.0, 2.0)])
    tr.update(0.4, [(2.0, 2.0)])  # 0.4s gap (under max_gap_s), 2.5 px/s (under the speed gate)
    assert len(tr.points) == 2


def test_ball_config_defaults():
    cfg = BallConfig()
    assert cfg.kink_angle_deg == 20.0
    assert cfg.max_speed_px_s == 2500.0
    assert cfg.audio_merge_tolerance_s == 0.25


def test_court_config_has_distortion_threshold():
    cfg = CourtConfig()
    assert cfg.distortion_max_residual_px == 1.5


def test_court_model_distortion_k_roundtrip():
    m = CourtModel(image_corners(1280, 720), source="auto", score=0.9, k=-0.12)
    d = m.to_dict()
    assert d["distortion_k"] == -0.12
    back = CourtModel.from_dict(d)
    assert back.k == -0.12

    old_format = {"corners_img": image_corners(1280, 720).tolist(), "source": "manual"}
    assert CourtModel.from_dict(old_format).k == 0.0


def test_parse_corners_and_zone():
    c = parse_corners("1,2;3,4;5,6;7,8")
    assert c.shape == (4, 2)
    with pytest.raises(ValueError):
        parse_corners("1,2;3,4")
    assert zone(9.0, 5.0) == "net" and zone(2.0, 5.0) == "back" and zone(16.0, 5.0) == "back"


# -- distortion ---------------------------------------------------------------
def _forward_distort(pts, k, w, h):
    """Test-only: construct the raw point that `apply_division(..., k)` would
    map back to `pts` (i.e. the point-level analog of `simulate_capture_maps`).
    """
    from padelfusion.distortion import apply_division_inverse

    return apply_division_inverse(np.asarray(pts, dtype=np.float64), k, w, h)


def test_apply_division_identity_at_k_zero():
    pts = np.array([[10.0, 20.0], [640.0, 360.0], [1270.0, 710.0]])
    out = apply_division(pts, 0.0, 1280, 720)
    assert np.allclose(out, pts, atol=1e-9)


def test_apply_division_straightens_a_known_forward_distortion():
    w, h = 1280, 720
    k_true = -0.18
    straight = np.array([[x, 600.0] for x in np.linspace(80, 1200, 40)])
    distorted = _forward_distort(straight, k_true, w, h)

    def residual(pts):
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        return float(np.mean(np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)))

    assert residual(distorted) > 2.0
    corrected = apply_division(distorted, k_true, w, h)
    assert residual(corrected) < 0.5


def test_distortion_rectify_is_noop_at_k_zero():
    frame = np.random.default_rng(0).integers(0, 255, (100, 120, 3), dtype=np.uint8)
    out = Distortion(0.0).rectify(frame)
    assert out is frame


def test_distortion_rectify_caches_maps():
    frame = np.zeros((100, 120, 3), np.uint8)
    d = Distortion(-0.1)
    out1 = d.rectify(frame)
    assert out1.shape == frame.shape
    assert d._map_x is not None
    maps_id = id(d._map_x)
    d.rectify(frame)
    assert id(d._map_x) == maps_id  # reused, not rebuilt


def test_fit_k_from_point_groups_recovers_known_k():
    w, h = 1280, 720
    k_true = -0.18
    lines = [np.array([[x, y0] for x in np.linspace(80, 1200, 40)])
             for y0 in (200.0, 500.0, 650.0)]
    groups = [_forward_distort(line, k_true, w, h) for line in lines]
    k_fit, residual = fit_k_from_point_groups(groups, w, h)
    assert abs(k_fit - k_true) < 0.03
    assert residual < 0.5


def test_estimate_k_on_distorted_synthetic_frame():
    spec = SynthSpec(width=1280, height=720)
    court = CourtModel(image_corners(spec.width, spec.height))
    ideal = render_background(spec, court)
    k_true = -0.3
    map_x, map_y = simulate_capture_maps(spec.width, spec.height, k_true)
    distorted = cv2.remap(ideal, map_x, map_y, cv2.INTER_LINEAR)

    assert detect_court(distorted, CourtConfig().auto_min_score) is None  # fails before correction

    k_fit = estimate_k(distorted, CourtConfig().distortion_max_residual_px)
    assert abs(k_fit - k_true) < 0.05
    corrected = Distortion(k_fit).rectify(distorted)
    assert detect_court(corrected, CourtConfig().auto_min_score) is not None


def test_estimate_k_returns_zero_on_blank_frame():
    assert estimate_k(np.zeros((720, 1280, 3), np.uint8), 1.5) == 0.0


# -- capture spec -------------------------------------------------------------
def test_capture_spec_rejections():
    bad = VideoInfo("clip.avi", 1280, 720, 24.0, 100, 60.0, False)
    errors, warnings = check_capture(bad, CaptureSpec())
    text = " ".join(errors)
    assert "audio" in text and "30 fps" in text and "1080p" in text and "MP4" in text
    assert warnings
    good = VideoInfo("clip.mov", 1920, 1080, 29.97, 100, 400.0, True)
    assert check_capture(good, CaptureSpec()) == ([], [])


# -- audio ----------------------------------------------------------------------
def test_onsets_found_and_quiet_ones_split_off():
    cfg = AudioConfig()
    sr = cfg.sample_rate
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.002, sr * 10)
    tt = np.arange(int(0.03 * sr)) / sr
    click = np.sin(2 * np.pi * 2500 * tt) * np.exp(-tt / 0.004)
    loud, quiet = [1.0, 2.2, 3.1, 5.5, 7.0], [4.3, 8.4]
    for t in loud:
        y[int(t * sr):int(t * sr) + tt.size] += 0.5 * click
    for t in quiet:
        y[int(t * sr):int(t * sr) + tt.size] += 0.01 * click
    kept, rejected = detect_onsets(y, cfg)
    assert [round(o.t, 1) for o in kept] == loud
    assert [round(o.t, 1) for o in rejected] == quiet


# -- tracking -------------------------------------------------------------------
def _cd(x, y, hist):
    return CourtDetection(Detection((0, 0, 10, 30), 1.0, hist), np.array([x, y], float))


def test_tracking_config_has_late_bind_setting():
    assert TrackingConfig().late_bind_confirm_frames == 3


def test_tracker_initializes_with_only_the_tagged_player():
    h = [np.eye(32)[i].astype(np.float32) for i in range(4)]
    tr = Tracker(TrackingConfig())
    me_px = (2.0, 2.0)
    assert tr.initialize(0.0, [_cd(2, 2, h[0])], me_px=me_px)
    assert set(tr.tracks) == {"P1"}
    assert tr.tracks["P1"].is_user and tr.tracks["P1"].team == "near"
    assert tr.my_team == "near"


def test_tracker_initializes_with_all_four_when_available():
    h = [np.eye(32)[i].astype(np.float32) for i in range(4)]
    tr = Tracker(TrackingConfig())
    init = [_cd(2, 2, h[0]), _cd(8, 5, h[1]), _cd(3, 17, h[2]), _cd(7, 15, h[3])]
    assert tr.initialize(0.0, init, me_px=(2, 2))
    assert set(tr.tracks) == {"P1", "P2", "P3", "P4"}


def test_tracker_initialize_fails_with_no_detections():
    tr = Tracker(TrackingConfig())
    assert tr.initialize(0.0, [], me_px=(2, 2)) is False


def test_track_players_message_reflects_relaxed_requirement():
    from padelfusion.pipeline import track_players
    import inspect

    src = inspect.getsource(track_players)
    assert "two players on each side" not in src
    assert "your tagged player" in src


def test_tracker_late_binds_a_consistently_seen_second_player():
    h = [np.eye(32)[i].astype(np.float32) for i in range(4)]
    cfg = TrackingConfig(late_bind_confirm_frames=3)
    tr = Tracker(cfg)
    assert tr.initialize(0.0, [_cd(2, 2, h[0])], me_px=(2, 2))
    assert set(tr.tracks) == {"P1"}
    for k in range(1, 4):  # 3 consecutive update() calls -> streak reaches 3
        tr.update(k / 15, [_cd(2 + 0.01 * k, 2, h[0]), _cd(8, 5, h[1])])
        if k < 3:
            assert set(tr.tracks) == {"P1"}
    assert set(tr.tracks) == {"P1", "P2"}
    assert tr.tracks["P2"].team == "near"


def test_tracker_does_not_late_bind_a_single_frame_blip():
    h = [np.eye(32)[i].astype(np.float32) for i in range(4)]
    cfg = TrackingConfig(late_bind_confirm_frames=3)
    tr = Tracker(cfg)
    assert tr.initialize(0.0, [_cd(2, 2, h[0])], me_px=(2, 2))
    tr.update(1 / 15, [_cd(2, 2, h[0]), _cd(8, 5, h[1])])  # blip appears once
    tr.update(2 / 15, [_cd(2, 2, h[0])])  # gone
    tr.update(3 / 15, [_cd(2, 2, h[0])])
    tr.update(4 / 15, [_cd(2, 2, h[0])])
    assert set(tr.tracks) == {"P1"}


def test_tracker_team_constraint_and_ids():
    h = [np.eye(32)[i].astype(np.float32) for i in range(4)]
    tr = Tracker(TrackingConfig())
    init = [_cd(2, 2, h[0]), _cd(8, 5, h[1]), _cd(3, 17, h[2]), _cd(7, 15, h[3])]
    assert tr.initialize(0.0, init, me_px=None)
    assert tr.tracks["P1"].team == "near" and tr.tracks["P3"].team == "far"
    for k in range(1, 20):
        t = k / 15
        tr.update(t, [_cd(2 + 0.1 * k, 2, h[0]), _cd(8, 5, h[1]), _cd(3, 17, h[2]), _cd(7, 15, h[3])])
    pts, segs = tr.finalize()
    assert abs(pts["P1"][-1].x - 3.9) < 1e-6
    assert all(p.y < 10 for p in pts["P1"] + pts["P2"])
    assert all(p.y > 10 for p in pts["P3"] + pts["P4"])
    assert segs == []


def test_tracker_flags_ambiguous_partners():
    same = np.full(32, 1 / 32, np.float32)
    tr = Tracker(TrackingConfig())
    tr.initialize(0.0, [_cd(4.8, 5, same), _cd(5.2, 5, same), _cd(3, 17, same), _cd(7, 15, same)], None)
    tr.update(0.1, [_cd(5.0, 5.1, same), _cd(5.1, 4.9, same), _cd(3, 17, same), _cd(7, 15, same)])
    _, segs = tr.finalize()
    reasons = {s["reason"] for s in segs if s["player"] in ("P1", "P2")}
    assert any("ambiguous_id" in r for r in reasons)


# -- rallies --------------------------------------------------------------------
def test_rally_segmentation_by_gap():
    shots = [Shot(t, 1, -5, "motion", "P1", 2.0) for t in [1, 2, 3, 10, 11.2, 30]]
    rallies, isolated = segment_rallies(shots, {}, RallyConfig())
    assert [(r.start, r.end, r.n_shots) for r in rallies] == [(1, 3, 3), (10, 11.2, 2)]
    assert isolated == [30]


# -- stats ----------------------------------------------------------------------
def test_distance_smoothing_damps_jitter():
    rng = np.random.default_rng(1)
    t = np.arange(0, 10, 1 / 15)
    pts = [{"t": ti, "x": 1 + 0.5 * ti + rng.normal(0, 0.05), "y": 3 + rng.normal(0, 0.05)} for ti in t]
    raw = sum(np.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(pts, pts[1:]))
    smooth = distance_covered(pts, StatsConfig(), 1 / 15)
    assert abs(smooth - 5.0) < abs(raw - 5.0)
    assert abs(smooth - 5.0) < 0.6


def test_stat_requires_confidence():
    with pytest.raises(ValueError):
        Stat("x", "X", 1, "1", "Unknown", "m")


def _tiny_timeline():
    positions = []
    for i in range(150):
        t = i / 15
        for pid, (x, y) in {"P1": (2, 8), "P2": (8, 2), "P3": (3, 18), "P4": (7, 12)}.items():
            positions.append({"t": t, "player": pid, "team": "near" if y < 10 else "far", "x": x,
                              "y": y, "zone": zone(y, 5.0), "interpolated": False, "flags": []})
    shots = [{"t": t, "rally_id": 1, "gate": "motion", "attributed_player": "P1"} for t in (1, 2, 3)]
    rallies = [{"rally_id": 1, "start": 1, "end": 3, "duration": 2.0, "n_shots": 3,
                "clip_start": 0, "clip_end": 5}]
    meta = {"duration_s": 10.0, "analysis_fps": 15, "n_samples": 150,
            "court": {"source": "manual", "score": None}}
    players = {p: {"team": "near" if p in ("P1", "P2") else "far", "is_user": p == "P1"}
               for p in ("P1", "P2", "P3", "P4")}
    return Timeline(meta, players, positions, shots, rallies, [])


def test_compute_stats_labels_every_stat():
    res = compute_stats(_tiny_timeline(), StatsConfig())
    assert all(s.confidence in TIERS for s in res.stats)
    assert res.by_key("net_time_pct", "P1").value == 100.0
    assert res.by_key("net_time_pct", "P2").value == 0.0
    assert res.by_key("rally_count").value == 1
    assert res.by_key("shot_count").value == 3
    assert res.by_key("shots_attributed", "P1").confidence == "Low"
    assert abs(res.by_key("partner_distance_m", "P1").value - 8.5) < 0.05
    assert abs(res.by_key("partner_distance_m", "P2").value - 8.5) < 0.05
    assert abs(res.by_key("partner_distance_m", "P3").value - 7.2) < 0.05


# -- LLM guard ------------------------------------------------------------------
def test_untraceable_numbers():
    allowed = {62.0, 38.0, 1234.0, 5.8}
    assert untraceable_numbers("P1 was at the net 62% of the time, 38% back.", allowed) == []
    assert untraceable_numbers("1. Run 1234 m\n2. Spend 70% at the net", allowed) == ["70"]
    assert untraceable_numbers("average 5.8 shots", allowed) == []


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        text = self.replies.pop(0)
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=text)])


def test_summary_accepted_retried_and_withheld():
    res = compute_stats(_tiny_timeline(), StatsConfig())
    nums = allowed_numbers(res)
    assert 3.0 in nums

    ok = summarize(res, client=FakeClient(["P1 played 3 shots in rallies (Medium confidence)."]))
    assert ok.status == "ok"

    fixed = FakeClient(["P1 covered 999 m.", "P1 played 3 shots."])
    assert summarize(res, client=fixed).status == "ok"
    assert len(fixed.calls) == 2 and fixed.calls[0]["model"] == "claude-opus-5-5"

    bad = summarize(res, client=FakeClient(["P1 covered 999 m.", "Still 999 m."]))
    assert bad.status == "withheld" and bad.text is None and "999" in bad.detail


# -- validation -----------------------------------------------------------------
def test_background_detector_depth_aware_threshold_admits_small_far_blob():
    from padelfusion.detection import BackgroundDetector

    court = CourtModel(image_corners(1280, 720))
    cfg = DetectionConfig(backend="background", bg_min_area_frac=0.01)  # deliberately strict flat gate
    bg = np.full((720, 1280, 3), 40, np.uint8)
    far_h = expected_player_height_px(court, 18.0)
    blob_h = int(0.5 * far_h)  # clears the depth-aware gate (0.35 * far_h) ...
    blob_w = max(2, int(blob_h * 0.9))
    foot = court.court_to_image([[5.0, 18.0]])[0]
    x0, y0 = int(foot[0] - blob_w / 2), int(foot[1] - blob_h)

    frame_flat = bg.copy()
    frame_flat[y0:y0 + blob_h, x0:x0 + blob_w] = (200, 200, 200)
    detector_flat = BackgroundDetector(cfg, [bg])
    assert detector_flat.detect(frame_flat) == []  # ... but not the old flat one

    detector_depth = BackgroundDetector(cfg, [bg], court=court)
    dets = detector_depth.detect(frame_flat)
    assert len(dets) == 1


def test_detection_config_defaults():
    cfg = DetectionConfig()
    assert cfg.yolo_model == "yolov8s.pt"
    assert cfg.yolo_min_conf == 0.25
    assert cfg.far_field_upscale == 2.0
    assert cfg.far_field_enabled is True
    assert cfg.bg_min_height_frac == 0.35


def _det(x1, y1, x2, y2, score=0.9):
    return Detection((x1, y1, x2, y2), score)


def test_merge_detections_collapses_overlapping_keeps_best_score():
    from padelfusion.detection import merge_detections

    a = [_det(0, 0, 10, 30, 0.5)]
    b = [_det(1, 1, 11, 31, 0.9)]
    merged = merge_detections(a, b, iou_thresh=0.3)
    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_merge_detections_keeps_non_overlapping():
    from padelfusion.detection import merge_detections

    a = [_det(0, 0, 10, 30)]
    b = [_det(500, 500, 510, 530)]
    merged = merge_detections(a, b, iou_thresh=0.3)
    assert len(merged) == 2


def test_detect_two_pass_runs_detector_on_full_and_upscaled_far_crop():
    from padelfusion.detection import detect_two_pass

    calls = []

    def fake_detect(frame):
        calls.append(frame.shape)
        if frame.shape[0] == 200:  # the far-crop pass (cropped+upscaled to 200 tall)
            return [_det(10, 10, 20, 40)]
        return [_det(300, 300, 320, 360)]  # the full-frame pass

    frame = np.zeros((720, 1280, 3), np.uint8)
    far_bbox = (100, 100, 300, 200)  # 200 wide x 100 tall -> upscaled 2x -> 400x200
    dets = detect_two_pass(fake_detect, frame, far_bbox, upscale=2.0)

    assert len(calls) == 2
    assert len(dets) == 2
    far_det = next(d for d in dets if d.bbox[0] < 200)
    assert abs(far_det.bbox[0] - 105.0) < 1e-6
    assert abs(far_det.bbox[1] - 105.0) < 1e-6


def test_detect_two_pass_without_far_bbox_runs_once():
    from padelfusion.detection import detect_two_pass

    calls = []

    def fake_detect(frame):
        calls.append(frame.shape)
        return []

    frame = np.zeros((100, 100, 3), np.uint8)
    detect_two_pass(fake_detect, frame, None, upscale=2.0)
    assert len(calls) == 1


def test_fit_corners_from_lines_recovers_known_corners():
    from padelfusion.court import fit_corners_from_lines

    c = image_corners(1280, 720)  # near_left, near_right, far_right, far_left
    lines = {
        "near_baseline": [c[0].tolist(), c[1].tolist()],
        "right_sideline": [c[1].tolist(), c[2].tolist()],
        "far_baseline": [c[2].tolist(), c[3].tolist()],
        "left_sideline": [c[3].tolist(), c[0].tolist()],
    }
    result = fit_corners_from_lines(lines)
    assert result["missing"] == {}
    for name, expected in zip(("near_left", "near_right", "far_right", "far_left"), c):
        got = result["corners"][name]
        assert got is not None
        assert np.allclose(got, expected, atol=1.0)


def test_fit_corners_from_lines_reports_missing_when_a_side_is_absent():
    from padelfusion.court import fit_corners_from_lines

    c = image_corners(1280, 720)
    lines = {
        "near_baseline": [c[0].tolist(), c[1].tolist()],
        "right_sideline": [c[1].tolist(), c[2].tolist()],
        "left_sideline": [c[3].tolist(), c[0].tolist()],
    }
    result = fit_corners_from_lines(lines)
    assert result["corners"]["near_left"] is not None
    assert result["corners"]["near_right"] is not None
    assert result["corners"]["far_right"] is None
    assert result["corners"]["far_left"] is None
    assert "far_right" in result["missing"]
    assert "far_left" in result["missing"]


def test_far_field_bbox_covers_far_half_only():
    from padelfusion.court import far_field_bbox

    m = CourtModel(image_corners(1280, 720))
    bbox = far_field_bbox(m, (720, 1280, 3), margin_m=1.0)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 < x2 <= 1280
    assert 0 <= y1 < y2 <= 720
    assert y2 < 720 * 0.9


def test_expected_player_height_decreases_with_depth():
    from padelfusion.court import expected_player_height_px

    m = CourtModel(image_corners(1280, 720))
    near = expected_player_height_px(m, 1.0)
    mid = expected_player_height_px(m, 10.0)
    far = expected_player_height_px(m, 19.0)
    assert near > mid > far > 0


def test_resolve_court_corrects_fisheye_where_detect_court_fails():
    from padelfusion.pipeline import resolve_court

    spec = SynthSpec(width=1280, height=720)
    truth = CourtModel(image_corners(spec.width, spec.height))
    ideal = render_background(spec, truth)
    map_x, map_y = simulate_capture_maps(spec.width, spec.height, -0.3)
    distorted = cv2.remap(ideal, map_x, map_y, cv2.INTER_LINEAR)

    assert detect_court(distorted, PipelineConfig().court.auto_min_score) is None

    court = resolve_court(distorted, PipelineConfig())
    assert court.source == "auto"
    assert court.k != 0.0


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


def test_validate_perfect_labels_pass():
    tl = _tiny_timeline()
    labels = {
        "rallies": [{"start": 1, "end": 3}],
        "shots": [1, 2, 3],
        "positions_at_shots": [
            {"t": t, "player": p, "zone": z, "side": s}
            for t in (1, 2, 3)
            for p, z, s in [("P1", "net", "left"), ("P2", "back", "right"),
                            ("P3", "back", "left"), ("P4", "net", "right")]
        ],
    }
    checks = validate(tl, labels, {"stats": [{"confidence": "High"}]})
    assert {c.metric for c in checks} == {"rally_count_err_pct", "shot_count_err_pct", "net_share_err_pp",
                                          "unflagged_swaps_per_5min", "stats_without_confidence"}
    assert all(c.passed for c in checks), checks
