"""Pipeline configuration.

Every threshold lives here so a run is reproducible: the full config is
written into the timeline's metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CaptureSpec:
    min_fps: float = 29.5  # 29.97 fps phone footage counts as 30
    min_short_side_px: int = 1080
    require_audio: bool = True
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov")
    recommended_min_s: float = 5 * 60
    recommended_max_s: float = 10 * 60


@dataclass
class CourtConfig:
    # A player counts as "at the net" while their feet are within this distance
    # of the net, i.e. in the front half of their own 10 m side. This matches
    # the ground-truth protocol, which labels which half each player is in.
    net_zone_depth_m: float = 5.0
    # Detections whose feet land further than this outside the court are
    # dropped. That removes glass reflections and people on other courts.
    out_of_court_margin_m: float = 0.75
    # Minimum line-support score for automatic court detection to be accepted.
    auto_min_score: float = 0.55
    # Mean perpendicular-pixel residual, after applying a candidate distortion
    # correction to the court's own line samples, below which auto
    # self-calibration (distortion.py) is accepted. Above this, k = 0.0 (no
    # correction) — the exact pre-fisheye-fix behavior.
    distortion_max_residual_px: float = 1.5


@dataclass
class DetectionConfig:
    backend: str = "auto"  # auto | yolo | background
    yolo_model: str = "yolov8s.pt"
    yolo_min_conf: float = 0.25
    bg_samples: int = 60
    bg_diff_threshold: int = 35
    bg_min_area_frac: float = 0.00025
    # Depth-aware background-detector gate: a component's bbox height must
    # clear this fraction of the expected on-screen player height at its
    # court depth (court.expected_player_height_px). Used only when a
    # CourtModel is available; otherwise bg_min_area_frac is the sole gate.
    bg_min_height_frac: float = 0.35
    # Court-aware two-pass detection: also run detection on an upscaled crop
    # of the far half of the court, to recover small/distant players.
    far_field_upscale: float = 2.0
    far_field_enabled: bool = True


@dataclass
class TrackingConfig:
    analysis_fps: float = 15.0
    max_speed_mps: float = 7.5
    gate_margin_m: float = 1.2
    appearance_weight: float = 2.0
    # If the swapped assignment of two partners costs less than this much more
    # than the chosen one, the frame is flagged as identity-ambiguous.
    ambiguity_margin: float = 0.6
    close_partner_m: float = 0.9
    max_interp_gap_s: float = 1.0
    init_search_s: float = 10.0
    # A side's extra (non-tagged) player only gets a track once an
    # unmatched detection there has appeared this many consecutive analyzed
    # frames — guards against a single-frame reflection/shadow (see the
    # real-footage background-subtraction diagnostic) becoming a phantom
    # player.
    late_bind_confirm_frames: int = 3


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    band_low_hz: float = 1000.0
    band_high_hz: float = 6000.0
    n_fft: int = 512
    hop: int = 128
    threshold_k: float = 4.0
    min_interval_s: float = 0.15
    # Onsets quieter than (90th percentile level - this) dB are treated as
    # coming from neighbouring courts.
    rel_level_floor_db: float = 22.0


@dataclass
class BallConfig:
    # HSV bounds for the ball's yellow-green color (OpenCV hue range 0-179).
    hsv_lower: tuple[int, int, int] = (25, 80, 100)
    hsv_upper: tuple[int, int, int] = (45, 255, 255)
    min_size_px: int = 3
    max_size_px: int = 30
    min_aspect: float = 0.4
    max_aspect: float = 2.5
    # A candidate blob inside any player's bbox (+ this margin) is dropped —
    # this is what excludes same-colored shoes/gear.
    player_exclude_margin_px: float = 6.0
    # Everything below is in pixel space, not court metres: the ball is
    # usually airborne, so there's no valid ground-plane projection for it
    # (see ball.py's module docstring for why).
    max_speed_px_s: float = 2500.0
    max_gap_s: float = 0.5
    # Minimum incoming/outgoing direction change (degrees) to call a
    # trajectory point a hit, rather than smooth gravity-curved flight.
    # Tuned against real hand-labeled footage (not just synthetic data) —
    # see the "Real-footage validation" note in ball.py.
    kink_angle_deg: float = 20.0
    # A hit's kink point must be within this many pixels of some player's
    # bounding box (0 if inside it) to count as a shot.
    close_to_player_px: float = 100.0
    # Real rally shots are at least several tenths of a second apart; this
    # also dedupes the double-kink artifact a single real hit can produce a
    # couple of analyzed frames apart (detection-timing jitter around the
    # same physical event, not two distinct hits).
    min_interval_s: float = 0.4
    # An audio onset within this of an existing ball-detected shot is treated
    # as the same event and not added again.
    audio_merge_tolerance_s: float = 0.25


@dataclass
class ShotConfig:
    motion_window_before_s: float = 0.4
    motion_window_after_s: float = 0.25
    min_player_speed_mps: float = 0.6
    min_bbox_change: float = 0.08
    # Require a swing-like change in a player's box shape. Speed alone is not
    # enough, because players walk between points while neighbouring courts
    # keep playing.
    require_swing: bool = True


@dataclass
class RallyConfig:
    max_shot_gap_s: float = 3.5
    min_shots: int = 2
    reset_still_s: float = 2.0
    reset_speed_mps: float = 0.35
    clip_pad_before_s: float = 1.5
    clip_pad_after_s: float = 2.0


@dataclass
class StatsConfig:
    heatmap_bin_m: float = 0.5
    distance_smooth_s: float = 0.6
    max_plausible_speed_mps: float = 8.0
    # A player whose track is flagged for more than this share of the clip
    # has their positional stats downgraded one confidence tier.
    flag_downgrade_frac: float = 0.15


@dataclass
class PipelineConfig:
    capture: CaptureSpec = field(default_factory=CaptureSpec)
    court: CourtConfig = field(default_factory=CourtConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    shots: ShotConfig = field(default_factory=ShotConfig)
    rallies: RallyConfig = field(default_factory=RallyConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    def to_dict(self) -> dict:
        return asdict(self)
