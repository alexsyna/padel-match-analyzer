# Ball-based shot detection

## Context

Shot detection today is purely audio-based (`audio.py`'s band-passed onset
detection). Real footage from the user has no audio track at all
(`has_audio: False`, confirmed on two separate exports of the same match),
so shot count, rally segmentation, and rally duration are all currently
empty for this user's actual use case.

Direct diagnostics against the real footage (before any design work) found:

- Stock YOLO's COCO "sports ball" class is not viable: 1 detection out of 14
  sampled frames (8s apart), at 0.10 confidence. Small/fast/motion-blurred
  ball detection is a known hard problem generic object detectors aren't
  trained for.
- A simple HSV yellow-green color threshold *does* find the ball — visually
  confirmed on a real frame — but is confounded by neon-yellow athletic
  shoes (worn by the players in this footage), which are near-identical in
  color and produce false-positive candidates in the same size range.

The user has decided: ball-based detection is the primary/default shot
source going forward. Audio (when present) only fills in gaps the ball
detector misses — the reverse of today's audio-only setup, not an equal-
weight combination.

## Goals

- Detect shot events from the ball's visual trajectory, reliably enough to
  be useful for rally segmentation and shot counts, on footage with no audio
  at all.
- Eliminate the shoe-color false-positive problem using information the
  pipeline already has (tracked player bounding boxes), not a more complex
  color model.
- When audio is present, use it only to fill gaps the ball detector missed,
  not as the primary signal.
- Be honest in the confidence tier when ball-detected shots are used — this
  is a materially less-validated signal than the existing audio path.

## Non-goals

- A research-grade ball tracker (e.g. TrackNet-style trained heatmap
  regression). Out of scope — this is a heuristic (color + trajectory-kink)
  approach, expected to need real-footage threshold tuning after first
  implementation, same as the fisheye `k` thresholds and detection
  confidence thresholds earlier in this project.
- Shot *type* classification (serve/volley/smash) — only shot *timing* and
  best-effort *attribution* (nearest player), matching what audio-based
  detection already produces.
- Ball detection for stats *other* than shots (e.g. no ball-speed stat) —
  out of scope for this pass.

## Architecture

### New `BallConfig` (`config.py`)

```
hsv_lower: tuple[int, int, int] = (25, 80, 100)
hsv_upper: tuple[int, int, int] = (45, 255, 255)
min_size_px: int = 3
max_size_px: int = 30
min_aspect: float = 0.4
max_aspect: float = 2.5
player_exclude_margin_px: float = 6.0
max_speed_mps: float = 35.0
max_gap_s: float = 0.5
kink_angle_deg: float = 55.0
close_to_player_m: float = 2.5
min_interval_s: float = 0.15
audio_merge_tolerance_s: float = 0.25
```

### New `ball.py`

- `detect_ball_candidates(frame, court_mask, player_bboxes, cfg) -> list[(x, y)]`
  — HSV threshold restricted to the court mask, connected-component size/
  aspect filter, then drops any candidate whose centroid falls inside any
  player bbox (+ margin). This is what removes the shoe false-positives.
- `BallTracker` — a single-object, gap-tolerant tracker (court-coordinate
  space, consistent with `Tracker`'s own coordinate space): nearest-
  candidate-within-speed-gate association frame to frame; starts a fresh
  trajectory segment when a gap exceeds `max_gap_s` rather than forcing a
  bad association across an occlusion.
- `detect_ball_shots(points, player_tracks, cfg) -> list[Shot]` — walks the
  trajectory in overlapping triples, flags a point as a hit when the
  incoming/outgoing velocity vectors' angle exceeds `kink_angle_deg` *and*
  some tracked player is within `close_to_player_m` at that instant
  (excludes floor/glass bounces away from any player). Reuses the existing
  `Shot` dataclass: `gate="ball"` (new value alongside today's `"motion"`/
  `"ungated"`), `strength` repurposed to hold the kink angle in degrees
  (documented, since audio's `strength`/`level_db` don't have a ball-
  detection equivalent), `level_db=0.0` placeholder, `attributed_player` =
  the nearest player.
- `merge_with_audio_gaps(ball_shots, audio_shots, tolerance_s) -> list[Shot]`
  — ball shots plus any audio shot not within `tolerance_s` of an existing
  ball shot.

### Wiring into `pipeline.py`

`track_players`'s existing per-frame loop already computes each frame's raw
player detections before filtering to court-detections — ball candidate
detection reuses that same per-frame detector call (no extra inference
pass) and feeds a `BallTracker` alongside the existing `Tracker`.
`track_players` returns the ball tracker's points as a new value.

`analyze()`: after `track_players`, call `detect_ball_shots` to get
`ball_shots` (always). If `info.has_audio`, compute audio onsets as today,
`fuse_shots` them into `audio_shots`, then
`merge_with_audio_gaps(ball_shots, audio_shots, cfg.ball.audio_merge_tolerance_s)`.
Record `meta["shots_source"]`:
- `"ball"` if ball found shots and either there's no audio or nothing needed
  filling
- `"ball+audio"` if both contributed
- `"audio"` only if ball found nothing at all and audio filled in
  everything
- `"none"` if neither found anything

### Confidence tiering (`stats.py`)

Hooks into the existing `downgrade()` mechanism (`stats.py:19-20`) and the
`shot_tier` logic already at `stats.py:162-166`: read
`meta.get("shots_source", "none")` alongside the existing ungated-ratio
check, and downgrade `shot_count`'s tier by one level whenever
`"ball"` is part of `shots_source` — ball-detected shots are less validated
than audio ones, regardless of how many there are.

## Testing

- Unit: `detect_ball_candidates` on a synthetic frame with a known ball-
  colored blob and a same-colored blob placed inside a given player bbox —
  confirms the in-bbox one is excluded and the free one survives.
- Unit: `BallTracker` — a smooth trajectory gets linked into one continuous
  run; a candidate far outside the speed gate does not get linked (starts a
  new segment instead); a gap under `max_gap_s` still links across it.
- Unit: `detect_ball_shots` — a synthetic trajectory with one sharp
  direction change near a player's position produces exactly one `Shot`
  with that player attributed; the same kink far from any player produces
  none (simulates a floor/wall bounce away from play).
- Unit: `merge_with_audio_gaps` — an audio shot far from any ball shot is
  added; one close to an existing ball shot is dropped.
- Integration: extend the synthetic match test data with a rendered ball
  (small colored circle following a scripted trajectory with deliberate
  direction-change "hits") to validate the full pipeline path end-to-end,
  the same pattern used for the fisheye and distant-player work.

## Risks

- The hit-detection thresholds (`kink_angle_deg`, `max_speed_mps`,
  `max_gap_s`, `close_to_player_m`) are validated only against the
  *candidate-detection* half of this (confirmed the ball is color-
  detectable and shoes are excludable) — the trajectory-kink logic itself
  has not been run against real footage yet. Expect a tuning pass once this
  lands, the same pattern as every other threshold introduced this session.
- Motion blur or partial occlusion by the net/glass could fragment the
  trajectory into many short segments rather than one continuous one,
  weakening the kink detector's ability to compute a clean in/out velocity
  around a hit. The gap-tolerant tracker mitigates this but doesn't
  eliminate it.
- A ball bouncing near a player without being hit (e.g. a missed shot) could
  register as a false shot, since the current design can't distinguish "hit
  by a racket" from "bounced near a player" — both produce a sharp direction
  change close to a tracked player. No mitigation planned for this pass;
  flagging it as a known limitation rather than solving it now.
