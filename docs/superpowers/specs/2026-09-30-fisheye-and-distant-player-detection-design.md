# Fisheye correction + distant-player detection

## Problem

Real footage shot with a wide/fisheye action-cam lens (confirmed via a sample
frame: the near baseline and sidelines are visibly curved arcs, not straight
lines) breaks two parts of the pipeline:

1. **Court detection fails.** `detect_court` (`court.py`) finds the court by
   Hough-detecting *straight* line segments and fitting one planar
   `cv2.getPerspectiveTransform` homography from two service lines. Neither
   assumption holds on a fisheye frame: Hough only catches short near-straight
   chords of the curve, and even correct corner points can't make one flat
   homography map the rest of the court correctly, because fisheye warp is
   non-linear.
2. **Distant players get missed**, confirmed to be genuine detector misses on
   players who are in-frame but far from the camera (small, few pixels) —
   not an off-screen/framing issue. Neither `YoloDetector`'s default
   confidence/model size nor `BackgroundDetector`'s single global
   `bg_min_area_frac` accounts for how much smaller a player gets at the far
   baseline vs. the near one.

## Goals

- Recover a working, accurate court model on fisheye footage, automatically
  in the common case, with a bounded manual fallback for the rest.
- Recover players at the far side of the court that are currently missed by
  both detector backends.
- Zero behavior change for existing non-fisheye videos (k≈0 must be a true
  no-op) and for the existing synthetic-match tests.

## Non-goals

- General camera calibration UX (checkerboard capture, per-device profiles).
  Out of scope per the product's own "no terminal, just buttons" philosophy.
- Correcting for anything beyond a single-parameter radial (barrel) distortion
  model. Tangential distortion / rolling shutter are not addressed.

---

## Part 1: Fisheye self-calibration and undistortion

### Model

Single-parameter division model (Fitzgibbon 2001, the standard "plumb-line"
barrel-correction model): with coordinates centred on the image and normalized
by the half-diagonal (so `k` stays resolution-independent, roughly in
`[-1, 0]` for barrel distortion),

```
r_u = r_d / (1 + k * r_d^2)
```

This is closed-form invertible in both directions, which is what lets us
build a `cv2.remap` map once per video and reuse it for every analyzed frame
cheaply.

### Auto self-calibration

Source of truth: the padel court already gives us multiple lines that must be
dead straight in reality — both sidelines, both baselines, both service
lines, the glass/fence edges. `line_mask` already extracts candidate line
pixels; today those get grouped and straight-line-fit per cluster. For
calibration, take the pixel points of the two or three longest such clusters
(picked for spanning different parts of the frame, e.g. one near-horizontal,
one near-vertical) and, for a scalar `k` from a bounded search
(`scipy.optimize.minimize_scalar`, bounded), apply the division-model warp to
those points and measure total squared perpendicular residual from a
straight-line fit (`cv2.fitLine`) after warping. Pick the `k` minimizing that
residual.

- **Accept** if mean residual after correction is below
  `CourtConfig.distortion_max_residual_px` (new field, default tuned during
  implementation against the synthetic fisheye test below).
- **Reject** (fall back to `k = 0`, i.e. today's exact behavior) if no `k` in
  the search range gets under that threshold, or if fewer than two usable
  line clusters were found. This keeps the failure mode identical to today's
  `CourtNotFoundError` path — pure regression safety.

### Manual fallback

Only reachable when auto self-calibration was rejected **and** the user is
already in the existing "Set corners by hand" flow (`web/index.html`). Add an
optional extra step: click 3 points along one visibly-curved line (reusing
the existing point-click UI pattern). Fit `k` from those 3 points directly
(same residual-minimization, now with one line instead of two — still
well-posed for a single scalar parameter). This step is skippable; skipping
it preserves exactly today's manual-corners behavior (`k = 0`).

### Where it plugs in

- `resolve_court` (`pipeline.py`) becomes the integration point: before
  calling `detect_court`, attempt self-calibration on the first frame,
  produce a `(map_x, map_y)` remap pair (or `None`), undistort the frame, then
  call `detect_court` unchanged on the corrected frame.
- The same `(map_x, map_y)` must be applied to **every** frame read during
  `track_players` and to the frames sampled for the background detector
  (`vid.sample_frames`) — the resulting `CourtModel`'s homography is only
  valid in undistorted pixel space. Thread this through as a small wrapper
  around frame reads rather than changing `video.py`'s public read functions.
- `CourtModel.to_dict()` gains an optional `distortion_k` field (default
  `0.0`) for reproducibility in `timeline.json`, mirroring how `source` and
  `score` are already recorded.
- Webapp: the `/court/preview` and `/analyze` endpoints (`webapp.py`) pass
  the calibration result through the same run state that already carries
  `corners`.

### Testing

- Unit: synthetic points on a known line, forward-warped with a known `k`;
  confirm the fit recovers `k` within tight tolerance.
- Unit: `detect_court` on a frame that's been forward-distorted with a known
  `k` fails (or scores low) before correction and succeeds after — using
  `synth.render_background`'s output as the pinhole source, since it already
  builds frames from a `CourtModel` we control.
- Extend `test_end_to_end.py` with a fisheye variant: forward-distort every
  rendered frame in the synthetic match at generation time, run the full
  pipeline including auto self-calibration, and check the same
  `validate.py` thresholds still pass. This is the real regression guard for
  the reported bug.
- Regression: all existing (non-distorted) tests must pass byte-for-byte
  unchanged — `k = 0` must be a no-op through the remap step.

---

## Part 2: Distant-player detection

### B. Cheap defaults (do alongside A)

Bump `DetectionConfig.yolo_model` default from `yolov8n.pt` to `yolov8s.pt`
and lower `yolo_min_conf` slightly. No new code path; ships with Part 1.

### A. Court-aware two-pass detection (the real fix)

Once a corrected `CourtModel` exists, we know exactly which pixel region
corresponds to the far half of the court (`y` from `NET_Y` to
`COURT_LENGTH`, padded by `out_of_court_margin_m`) via
`court.court_to_image` on that region's corners, clipped to frame bounds.

- **YOLO backend**: per analyzed frame, run detection once on the full frame
  (as today), then crop the far-court bounding box, upscale by
  `DetectionConfig.far_field_upscale` (new field, default `2.0`), run
  detection again, and map resulting boxes back to full-frame coordinates.
  Merge the two detection sets with a new `merge_detections(dets_a, dets_b,
  iou_thresh)` helper in `detection.py`: pairs whose IoU (in full-frame
  coordinates) exceeds the threshold keep only the higher-confidence box;
  non-overlapping boxes from either pass are both kept.
- **Background backend**: replace the single global `bg_min_area_frac` cutoff
  with a depth-aware one. For each connected component, take its foot point,
  convert to court coordinates, and estimate the expected on-screen player
  height at that depth from the homography's local scale (the pixel distance
  between two points 1 m apart in `y` at that same `x`, times an assumed
  ~1.7 m player height). Require the component's bbox height to clear a
  fraction of that expected height rather than a fixed pixel/area count.
  `bg_min_area_frac` remains as the reference value at mid-court, for
  backward-compatible config continuity.
- **Graceful degradation**: if no `CourtModel` is available when detection
  runs (shouldn't happen post-`resolve_court`, but defensively), skip the
  far-field pass and log a warning rather than failing the run.

### Testing

- Unit: `merge_detections` — overlapping boxes collapse to the
  higher-confidence one; non-overlapping boxes both survive.
- Unit: depth-aware background threshold — construct a known homography and
  confirm the expected pixel height decreases monotonically from near to far
  baseline, and that a small far-court blob that would fail the old flat
  `bg_min_area_frac` now clears the depth-aware one.
- Extend the synthetic end-to-end test to include players who spend time at
  the far baseline at a size that was chosen to fail today's flat threshold,
  confirming they're now tracked (rally/shot counts stop under-counting).

---

## Config changes summary

`CourtConfig`: `+ distortion_max_residual_px: float` (new)
`DetectionConfig`: `yolo_model` default change, `yolo_min_conf` default
change, `+ far_field_upscale: float`, `+ far_field_enabled: bool = True`

All additive/default-changing — no existing config field is removed or
repurposed.

## Risks

- Self-calibration residual threshold needs real tuning against actual
  fisheye footage, not just synthetic data — flag this as a "verify against
  the user's real clip" step before considering Part 1 done, not just green
  tests.
- Two-pass detection roughly doubles YOLO inference cost per analyzed frame
  (full frame + far-crop). Acceptable at `analysis_fps = 15` on CPU/GPU for a
  5–10 min clip, but worth a quick timing check during implementation.
