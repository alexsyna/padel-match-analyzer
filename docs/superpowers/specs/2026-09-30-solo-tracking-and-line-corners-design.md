# Solo-player tracking init + line-intersection corner entry

## Context

Real-footage testing (against the user's actual clip, `Sarmad-Padel-Game copy.mp4`,
via the data already produced by a prior analysis run) surfaced two separate
blockers, diagnosed directly against that footage before design work started:

1. **Tracker initialization is all-or-nothing.** `Tracker.initialize` (via
   `choose_initial` in `tracking.py`) requires 2+ detections on *both* sides
   of the net in one frame before creating any tracks at all. A live
   diagnostic (YOLO against an actual frame from the real clip) found 3 of 4
   real players cleanly, but that's not "2 per side simultaneously" every
   frame — and background-subtraction (the silent fallback when YOLO isn't
   installed) found zero real players and ~30 false positives on background
   clutter (ceiling lights, fence posts, a person on a neighbouring court).
   The user has confirmed they only care about their own tagged player
   (`me_px`) being tracked reliably — partner/opponent tracking is a bonus
   when available, not a requirement.
2. **Manual corner entry requires 4 literal clicks on the actual corners.**
   On wide/fisheye footage the true corners can be awkward to click precisely
   (curved lines near the edge) or fall outside the visible frame entirely.
   The user has agreed to a line-intersection based entry method instead,
   accepting that a corner is simply unavailable when an entire side of the
   court is never visible.

A prior investigation (background Explore agent) confirmed `stats.py`,
`report.py`, and `validate.py` already iterate dynamically over however many
players are present in `timeline.players` — no hardcoded "P1".."P4" anywhere,
and the one team-aware check (`validate.py`'s partner-swap metric) already
skips any team with != 2 members rather than crashing. This means relaxing
the tracker to sometimes produce only 1-3 players carries no downstream risk
in those three files.

## Goals

- Never block an entire analysis just because the tagged player's partner
  or opponents weren't both visible early in the clip.
- Still get full 4-player stats whenever the footage does show everyone
  (no regression for videos where the old all-4 requirement already worked).
- Let the user supply court geometry via line-point clicks instead of exact
  corner clicks, working correctly when a corner is off-screen but both its
  contributing lines have *some* visible stub.
- When an entire side of the court is never visible at all, say so clearly
  rather than guess.

## Non-goals

- General partial-homography estimation from fewer than 4 determined
  corners (e.g. deriving a court model from only 2-3 corners via some other
  mixed point/line least-squares solve). Out of scope — if fewer than 4
  corners can be determined via line intersection, manual entry fails with
  a clear message naming which corners are missing and why.
- Changing anything in `stats.py`/`report.py`/`validate.py` — confirmed
  unnecessary.

---

## Part A: Relaxed tracker initialization

### `choose_initial` / `Tracker.initialize`

Today, `choose_initial` requires `len(sides["near"]) >= 2 and len(sides["far"]) >= 2`
in a single frame or returns `None` (nothing is created). New behavior:

- If `me_px` is given: initialization succeeds on the first frame where some
  detection's bbox contains `me_px` (or is nearest to it, as today). Only
  that player's track (`P1`, `is_user=True`) is guaranteed. If, in that same
  frame, a second detection exists on `P1`'s side, it becomes `P2`
  (partner); if 2 detections exist on the opposite side, they become `P3`/
  `P4` (opponents) — exactly today's assignment logic, just no longer
  *required* for success.
- If `me_px` is `None`: initialization succeeds on the first frame with at
  least 1 detection on the near side (today's "P1 defaults to near-left"
  convention), with the same opportunistic P2/P3/P4 assignment as above.
- `track_players` in `pipeline.py` keeps its `init_search_s` timeout, but the
  `TrackingInitError` message changes to reflect the new requirement: "could
  not find your tagged player in the first Ns; check the capture framing"
  (or, when `me_px` is None, "could not find any player in the first Ns").

### Late-binding in `Tracker.update`

Once tracking has started with fewer than 2 players on a side, new players
should be able to join mid-clip rather than being permanently absent for the
whole analysis:

- Per team side, if `len(tracks_on_that_side) < 2`, detections on that side
  that don't match an existing track (i.e. weren't assigned by the existing
  Hungarian `_assign` call) become **candidates**.
- A candidate must be seen within a small movement gate (reuse
  `TrackingConfig.gate_margin_m`, no velocity prediction needed since there's
  no track yet) across a configurable number of consecutive analyzed frames
  (new `TrackingConfig.late_bind_confirm_frames`, default small, e.g. 3) —
  this specifically guards against the false-positive-blob problem the real
  diagnostic surfaced (a single-frame reflection or shadow must not become a
  phantom player).
- Once confirmed, the candidate is promoted to a real track, filling the
  next available `P2` (same side as `P1`) / `P3`/`P4` (opposite side, by
  x-position among whichever of those two slots remain, matching today's
  left-to-right convention) slot. Promotion starts the new track's points
  from the confirming frame onward — no retroactive backfill before that.
- If a side never confirms a second player for the whole clip, that slot
  simply never exists in `tracker.tracks` — already safe downstream per the
  investigation above.

### Config

New `TrackingConfig.late_bind_confirm_frames: int = 3`.

---

## Part B: Line-intersection manual corner entry

### Backend

New function in `court.py`, alongside the existing `COURT_LINES` dict (which
already names `near_baseline`, `far_baseline`, `left_sideline`,
`right_sideline`, plus the three service lines):

```
fit_corners_from_lines(clicked: dict[str, list[tuple[float,float]]]) -> dict[str, tuple[float,float] | None]
```

- `clicked` maps a subset of `{"near_baseline", "far_baseline",
  "left_sideline", "right_sideline"}` to 2+ clicked image points each (only
  these 4 — the two baselines and two sidelines — bound the court
  rectangle; service lines aren't needed for corner-finding and stay
  available separately for the existing auto-detector).
- For each provided line, fit an image-space line via `cv2.fitLine` (same
  primitive the auto-detector already uses).
- Compute each of the 4 corners as the intersection of its two adjacent
  lines (`near_left = near_baseline ∩ left_sideline`, etc.) whenever *both*
  contributing lines were provided; otherwise that corner is `None`.
- Return all 4 slots (some possibly `None`) plus, for any `None`, which
  line was missing — the webapp uses this to message clearly.

### Webapp / frontend

- Replace the "click corner 1 of 4" flow in `web/index.html` with: pick a
  line (button per line name, only lines with at least a visible stub need
  to be attempted), click 2+ points along it, mark it done, move to the next
  line or finish.
- On finish: call `fit_corners_from_lines`. If all 4 corners resolve, build
  the `CourtModel` exactly as today (`cv2.getPerspectiveTransform` from the
  4 points). If any are `None`, show which corners couldn't be determined
  and why (which line was missing) — do not silently proceed with a
  degraded/guessed model.

### CLI

`padelfusion analyze --corners` keeps accepting the existing literal
4-point format unchanged — this is an additive alternative entry path via
the webapp's manual-corners UI only, not a CLI change.

---

## Testing

- **Part A**: extend the synthetic match to a scenario where the partner/
  opponents don't appear simultaneously with `P1` early in the clip (e.g.
  delay one team's spawn), confirming today's all-4 requirement would fail
  init but the relaxed version succeeds and still produces `P1`'s solo
  stats. A second test confirms late-binding: a track that appears only
  after N confirm-frames gets a `P2`/`P3`/`P4` slot; one that appears for
  only 1-2 frames (below the confirm threshold) does not.
- **Part B**: unit tests for `fit_corners_from_lines` — full 4-line input
  recovers known corners (within tolerance) on a synthetic rendered court;
  a 3-line input (one baseline omitted) returns exactly the 2 corners that
  don't depend on the missing line as `None`, the other 2 correctly
  resolved.

## Risks

- Late-binding's confirm-frame threshold is a new tunable with no real-world
  validation yet (same caveat as the fisheye `k` thresholds) — flagged for
  the same real-footage follow-up pass.
- The line-intersection UI is a real frontend rework of the existing
  corner-picking flow in `web/index.html`, not just backend math — worth
  budgeting real UI iteration time, not just the geometry function.
