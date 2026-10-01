# Solo-player tracking init + line-intersection corners Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop blocking an entire analysis when only the user's own tagged player is reliably found early in a clip, and let manual court-corner entry work by clicking points along visible lines (with intersections computed) instead of requiring 4 exact corner clicks.

**Architecture:** `tracking.py`'s `Tracker` gains a relaxed, single-player-sufficient initialization path plus a confirm-frame "late-binding" mechanism in `update()` that promotes consistently-seen extra detections into P2/P3/P4 tracks as they appear, instead of requiring all 4 simultaneously up front. Separately, a new `court.fit_corners_from_lines` function computes each of the 4 court corners as the intersection of two fitted lines from arbitrary clicked points, wired through a new webapp endpoint and a reworked corner-picking step in `web/index.html`.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`linear_sum_assignment`, already used), OpenCV (`cv2.fitLine`), pytest, vanilla JS (no new frontend dependency).

## Global Constraints

- `stats.py`, `report.py`, `validate.py` already iterate dynamically over however many players exist in `timeline.players` (confirmed by investigation) — no changes needed there, and any test producing fewer than 4 tracked players is safe to rely on this.
- Existing all-4-simultaneous-at-init videos must keep working identically — `test_tracker_team_constraint_and_ids` and the full synthetic end-to-end tests must keep passing unchanged.
- No new third-party dependency.
- `CourtModel`'s corner order convention (`near_left, near_right, far_right, far_left`) is used consistently everywhere already — new code follows it, doesn't redefine it.

---

### Task 1: `TrackingConfig.late_bind_confirm_frames`

**Files:**
- Modify: `src/padelfusion/config.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `TrackingConfig.late_bind_confirm_frames: int = 3`

- [ ] **Step 1: Write the failing test**

```python
def test_tracking_config_has_late_bind_setting():
    assert TrackingConfig().late_bind_confirm_frames == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_tracking_config_has_late_bind_setting -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add the field**

In `src/padelfusion/config.py`, in `TrackingConfig`:

```python
@dataclass
class TrackingConfig:
    analysis_fps: float = 15.0
    max_speed_mps: float = 7.5
    gate_margin_m: float = 1.2
    appearance_weight: float = 2.0
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_tracking_config_has_late_bind_setting -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/config.py tests/test_units.py
git commit -m "feat: add TrackingConfig.late_bind_confirm_frames"
```

---

### Task 2: Relax `Tracker.initialize` to only require the tagged player

**Files:**
- Modify: `src/padelfusion/tracking.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `Tracker.initialize(t, cdets, me_px) -> bool` — unchanged signature, relaxed requirement. `Tracker.my_team: str | None` (new attribute, set on successful init).
- Removes: `choose_initial` (inlined into `initialize`; not used elsewhere per current imports).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_units.py` near the existing tracker tests:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "initializes_with_only or initializes_with_all_four or initialize_fails" -v`
Expected: FAIL — `test_tracker_initializes_with_only_the_tagged_player` fails because `choose_initial` requires 2+ per side; the other two should already pass (confirm this, they're regression guards).

- [ ] **Step 3: Rewrite `initialize`**

In `src/padelfusion/tracking.py`, replace `choose_initial` and `Tracker.initialize`:

```python
def _side_candidates(cdets: list[CourtDetection], side: str) -> list[CourtDetection]:
    cands = [c for c in cdets if team_side(c.xy[1]) == side]
    return sorted(cands, key=lambda c: (c.det.score, c.det.size[1]), reverse=True)[:2]


class Tracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.tracks: dict[str, Track] = {}
        self.sample_times: list[float] = []
        self.my_team: str | None = None
        self.pending: dict[str, list[dict]] = {"near": [], "far": []}

    # -- initialisation ---------------------------------------------------
    def initialize(self, t: float, cdets: list[CourtDetection], me_px: tuple[float, float] | None) -> bool:
        near = _side_candidates(cdets, "near")
        far = _side_candidates(cdets, "far")
        all_c = near + far
        if not all_c:
            return False
        if me_px is not None:
            def dist_to_me(c: CourtDetection) -> float:
                x1, y1, x2, y2 = c.det.bbox
                if x1 <= me_px[0] <= x2 and y1 <= me_px[1] <= y2:
                    return 0.0
                return float(np.hypot(c.det.foot[0] - me_px[0], c.det.foot[1] - me_px[1]))

            me = min(all_c, key=dist_to_me)
        else:
            if not near:
                return False
            me = min(near, key=lambda c: c.xy[0])  # default: near-left player
        my_team = team_side(me.xy[1])
        self.my_team = my_team
        opp_team = "far" if my_team == "near" else "near"
        sides = {"near": near, "far": far}
        partner = next((c for c in sides[my_team] if c is not me), None)
        opps = sorted(sides[opp_team], key=lambda c: c.xy[0])
        assignments = [("P1", me, my_team, True)]
        if partner is not None:
            assignments.append(("P2", partner, my_team, False))
        if len(opps) >= 1:
            assignments.append(("P3", opps[0], opp_team, False))
        if len(opps) >= 2:
            assignments.append(("P4", opps[1], opp_team, False))
        for pid, c, team, is_user in assignments:
            tr = Track(pid, team, is_user, c.det.hist.copy())
            tr.points.append(TrackPoint(t, float(c.xy[0]), float(c.xy[1]), c.det.bbox))
            self.tracks[pid] = tr
        self.sample_times.append(t)
        return True
```

Leave `update`, `_cost`, `_assign`, `finalize` untouched for this task (Task 3 changes `_assign`/`update`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "tracker" -v`
Expected: all PASS, including the pre-existing `test_tracker_team_constraint_and_ids` and `test_tracker_flags_ambiguous_partners` (regression check — both already pass 2-per-side detections at init, so the relaxed logic must still assign all 4 identically).

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/tracking.py tests/test_units.py
git commit -m "feat: relax Tracker.initialize to only require the tagged player"
```

---

### Task 3: Late-binding for players that appear after initialization

**Files:**
- Modify: `src/padelfusion/tracking.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Consumes: `TrackingConfig.late_bind_confirm_frames` (Task 1), `Tracker.my_team`/`Tracker.pending` (Task 2)
- Produces: `Tracker.update(t, cdets) -> None` — unchanged signature; now creates new tracks mid-clip when warranted. `Tracker._assign` now returns `list[CourtDetection]` (the unmatched ones) instead of `None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_tracker_late_binds_a_consistently_seen_second_player():
    h = [np.eye(32)[i].astype(np.float32) for i in range(4)]
    cfg = TrackingConfig(late_bind_confirm_frames=3)
    tr = Tracker(cfg)
    assert tr.initialize(0.0, [_cd(2, 2, h[0])], me_px=(2, 2))
    assert set(tr.tracks) == {"P1"}
    for k in range(1, 3):  # 2 more consecutive frames -> streak reaches 3
        tr.update(k / 15, [_cd(2 + 0.01 * k, 2, h[0]), _cd(8, 5, h[1])])
        if k < 2:
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "late_binds or late_bind_a_single" -v`
Expected: FAIL (no P2 ever appears — `update` doesn't create tracks yet)

- [ ] **Step 3: Implement late-binding**

In `src/padelfusion/tracking.py`, change `_assign` to return the unmatched detections, and add `_update_candidates`/`_next_pid`, then update `update`:

```python
    def _assign(self, t: float, tracks: list[Track], tds: list[CourtDetection]) -> list[CourtDetection]:
        if not tracks or not tds:
            return list(tds)
        cost = np.array([[self._cost(tr, c, t) for c in tds] for tr in tracks])
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < BIG]
        flags: dict[int, list[str]] = {r: [] for r, _ in matches}

        if len(matches) == 2:
            (r0, c0), (r1, c1) = matches
            best = cost[r0, c0] + cost[r1, c1]
            alt = cost[r0, c1] + cost[r1, c0]
            if alt < BIG and alt - best < self.cfg.ambiguity_margin:
                flags[r0].append("ambiguous_id")
                flags[r1].append("ambiguous_id")
            if np.linalg.norm(tds[c0].xy - tds[c1].xy) < self.cfg.close_partner_m:
                flags[r0].append("close_partners")
                flags[r1].append("close_partners")
        elif len(matches) == 1 and len(tracks) == 2:
            r, c = matches[0]
            other = tracks[1 - r]
            if np.linalg.norm(other.predict(t) - tds[c].xy) < 1.5 * self.cfg.close_partner_m:
                flags[r].append("merged_detection")

        for r, c in matches:
            tr, cd = tracks[r], tds[c]
            dt = max(t - tr.last.t, 1e-3)
            new_vel = (cd.xy - np.array([tr.last.x, tr.last.y])) / dt
            tr.vel = 0.6 * tr.vel + 0.4 * new_vel if dt < 0.5 else np.zeros(2)
            tr.points.append(TrackPoint(t, float(cd.xy[0]), float(cd.xy[1]), cd.det.bbox, flags[r]))
            if not flags[r]:
                tr.hist = 0.9 * tr.hist + 0.1 * cd.det.hist

        matched_cols = {c for _, c in matches}
        return [tds[c] for c in range(len(tds)) if c not in matched_cols]

    def _next_pid(self, team: str) -> str | None:
        candidates = ("P1", "P2") if team == self.my_team else ("P3", "P4")
        for pid in candidates:
            if pid not in self.tracks:
                return pid
        return None

    def _update_candidates(self, t: float, team: str, unmatched: list[CourtDetection], n_tracks: int) -> None:
        pend = self.pending[team]
        matched_idx: set[int] = set()
        for cand in pend:
            best_i, best_d = None, self.cfg.gate_margin_m
            for i, c in enumerate(unmatched):
                if i in matched_idx:
                    continue
                d = float(np.linalg.norm(cand["xy"] - c.xy))
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                matched_idx.add(best_i)
                cand["xy"] = unmatched[best_i].xy
                cand["det"] = unmatched[best_i].det
                cand["streak"] += 1
            else:
                cand["streak"] = 0
        self.pending[team] = [c for c in pend if c["streak"] > 0]
        for i, c in enumerate(unmatched):
            if i not in matched_idx:
                self.pending[team].append({"xy": c.xy, "det": c.det, "streak": 1})

        for cand in list(self.pending[team]):
            if n_tracks >= 2:
                break
            if cand["streak"] >= self.cfg.late_bind_confirm_frames:
                pid = self._next_pid(team)
                if pid is None:
                    continue
                tr = Track(pid, team, False, cand["det"].hist.copy())
                tr.points.append(TrackPoint(t, float(cand["xy"][0]), float(cand["xy"][1]), cand["det"].bbox))
                self.tracks[pid] = tr
                self.pending[team].remove(cand)
                n_tracks += 1

    # -- per-frame update -------------------------------------------------
    def update(self, t: float, cdets: list[CourtDetection]) -> None:
        self.sample_times.append(t)
        for team in ("near", "far"):
            tracks = [tr for tr in self.tracks.values() if tr.team == team]
            tds = [c for c in cdets if team_side(c.xy[1]) == team]
            unmatched = self._assign(t, tracks, tds)
            if len(tracks) < 2:
                self._update_candidates(t, team, unmatched, len(tracks))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "tracker" -v`
Expected: all PASS, including both new tests and every pre-existing tracker test (`test_tracker_team_constraint_and_ids`, `test_tracker_flags_ambiguous_partners`, plus Task 2's three tests).

Also run the full unit suite:

Run: `pytest tests/test_units.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/tracking.py tests/test_units.py
git commit -m "feat: late-bind consistently-seen extra players after initialization"
```

---

### Task 4: Update `track_players`'s error message and `init_search_s` behavior

**Files:**
- Modify: `src/padelfusion/pipeline.py`
- Test: `tests/test_units.py`

**Interfaces:**
- No signature change — `track_players` still raises `TrackingInitError` when nothing is ever found, with an updated message reflecting the relaxed requirement.

- [ ] **Step 1: Write the failing test**

```python
def test_track_players_message_reflects_relaxed_requirement():
    from padelfusion.pipeline import TrackingInitError, track_players
    import inspect

    src = inspect.getsource(track_players)
    assert "two players on each side" not in src
    assert "your tagged player" in src
```

This is a light source-level check (rather than a slow full-video test) confirming the message was actually updated, since a full-video test of this exact path is already covered by existing end-to-end tests and by Tasks 2-3's direct `Tracker` tests.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_units.py::test_track_players_message_reflects_relaxed_requirement -v`
Expected: FAIL (old message still present)

- [ ] **Step 3: Update the message**

In `src/padelfusion/pipeline.py`, in `track_players`:

```python
        if not initialized:
            if t > tcfg.init_search_s:
                raise TrackingInitError(
                    f"could not find your tagged player in the first {tcfg.init_search_s:.0f}s; "
                    f"check the capture framing")
            initialized = tracker.initialize(t, cdets, me_px)
            continue
```

And the final guard:

```python
    if not initialized:
        raise TrackingInitError("no frames with any player found")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_units.py::test_track_players_message_reflects_relaxed_requirement -v`
Expected: PASS

Also run the full suite:

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Update the webapp's error hint**

In `src/padelfusion/web/index.html`, the `hint()` function currently matches on `"two players"`:

```js
function hint(err) {
  err = err || "";
  if (err.includes("court")) return `<p class="muted">Tip: go back and use <b>Set corners by hand</b>: click the four court corners on the frame.</p>`;
  if (err.includes("tagged player")) return `<p class="muted">Tip: make sure the player you tapped as yourself is clearly visible early in the clip.</p>`;
  if (err.includes("rejected")) return `<p class="muted">Tip: go back and tick <b>Analyze anyway</b> to try regardless.</p>`;
  return "";
}
```

- [ ] **Step 6: Commit**

```bash
git add src/padelfusion/pipeline.py src/padelfusion/web/index.html tests/test_units.py
git commit -m "feat: update tracking-init messaging for the relaxed requirement"
```

---

### Task 5: `court.fit_corners_from_lines`

**Files:**
- Modify: `src/padelfusion/court.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Produces: `fit_corners_from_lines(clicked: dict[str, list[list[float]]]) -> dict` — returns `{"corners": {name: [x, y] | None for name in ("near_left","near_right","far_right","far_left")}, "missing": {name: reason}}`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_units.py -k "fit_corners_from_lines" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement it**

In `src/padelfusion/court.py`, add near `far_field_bbox`:

```python
CORNER_ADJACENCY = {
    "near_left": ("near_baseline", "left_sideline"),
    "near_right": ("near_baseline", "right_sideline"),
    "far_right": ("far_baseline", "right_sideline"),
    "far_left": ("far_baseline", "left_sideline"),
}


def _fit_image_line(pts: np.ndarray):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return float(vx), float(vy), float(x0), float(y0)


def _intersect_lines(a, b) -> tuple[float, float] | None:
    vx1, vy1, x1, y1 = a
    vx2, vy2, x2, y2 = b
    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < 1e-9:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
    return (x1 + t * vx1, y1 + t * vy1)


def fit_corners_from_lines(clicked: dict[str, list]) -> dict:
    """Compute the 4 court corners as intersections of fitted lines from
    clicked points along any subset of the 4 boundary lines (near/far
    baseline, left/right sideline). A corner is None when either of its two
    contributing lines wasn't given (or ended up parallel in the image)."""
    fits = {}
    for name, pts in clicked.items():
        if name not in ("near_baseline", "far_baseline", "left_sideline", "right_sideline"):
            continue
        arr = np.asarray(pts, dtype=np.float64)
        if arr.shape[0] < 2:
            continue
        fits[name] = _fit_image_line(arr)

    corners: dict[str, list[float] | None] = {}
    missing: dict[str, str] = {}
    for corner, (a_name, b_name) in CORNER_ADJACENCY.items():
        if a_name in fits and b_name in fits:
            p = _intersect_lines(fits[a_name], fits[b_name])
            if p is None:
                corners[corner] = None
                missing[corner] = f"{a_name} and {b_name} look parallel in the image"
            else:
                corners[corner] = [p[0], p[1]]
        else:
            corners[corner] = None
            need = [n for n in (a_name, b_name) if n not in fits]
            missing[corner] = "missing points along: " + ", ".join(need)
    return {"corners": corners, "missing": missing}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_units.py -k "fit_corners_from_lines" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/court.py tests/test_units.py
git commit -m "feat: compute court corners from clicked lines via intersection"
```

---

### Task 6: `/court/from_lines` webapp endpoint

**Files:**
- Modify: `src/padelfusion/webapp.py`
- Test: `tests/test_webapp.py`

**Interfaces:**
- Produces: `POST /api/runs/{id}/court/from_lines` with body `{"lines": {...}}` → JSON `{"corners": {...}, "missing": {...}}` (the return value of `fit_corners_from_lines`, unchanged).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_webapp.py`:

```python
def test_court_from_lines_endpoint(server):
    base, video = server
    status, body = call(base, "/api/upload?filename=lines.mp4", "PUT", raw=video.read_bytes())
    run = json.loads(body)
    from padelfusion.synth import image_corners

    c = image_corners(1280, 720)
    req = {"lines": {
        "near_baseline": [c[0].tolist(), c[1].tolist()],
        "right_sideline": [c[1].tolist(), c[2].tolist()],
        "far_baseline": [c[2].tolist(), c[3].tolist()],
        "left_sideline": [c[3].tolist(), c[0].tolist()],
    }}
    result = json.loads(call(base, f"/api/runs/{run['id']}/court/from_lines", "POST", body=req)[1])
    assert result["missing"] == {}
    assert all(result["corners"].values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webapp.py::test_court_from_lines_endpoint -v`
Expected: FAIL with 404 (`unknown endpoint`)

- [ ] **Step 3: Add the endpoint**

In `src/padelfusion/webapp.py`, add the import (`from .court import COURT_LINES, CourtModel, detect_court, fit_corners_from_lines`) and add, right after the existing `court/preview` handler in `_api`:

```python
            if rest == ["court", "preview"] and method == "POST":
                return self._json({"lines": court_lines(self._body()["corners"])})
            if rest == ["court", "from_lines"] and method == "POST":
                return self._json(fit_corners_from_lines(self._body()["lines"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_webapp.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/padelfusion/webapp.py tests/test_webapp.py
git commit -m "feat: add /court/from_lines endpoint"
```

---

### Task 7: Rework the manual corner-picking UI to click lines, not corners

**Files:**
- Modify: `src/padelfusion/web/index.html`

**Interfaces:**
- No backend interface change — consumes the Task 6 endpoint. This task has no automated test (frontend JS isn't covered by the pytest suite); verify manually via the running webapp after this change.

- [ ] **Step 1: Replace the `setupView` function**

In `src/padelfusion/web/index.html`, replace the entire `setupView` function (from `async function setupView(run) {` to its closing `}`) with:

```javascript
async function setupView(run) {
  const id = run.id, opts = run.options || {};
  const st = {me: opts.me || null, corners: opts.corners || null, auto: null, mode: "me",
             linePts: {}, lineName: null, missing: null, lines: null, manualOn: !!opts.corners};
  const info = run.info, errs = run.capture_errors || [], warns = run.capture_warnings || [];
  $("#app").innerHTML = `<h1>${esc(run.name)}</h1>
  <div class="card"><h2>Video check</h2>
    <p class="muted small">${info.width}×${info.height} · ${info.fps.toFixed(1)} fps · ${(info.duration_s / 60).toFixed(1)} min · audio ${info.has_audio ? "yes" : "no"}</p>
    ${errs.length ? `<ul class="bad">${errs.map(e => `<li>${esc(e)}</li>`).join("")}</ul>
      <label class="row"><input type="checkbox" id="anyway"> Analyze anyway (numbers may be unreliable)</label>`
      : `<p class="ok">✓ Meets the capture spec.</p>`}
    ${warns.map(w => `<p class="warn small">⚠ ${esc(w)}</p>`).join("")}
  </div>
  <div class="card">
    <div class="row" style="justify-content:space-between"><h2 style="margin:0">Set up the first frame</h2>
      <div class="seg" id="modes"><button data-m="me" class="on">1 · Tap yourself</button><button data-m="corners">2 · Court corners</button></div></div>
    <p id="instr" class="small" style="margin:10px 0"></p>
    <p id="missing" class="bad small"></p>
    <div class="canvaswrap"><canvas id="cv"></canvas></div>
    <div class="row" style="margin-top:10px" id="courtbtns"></div>
    <div class="row" style="margin-top:8px" id="linebtns"></div>
  </div>
  <div class="card"><h2>Options</h2>
    <div class="row" style="margin-bottom:8px"><label>Player detector
      <select id="det"><option value="auto">Automatic</option>
      <option value="yolo" ${ENV.yolo ? "" : "disabled"}>YOLO (best for real video)${ENV.yolo ? "" : ", not installed"}</option>
      <option value="background">Background subtraction (fast)</option></select></label></div>
    <label class="row"><input type="checkbox" id="llm" ${ENV.anthropic_key ? "checked" : "disabled"}> Coaching summary by Claude
      ${ENV.anthropic_key ? "" : `<span class="muted small">(add an API key on the home page)</span>`}</label>
    <label class="row"><input type="checkbox" id="clips" ${opts.clips ? "checked" : ""}> Save a video clip for every rally (slower)</label>
  </div>
  <div class="row"><button class="primary big" id="go">Analyze match</button><span class="muted small" id="gonote"></span></div>`;
  if (opts.detector) $("#det").value = opts.detector;

  const cv = $("#cv"), ctx = cv.getContext("2d"), img = new Image();
  img.src = `/files/${id}/frame0.jpg?${Date.now()}`;
  await new Promise(r => img.onload = r);
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  const LINE_DEFS = [["near_baseline", "Near baseline"], ["far_baseline", "Far baseline"],
    ["left_sideline", "Left sideline"], ["right_sideline", "Right sideline"]];
  const LINE_COLORS = {near_baseline: "#ffcc00", far_baseline: "#ff8a00", left_sideline: "#00d1ff", right_sideline: "#7dff4d"};

  function draw() {
    ctx.drawImage(img, 0, 0);
    const lw = Math.max(2, cv.width / 500);
    if (st.lines) { ctx.strokeStyle = "rgba(0,255,170,.95)"; ctx.lineWidth = lw;
      st.lines.forEach(pl => { ctx.beginPath(); pl.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); }); }
    Object.entries(st.linePts).forEach(([name, pts]) => {
      ctx.fillStyle = LINE_COLORS[name] || "#ffcc00";
      pts.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, lw * 3, 0, 7); ctx.fill(); });
    });
    if (st.me) { const [x, y] = st.me; ctx.strokeStyle = "#ff3b6b"; ctx.lineWidth = lw * 1.5;
      ctx.beginPath(); ctx.arc(x, y, lw * 9, 0, 7); ctx.stroke();
      ctx.fillStyle = "#ff3b6b"; ctx.font = `bold ${lw * 10}px system-ui`; ctx.fillText("ME", x + lw * 11, y); }
  }
  function refresh() {
    draw();
    document.querySelectorAll("#modes button").forEach(b => b.classList.toggle("on", b.dataset.m === st.mode));
    const court = st.corners ? (st.auto && st.corners === st.auto.corners ? "Court found automatically." : "Court corners set by hand.") : "";
    if (st.mode === "me") $("#instr").innerHTML = st.me ? `✓ You're marked. Click again to move the marker. Then check the court in step 2.`
      : `<b>Click on yourself</b> in the frame, so the app knows which player is you.`;
    else if (st.manualOn) $("#instr").innerHTML = st.lineName
      ? `Click 2 or more points along the <b>${LINE_DEFS.find(l => l[0] === st.lineName)[1]}</b>, anywhere it's visible. (${(st.linePts[st.lineName] || []).length} placed)`
      : `Pick a line below, then click points along it. Only the lines you can actually see need any points — click <b>Compute corners</b> when done.`;
    else $("#instr").innerHTML = st.corners ? `✓ ${court} Check that the green lines sit on the court's painted lines and walls.`
      : `The court wasn't found automatically. Click <b>Set corners by hand</b>.`;
    $("#missing").textContent = st.missing
      ? "Couldn't determine: " + Object.entries(st.missing).map(([c, r]) => `${c} (${r})`).join("; ")
      : "";
    $("#courtbtns").innerHTML = st.mode === "corners" ? `<button id="auto">Detect court automatically</button>
      <button id="manual">Set corners by hand</button>
      ${st.manualOn ? `<button id="compute">Compute corners</button><button id="undo">Undo point</button>` : ""}` : "";
    $("#linebtns").innerHTML = (st.mode === "corners" && st.manualOn) ? LINE_DEFS.map(([name, label]) =>
      `<button data-line="${name}" class="${st.lineName === name ? "on" : ""}" style="border-color:${LINE_COLORS[name]}">
        ${label} (${(st.linePts[name] || []).length})</button>`).join("") : "";
    if ($("#auto")) $("#auto").onclick = detect;
    if ($("#manual")) $("#manual").onclick = () => { st.linePts = {}; st.lineName = null; st.corners = null; st.lines = null;
      st.missing = null; st.mode = "corners"; st.manualOn = true; refresh(); };
    if ($("#compute")) $("#compute").onclick = computeCorners;
    if ($("#undo")) $("#undo").onclick = () => { if (st.lineName && (st.linePts[st.lineName] || []).length)
      st.linePts[st.lineName].pop(); st.corners = null; st.lines = null; refresh(); };
    document.querySelectorAll("#linebtns [data-line]").forEach(b => b.onclick = () => { st.lineName = b.dataset.line; refresh(); });
    $("#go").disabled = !st.corners;
    $("#gonote").textContent = !st.corners ? "Set the court first (step 2)." : !st.me ? "Tip: tap yourself first, or P1 will be the near-left player." : "";
  }
  async function detect() {
    $("#instr").textContent = "Looking for the court lines…";
    const r = await api(`runs/${id}/court/detect`, {method: "POST"});
    st.linePts = {}; st.lineName = null; st.manualOn = false; st.missing = null;
    if (r.found) { st.auto = r; st.corners = r.corners; st.lines = r.lines; toast(`Court found (line score ${r.score})`); }
    else { st.corners = null; st.lines = null; toast("Court not found automatically"); }
    refresh();
  }
  async function computeCorners() {
    const r = await api(`runs/${id}/court/from_lines`, {method: "POST", body: {lines: st.linePts}});
    const order = ["near_left", "near_right", "far_right", "far_left"];
    if (order.every(k => r.corners[k])) {
      st.corners = order.map(k => r.corners[k]);
      st.missing = null;
      st.lines = (await api(`runs/${id}/court/preview`, {method: "POST", body: {corners: st.corners}})).lines;
      toast("Corners computed");
    } else {
      st.corners = null;
      st.missing = r.missing;
      toast("Some corners couldn't be determined — see details below");
    }
    refresh();
  }
  cv.onclick = e => {
    const b = cv.getBoundingClientRect();
    const p = [(e.clientX - b.left) * cv.width / b.width, (e.clientY - b.top) * cv.height / b.height].map(v => Math.round(v));
    if (st.mode === "me") { st.me = p; refresh(); if (!st.corners) { st.mode = "corners"; refresh(); } return; }
    if (!st.manualOn) return toast("Click “Set corners by hand” to place points");
    if (!st.lineName) return toast("Pick a line first");
    (st.linePts[st.lineName] = st.linePts[st.lineName] || []).push(p);
    st.corners = null;
    refresh();
  };
  document.querySelectorAll("#modes button").forEach(b => b.onclick = () => { st.mode = b.dataset.m; refresh(); });
  $("#go").onclick = async () => {
    if (errs.length && !($("#anyway") && $("#anyway").checked)) return toast("The video fails the capture check. Tick “Analyze anyway” to continue.");
    await api(`runs/${id}/analyze`, {method: "POST", body: {me: st.me, corners: st.manualOn ? st.corners : null,
      detector: $("#det").value, llm: $("#llm").checked, clips: $("#clips").checked, skip_capture_check: !!($("#anyway") && $("#anyway").checked)}});
    runView(id);
  };
  refresh();
  if (st.corners) st.lines = (await api(`runs/${id}/court/preview`, {method: "POST", body: {corners: st.corners}})).lines, refresh();
  else detect();
}
```

- [ ] **Step 2: Manually verify in the running app**

Restart the webapp (`python -m padelfusion ui`), upload or open a run that needs manual corners, go to step 2, click **Set corners by hand**, select a line, click 2+ points along it (repeat for at least 2 adjacent lines, e.g. `near_baseline` + `left_sideline`), click **Compute corners**, and confirm:
- with all 4 lines given: all 4 corners resolve and the green overlay matches the court
- with only 3 lines given (skip one baseline): exactly the 2 corners that don't depend on the missing line resolve; the "Couldn't determine" message names the other 2 and why

- [ ] **Step 3: Run the full automated suite to confirm no regression**

Run: `pytest tests/ -v`
Expected: all PASS (this task touches no Python code, so this is a pure regression guard)

- [ ] **Step 4: Commit**

```bash
git add src/padelfusion/web/index.html
git commit -m "feat: pick court corners via line intersection instead of exact clicks"
```
