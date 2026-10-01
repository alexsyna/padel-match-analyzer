"""Court model: image <-> court coordinates, zones, automatic line detection.

Court frame (metres), as seen from the camera behind the near back glass:
  x: 0 (left sideline) .. 10 (right sideline)
  y: 0 (near baseline, camera side) .. 20 (far baseline); net at y = 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

COURT_WIDTH = 10.0
COURT_LENGTH = 20.0
NET_Y = 10.0
SERVICE_LINE_FROM_NET = 6.95

# Corner order used everywhere: near-left, near-right, far-right, far-left.
COURT_CORNERS = np.array(
    [[0.0, 0.0], [COURT_WIDTH, 0.0], [COURT_WIDTH, COURT_LENGTH], [0.0, COURT_LENGTH]],
    dtype=np.float64,
)

# Painted lines as (start, end) segments in court metres.
COURT_LINES = {
    "near_baseline": ((0, 0), (10, 0)),
    "far_baseline": ((0, 20), (10, 20)),
    "left_sideline": ((0, 0), (0, 20)),
    "right_sideline": ((10, 0), (10, 20)),
    "near_service": ((0, NET_Y - SERVICE_LINE_FROM_NET), (10, NET_Y - SERVICE_LINE_FROM_NET)),
    "far_service": ((0, NET_Y + SERVICE_LINE_FROM_NET), (10, NET_Y + SERVICE_LINE_FROM_NET)),
    "centre_service": ((5, NET_Y - SERVICE_LINE_FROM_NET), (5, NET_Y + SERVICE_LINE_FROM_NET)),
}


@dataclass
class CourtModel:
    corners_img: np.ndarray  # (4, 2) pixel coords, order near-left, near-right, far-right, far-left
    source: str = "manual"  # "auto" | "manual"
    score: float | None = None  # line-support score for auto detection
    k: float = 0.0  # distortion.py's self-calibrated division-model coefficient; 0.0 = no correction

    def __post_init__(self) -> None:
        self.corners_img = np.asarray(self.corners_img, dtype=np.float64).reshape(4, 2)
        self.H = cv2.getPerspectiveTransform(
            self.corners_img.astype(np.float32), COURT_CORNERS.astype(np.float32)
        )
        self.H_inv = np.linalg.inv(self.H)

    def image_to_court(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def court_to_image(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "corners_img": self.corners_img.tolist(),
            "corner_order": ["near_left", "near_right", "far_right", "far_left"],
            "source": self.source,
            "score": self.score,
            "distortion_k": self.k,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CourtModel":
        return cls(np.array(d["corners_img"]), d.get("source", "manual"), d.get("score"),
                   d.get("distortion_k", 0.0))


def team_side(y: float) -> str:
    return "near" if y < NET_Y else "far"


def zone(y: float, net_zone_depth_m: float) -> str:
    return "net" if abs(y - NET_Y) <= net_zone_depth_m else "back"


def in_court(x: float, y: float, margin: float) -> bool:
    return -margin <= x <= COURT_WIDTH + margin and -margin <= y <= COURT_LENGTH + margin


def px_per_m(court: CourtModel, y: float) -> float:
    """In-plane pixel scale at court depth `y`: how many pixels wide 1 metre
    is at that depth. Used as a stand-in for vertical scale too (the camera
    isn't looking straight down), the same approximation `synth.py` already
    uses to draw players of a roughly-correct size."""
    p = court.court_to_image([[0, y], [10, y]])
    return float(np.linalg.norm(p[1] - p[0]) / 10.0)


def expected_player_height_px(court: CourtModel, y: float, real_height_m: float = 1.7) -> float:
    return real_height_m * px_per_m(court, y)


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


def far_field_bbox(court: CourtModel, frame_shape, margin_m: float) -> tuple[int, int, int, int] | None:
    """Pixel bounding box of the far half of the court (beyond the net),
    clipped to the frame. None if the projected region is degenerate."""
    h, w = frame_shape[:2]
    poly = court.court_to_image([
        [-margin_m, NET_Y], [COURT_WIDTH + margin_m, NET_Y],
        [COURT_WIDTH + margin_m, COURT_LENGTH + margin_m], [-margin_m, COURT_LENGTH + margin_m],
    ])
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def parse_corners(text: str) -> np.ndarray:
    """Parse "x1,y1;x2,y2;x3,y3;x4,y4" (near-left, near-right, far-right, far-left)."""
    pts = [tuple(float(v) for v in p.split(",")) for p in text.strip().split(";") if p.strip()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise ValueError("corners must be four 'x,y' pairs separated by ';'")
    return np.array(pts, dtype=np.float64)


# --------------------------------------------------------------------------
# Automatic detection
# --------------------------------------------------------------------------

def line_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of thin bright (white) structures, i.e. painted lines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = max(9, (min(frame.shape[:2]) // 60) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    whiteish = (hsv[..., 1] < 70) & (hsv[..., 2] > 150)
    mask = ((tophat > 40) & whiteish).astype(np.uint8) * 255
    return mask


def _cluster_lines(segs, key, tol):
    """Group segments whose `key` value is within `tol`; return length-weighted fits."""
    if not segs:
        return []
    segs = sorted(segs, key=key)
    groups, cur = [], [segs[0]]
    for s in segs[1:]:
        if key(s) - key(cur[-1]) <= tol:
            cur.append(s)
        else:
            groups.append(cur)
            cur = [s]
    groups.append(cur)
    out = []
    for g in groups:
        pts = []
        for x1, y1, x2, y2 in g:
            n = max(2, int(np.hypot(x2 - x1, y2 - y1) / 10))
            pts.extend(zip(np.linspace(x1, x2, n), np.linspace(y1, y2, n)))
        pts = np.array(pts, dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        length = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in g)
        out.append({"line": np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0]), "length": length,
                    "key": float(np.mean([key(s) for s in g]))})
    return out


# Padel courts paint only the service lines and the centre service line. The
# perimeter is the glass and fence. Detection is therefore anchored on the two
# service lines: they run wall to wall, so their endpoints are four known
# court points.
PAINTED_LINES = ("near_service", "far_service", "centre_service")
SERVICE_RECT = np.array(
    [[0.0, NET_Y - SERVICE_LINE_FROM_NET], [COURT_WIDTH, NET_Y - SERVICE_LINE_FROM_NET],
     [COURT_WIDTH, NET_Y + SERVICE_LINE_FROM_NET], [0.0, NET_Y + SERVICE_LINE_FROM_NET]],
    dtype=np.float64,
)


def score_court(model: CourtModel, mask: np.ndarray, lines=PAINTED_LINES) -> float:
    """Line support: the fraction of projected painted-line samples that fall
    on white pixels. The centre line counts double, because it is what tells
    a real service-line pair apart from, say, the net tape."""
    h, w = mask.shape
    fat = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    hits = total = 0.0
    for name in lines:
        a, b = COURT_LINES[name]
        weight = 2.0 if name == "centre_service" else 1.0
        ts = np.linspace(0, 1, 80)
        img_pts = model.court_to_image(np.outer(1 - ts, a) + np.outer(ts, b))
        for x, y in img_pts:
            xi, yi = int(round(x)), int(round(y))
            total += weight
            if 0 <= xi < w and 0 <= yi < h and fat[yi, xi]:
                hits += weight
    support = hits / max(total, 1.0)
    # The centre line stops at the service lines. White beyond them means the
    # candidate pair is wrong (typically the net tape taken for a service line).
    lo, hi = NET_Y - SERVICE_LINE_FROM_NET, NET_Y + SERVICE_LINE_FROM_NET
    over = 0
    ext = [(5.0, y) for y in np.linspace(hi + 0.3, hi + 1.8, 20)] + \
          [(5.0, y) for y in np.linspace(lo - 1.8, lo - 0.3, 20)]
    for x, y in model.court_to_image(ext):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h and fat[yi, xi]:
            over += 1
    return support - over / len(ext)


def _line_extent(line, mask: np.ndarray, max_gap_frac: float = 0.05):
    """Endpoints of the longest white run along a near-horizontal line.
    Gaps (players standing on the line) shorter than max_gap_frac * width are bridged."""
    h, w = mask.shape
    a, b, c = line
    if abs(b) < 1e-9:
        return None
    xs = np.arange(w, dtype=np.float64)
    ys = -(a * xs + c) / b
    ok = (ys >= 0) & (ys < h)
    on = np.zeros(w, bool)
    on[ok] = mask[ys[ok].astype(int), xs[ok].astype(int)] > 0
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return None
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap_frac * w:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    x0, x1 = max(runs, key=lambda r: r[1] - r[0])
    if x1 - x0 < 0.12 * w:
        return None
    return np.array([x0, ys[x0]]), np.array([x1, ys[x1]])


def horizontal_line_segments(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Near-horizontal Hough segments from a line mask, as (x1, y1, x2, y2)."""
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60,
                           minLineLength=min(h, w) // 10, maxLineGap=min(h, w) // 40)
    if segs is None:
        return []
    out = []
    for s in np.asarray(segs).reshape(-1, 4):
        s = tuple(map(float, s))
        ang = np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180
        if ang < 10 or ang > 170:
            out.append(s)
    return out


def detect_court(frame: np.ndarray, min_score: float = 0.55) -> CourtModel | None:
    """Find the court from its painted service lines.

    Returns None when no candidate reaches `min_score`. The caller then falls
    back to manual corners.
    """
    h, w = frame.shape[:2]
    mask = line_mask(frame)
    fat = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    horiz = horizontal_line_segments(mask)
    hlines = _cluster_lines(horiz, lambda s: (s[1] + s[3]) / 2, h * 0.012)
    hlines = sorted(hlines, key=lambda c: c["length"], reverse=True)[:8]
    extents = []
    for c in hlines:
        e = _line_extent(c["line"], fat)
        if e is not None:
            extents.append((c, e))
    best, best_score = None, -1.0
    for (c1, e1), (c2, e2) in combinations(extents, 2):
        far, near = (e1, e2) if e1[0][1] < e2[0][1] else (e2, e1)
        if near[1][0] - near[0][0] <= far[1][0] - far[0][0]:
            continue  # the near service line must look wider than the far one
        img_pts = np.array([near[0], near[1], far[1], far[0]], dtype=np.float32)
        try:
            H = cv2.getPerspectiveTransform(SERVICE_RECT.astype(np.float32), img_pts)
            corners = cv2.perspectiveTransform(COURT_CORNERS.reshape(-1, 1, 2), H).reshape(-1, 2)
            model = CourtModel(corners, source="auto")
        except (cv2.error, np.linalg.LinAlgError):
            continue
        sc = score_court(model, mask)
        if sc > best_score:
            best, best_score = model, sc
    if best is None or best_score < min_score:
        return None
    best.score = round(best_score, 3)
    return best
