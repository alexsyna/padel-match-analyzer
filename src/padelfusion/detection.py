"""Player detection backends.

- ``yolo``: Ultralytics person detector (``pip install padelfusion[yolo]``).
  It's the recommended backend for real footage.
- ``background``: fixed-camera background subtraction against a temporal
  median of the clip. It needs no model weights and runs on CPU. It merges
  players who overlap, and the tracker flags those frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DetectionConfig
from .court import CourtModel, expected_player_height_px


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float
    hist: np.ndarray = field(repr=False, default=None)

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)


def appearance_hist(frame: np.ndarray, bbox) -> np.ndarray:
    """Hue/saturation histogram of the upper body, used to tell partners apart."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h = y2 - y1
    y1c, y2c = y1 + int(0.15 * h), y1 + int(0.6 * h)
    x1, x2 = max(x1, 0), min(x2, frame.shape[1])
    y1c, y2c = max(y1c, 0), min(y2c, frame.shape[0])
    if x2 - x1 < 2 or y2c - y1c < 2:
        return np.full(32, 1 / 32, dtype=np.float32)
    hsv = cv2.cvtColor(frame[y1c:y2c, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel().astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else np.full(32, 1 / 32, dtype=np.float32)


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def merge_detections(dets_a: list[Detection], dets_b: list[Detection],
                     iou_thresh: float = 0.3) -> list[Detection]:
    """Pairs across the two lists whose IoU exceeds `iou_thresh` collapse to
    the higher-confidence detection; everything else from both lists survives."""
    used_b = set()
    merged = []
    for a in dets_a:
        best_j, best_iou = None, 0.0
        for j, b in enumerate(dets_b):
            if j in used_b:
                continue
            i = _iou(a, b)
            if i > best_iou:
                best_j, best_iou = j, i
        if best_j is not None and best_iou > iou_thresh:
            used_b.add(best_j)
            merged.append(a if a.score >= dets_b[best_j].score else dets_b[best_j])
        else:
            merged.append(a)
    merged.extend(b for j, b in enumerate(dets_b) if j not in used_b)
    return merged


def detect_two_pass(detect_fn, frame: np.ndarray, far_bbox: tuple[int, int, int, int] | None,
                    upscale: float, iou_thresh: float = 0.3) -> list[Detection]:
    """Run `detect_fn` on the full frame, and again on an upscaled crop of
    `far_bbox` (if given and `upscale > 1`), mapping the far pass's boxes
    back to full-frame coordinates before merging."""
    dets = detect_fn(frame)
    if far_bbox is None or upscale <= 1.0:
        return dets
    x1, y1, x2, y2 = far_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return dets
    up = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    far_dets = detect_fn(up)
    mapped = [Detection((d.bbox[0] / upscale + x1, d.bbox[1] / upscale + y1,
                         d.bbox[2] / upscale + x1, d.bbox[3] / upscale + y1), d.score, d.hist)
             for d in far_dets]
    return merge_detections(dets, mapped, iou_thresh)


class BackgroundDetector:
    name = "background"

    def __init__(self, cfg: DetectionConfig, sample_frames: list[np.ndarray],
                 court_mask: np.ndarray | None = None, court: "CourtModel | None" = None):
        if not sample_frames:
            raise ValueError("background detector needs sample frames")
        self.cfg = cfg
        self.background = np.median(np.stack(sample_frames), axis=0).astype(np.uint8)
        self.court_mask = court_mask
        self.court = court
        h, w = self.background.shape[:2]
        self.min_area = cfg.bg_min_area_frac * h * w

    def detect(self, frame: np.ndarray) -> list[Detection]:
        diff = cv2.absdiff(frame, self.background)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(diff, self.cfg.bg_diff_threshold, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
        if self.court_mask is not None:
            fg = cv2.bitwise_and(fg, self.court_mask)
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg)
        edges = _edge_energy(frame)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if h < w * 0.8:
                continue
            if self.court is not None:
                foot = self.court.image_to_court([[x + w / 2.0, y + h]])[0]
                if h < self.cfg.bg_min_height_frac * expected_player_height_px(self.court, foot[1]):
                    continue
            elif area < self.min_area:
                continue
            if self._is_ghost(edges, x, y, w, h):
                continue
            bbox = (float(x), float(y), float(x + w), float(y + h))
            dets.append(Detection(bbox, 1.0, appearance_hist(frame, bbox)))
        return dets


    def _is_ghost(self, edges: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """A player who stood still long enough ends up in the median background.
        When they leave, the empty spot differs from the background too. It is
        a ghost if the structure (edges) is in the background patch, not the frame."""
        p = 3
        ys, xs = slice(max(y - p, 0), y + h + p), slice(max(x - p, 0), x + w + p)
        now, bg = float(edges[ys, xs].mean()), float(self.bg_edges[ys, xs].mean())
        return now < 0.5 * bg

    @property
    def bg_edges(self) -> np.ndarray:
        if not hasattr(self, "_bg_edges"):
            self._bg_edges = _edge_energy(self.background)
        return self._bg_edges


def _edge_energy(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))


class YoloDetector:
    name = "yolo"

    def __init__(self, cfg: DetectionConfig, far_bbox: tuple[int, int, int, int] | None = None):
        from ultralytics import YOLO  # optional dependency

        self.cfg = cfg
        self.far_bbox = far_bbox if cfg.far_field_enabled else None
        self.model = YOLO(cfg.yolo_model)

    def _raw_detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, classes=[0], conf=self.cfg.yolo_min_conf, verbose=False)[0]
        dets = []
        for box, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            bbox = tuple(float(v) for v in box)
            dets.append(Detection(bbox, float(conf), appearance_hist(frame, bbox)))
        return dets

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return detect_two_pass(self._raw_detect, frame, self.far_bbox, self.cfg.far_field_upscale)


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


def make_detector(cfg: DetectionConfig, sample_frames_fn, court_mask=None, court=None,
                  far_bbox: tuple[int, int, int, int] | None = None):
    backend = cfg.backend
    if backend == "auto":
        backend = "yolo" if yolo_available() else "background"
    if backend == "yolo":
        return YoloDetector(cfg, far_bbox=far_bbox)
    if backend == "background":
        return BackgroundDetector(cfg, sample_frames_fn(), court_mask, court=court)
    raise ValueError(f"unknown detector backend: {cfg.backend}")
