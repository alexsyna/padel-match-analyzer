"""Video probing, capture-spec validation, frame and audio access."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureSpec


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - environment specific
        raise RuntimeError("ffmpeg not found: install ffmpeg or `pip install imageio-ffmpeg`") from e


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    has_audio: bool

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: str | Path) -> VideoInfo:
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path],
                          capture_output=True, text=True)
    has_audio = bool(re.search(r"Stream #\S+.*Audio:", proc.stderr))
    duration = n / fps if fps > 0 else 0.0
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", proc.stderr)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return VideoInfo(path, width, height, fps, n, duration, has_audio)


def check_capture(info: VideoInfo, spec: CaptureSpec) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) against the v0 capture spec."""
    errors, warnings = [], []
    ext = Path(info.path).suffix.lower()
    if ext not in spec.allowed_extensions:
        errors.append(f"unsupported container {ext or '(none)'}: use MP4 or MOV")
    if spec.require_audio and not info.has_audio:
        errors.append("no audio track: shot detection needs audio, record with sound on")
    if info.fps < spec.min_fps:
        errors.append(f"frame rate {info.fps:.2f} fps is below the 30 fps minimum")
    if min(info.width, info.height) < spec.min_short_side_px:
        errors.append(f"resolution {info.width}x{info.height} is below 1080p")
    if info.height > info.width:
        errors.append("portrait video: record in landscape")
    if not (spec.recommended_min_s <= info.duration_s <= spec.recommended_max_s):
        warnings.append(
            f"clip is {info.duration_s / 60:.1f} min; v0 is tuned on 5-10 min clips"
        )
    return errors, warnings


def iter_frames(path: str | Path, sample_fps: float, start_s: float = 0.0, end_s: float | None = None):
    """Yield (frame_index, t_seconds, frame) at roughly `sample_fps`."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    idx = 0
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
        idx = int(start_s * fps)
    try:
        while True:
            if end_s is not None and idx / fps > end_s:
                break
            if idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / fps, frame
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()


def read_frame(path: str | Path, t_s: float = 0.0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame at {t_s}s from {path}")
    return frame


def sample_frames(path: str | Path, n: int, times: list[float] | None = None) -> list[np.ndarray]:
    """`n` frames spread over the clip, or taken at the given times (subsampled to `n`)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if times:
        times = sorted(times)
        pick = np.linspace(0, len(times) - 1, min(n, len(times))).round().astype(int)
        idxs = sorted({min(int(times[i] * fps), max(total - 1, 0)) for i in pick})
    else:
        idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def extract_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode the audio track to mono float32 in [-1, 1]."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"audio extraction failed: {proc.stderr.decode(errors='replace')[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def cut_clip(path: str | Path, out: str | Path, start_s: float, end_s: float) -> None:
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(start_s, 0):.3f}",
         "-i", str(path), "-t", f"{end_s - max(start_s, 0):.3f}", "-c:v", "libx264",
         "-preset", "veryfast", "-c:a", "aac", str(out)],
        check=True,
    )
