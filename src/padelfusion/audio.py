"""Ball-impact onset detection from the audio track."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import butter, sosfiltfilt, stft

from .config import AudioConfig


@dataclass
class Onset:
    t: float
    strength: float  # onset-function peak (band-limited spectral flux)
    level_db: float  # peak band level around the onset, dBFS

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, 0.45 * sr)
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def onset_function(y: np.ndarray, cfg: AudioConfig) -> tuple[np.ndarray, float]:
    """Band-limited log spectral flux. Returns (odf, frame_rate)."""
    sr = cfg.sample_rate
    freqs, _, Z = stft(y, fs=sr, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, boundary=None)
    band = (freqs >= cfg.band_low_hz) & (freqs <= cfg.band_high_hz)
    mag = np.log1p(100.0 * np.abs(Z[band]))
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    odf = np.concatenate([[0.0], flux])
    return odf, sr / cfg.hop


def detect_onsets(y: np.ndarray, cfg: AudioConfig) -> tuple[list[Onset], list[Onset]]:
    """Return (kept, rejected_as_quiet) onsets.

    Peak picking uses an adaptive threshold: local median plus k times the
    local median absolute deviation, over a one-second window. Onsets far
    quieter than this court's typical impacts are split off: they are
    usually neighbouring courts.
    """
    if y.size < cfg.n_fft * 4:
        return [], []
    sr = cfg.sample_rate
    yb = bandpass(y, sr, cfg.band_low_hz, cfg.band_high_hz)
    odf, fr = onset_function(yb, cfg)
    win = max(3, int(fr * 1.0) | 1)
    med = median_filter(odf, size=win, mode="nearest")
    mad = median_filter(np.abs(odf - med), size=win, mode="nearest")
    floor = np.percentile(odf, 50) + 1e-6
    thresh = med + cfg.threshold_k * np.maximum(mad, 0.25 * floor)
    min_gap = max(1, int(cfg.min_interval_s * fr))
    local_max = odf == maximum_filter1d(odf, size=2 * min_gap + 1, mode="nearest")
    peaks = np.flatnonzero(local_max & (odf > thresh) & (odf > 0))

    onsets: list[Onset] = []
    half = int(0.02 * sr)
    last_t = -1e9
    for p in peaks:
        t = p * cfg.hop / sr
        if t - last_t < cfg.min_interval_s:
            continue
        c = int(t * sr)
        seg = yb[max(0, c - half): c + 3 * half]
        level = 20 * np.log10(np.max(np.abs(seg)) + 1e-9) if seg.size else -120.0
        onsets.append(Onset(float(t), float(odf[p]), float(level)))
        last_t = t
    if not onsets:
        return [], []
    ref = np.percentile([o.level_db for o in onsets], 90)
    kept = [o for o in onsets if o.level_db >= ref - cfg.rel_level_floor_db]
    quiet = [o for o in onsets if o.level_db < ref - cfg.rel_level_floor_db]
    return kept, quiet
