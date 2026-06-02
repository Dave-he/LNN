"""EMMA rover video + audio feature extraction (numpy-only).

Extracts two time-aligned feature streams from a single rover video:

- **Video**: per-frame motion magnitude + centroid of motion (2-dim).
  Captures the *wheel pose / actuation state* EMMA's YOLO pipeline targets,
  but uses frame-differencing instead of object detection so the pipeline
  has zero heavy dependencies (only ``PIL`` + ``numpy``).
- **Audio**: dominant spectral peak frequency (1-dim, in Hz) per video
  frame. EMMA's rover paper shows this correlates with motor speed and
  is the channel that breaks the audio-vs-video complementarity.

The function is idempotent on cached features: it skips extraction when
the output file already exists.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Tuple

import numpy as np
from PIL import Image


# Frame extraction is run at 15 fps to align with the audio STFT window
# below; the resulting streams are time-aligned to within a single frame.
DEFAULT_FPS = 15
DEFAULT_SR = 22050  # audio resampling rate (Hz)
DEFAULT_HOP = DEFAULT_SR // DEFAULT_FPS  # ~1467 samples per video frame


def _ensure_frames_and_audio(
    video_path: str,
    out_dir: str,
    fps: int = DEFAULT_FPS,
    audio_sr: int = DEFAULT_SR,
) -> Tuple[list[str], str]:
    """Run ffmpeg to dump per-frame PNGs and a mono WAV under ``out_dir``.

    Returns the list of frame paths (in order) and the audio wav path.
    Cached outputs are reused without re-running ffmpeg.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(out.glob("frame_*.png"))
    audio_path = out / "audio.wav"
    if not frames or not audio_path.exists():
        # Re-extract (cheap; the rover clip is 4 s).
        for old in out.glob("frame_*.png"):
            old.unlink()
        if audio_path.exists():
            audio_path.unlink()
        r1 = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={fps},scale=320:240",
                str(out / "frame_%04d.png"),
            ],
            capture_output=True, text=True,
        )
        if r1.returncode != 0:
            raise RuntimeError(f"ffmpeg frame extract failed: {r1.stderr[-400:]}")
        r2 = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1", "-ar", str(audio_sr),
                str(audio_path),
            ],
            capture_output=True, text=True,
        )
        if r2.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extract failed: {r2.stderr[-400:]}")
    frames = sorted(out.glob("frame_*.png"))
    return [str(p) for p in frames], str(audio_path)


def _load_audio_mono(path: str) -> np.ndarray:
    """Read a WAV file as mono float32 in [-1, 1] using only stdlib + numpy."""
    import wave
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sampwidth}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sr


def _stft_peak_hz(audio: np.ndarray, sr: int, hop: int, win: int = 1024) -> np.ndarray:
    """Dominant spectral peak (Hz) per STFT frame.  Returns shape ``[T]``."""
    if audio.size < win:
        # Pad to at least one full window.
        audio = np.pad(audio, (0, win - audio.size))
    n_frames = max(1, (audio.size - win) // hop + 1)
    window = np.hanning(win).astype(np.float32)
    peaks = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = audio[i * hop : i * hop + win] * window
        spec = np.abs(np.fft.rfft(seg))
        # Skip the DC bin when picking the peak so we focus on the tonal peak.
        if spec.size > 1:
            spec[0] = 0.0
        peak_bin = int(np.argmax(spec))
        peaks[i] = peak_bin * sr / win
    return peaks


def _motion_features(frame_paths: list[str]) -> np.ndarray:
    """Per-frame motion magnitude and centroid (3-dim: mag, cx, cy).

    Output shape ``[T, 3]`` where ``T == len(frame_paths)``.  The first
    frame is zero-filled because there is no preceding frame to diff.
    """
    n = len(frame_paths)
    feats = np.zeros((n, 3), dtype=np.float32)
    prev_gray = None
    for i, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("L")
        gray = np.asarray(img, dtype=np.float32) / 255.0
        if prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            mag = float(diff.mean())
            if diff.sum() > 1e-8:
                ys, xs = np.nonzero(diff > 0.05)
                cy = float(ys.mean()) / gray.shape[0] if ys.size else 0.5
                cx = float(xs.mean()) / gray.shape[1] if xs.size else 0.5
            else:
                cx, cy = 0.5, 0.5
            feats[i] = (mag, cx, cy)
        prev_gray = gray
    return feats


def extract_rover_features(
    video_path: str,
    out_dir: str,
    fps: int = DEFAULT_FPS,
    audio_sr: int = DEFAULT_SR,
) -> dict[str, np.ndarray]:
    """Return ``{"video": [T, 3], "audio": [T], "sr": int, "fps": int}``.

    Video features are (motion_magnitude, centroid_x, centroid_y); audio
    is the dominant spectral peak frequency (Hz) at each video frame
    boundary.  Both streams are time-aligned with the same length ``T``.
    """
    frames, audio_path = _ensure_frames_and_audio(video_path, out_dir, fps, audio_sr)
    video_feats = _motion_features(frames)
    audio, sr = _load_audio_mono(audio_path)
    hop = sr // fps
    audio_peaks = _stft_peak_hz(audio, sr, hop)
    # Trim or pad audio to match video length.
    T = video_feats.shape[0]
    if audio_peaks.size >= T:
        audio_peaks = audio_peaks[:T]
    else:
        audio_peaks = np.pad(audio_peaks, (0, T - audio_peaks.size), mode="edge")
    return {
        "video": video_feats,
        "audio": audio_peaks.astype(np.float32),
        "sr": sr,
        "fps": fps,
    }


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Extract EMMA rover video+audio features")
    parser.add_argument("--video", default="/tmp/RoverVideo.mp4")
    parser.add_argument("--out-dir", default="/tmp/emma_features")
    parser.add_argument("--cache", default="/tmp/emma_features/features.npz")
    args = parser.parse_args()
    if os.path.exists(args.cache):
        d = np.load(args.cache)
        out = {k: d[k] for k in d.files}
    else:
        out = extract_rover_features(args.video, args.out_dir)
        np.savez(args.cache, **out)
    print(json.dumps(
        {k: (v.shape if hasattr(v, "shape") else v) for k, v in out.items()},
        indent=2, default=str,
    ))
