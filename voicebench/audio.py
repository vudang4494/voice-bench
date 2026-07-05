"""I/O audio tối thiểu qua soundfile. Waveform chuẩn nội bộ: float32 mono [-1,1]."""
from __future__ import annotations

import numpy as np


def load_wav(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:  # stereo -> mono
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def save_wav(path: str, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(path, wav.astype(np.float32), sr)


def duration_s(wav: np.ndarray, sr: int) -> float:
    return len(wav) / sr if sr else 0.0


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample polyphase (scipy). Trả về chính wav nếu sr đã đúng target."""
    if sr == target_sr:
        return wav
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target_sr))
    out = resample_poly(wav, int(target_sr) // g, int(sr) // g)
    return np.asarray(out, dtype=np.float32)
