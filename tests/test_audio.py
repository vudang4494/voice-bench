"""T2 fix: resample helper (faster-whisper cần 16k, không tự resample ndarray)."""
import numpy as np

from voicebench.audio import duration_s, resample


def test_resample_441k_to_16k_length_and_dtype():
    sr, dur = 44100, 2.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = resample(wav, sr, 16000)
    assert abs(len(out) - int(dur * 16000)) <= 2
    assert out.dtype == np.float32


def test_resample_identity_same_sr():
    wav = np.zeros(1600, dtype=np.float32)
    assert resample(wav, 16000, 16000) is wav


def test_resample_preserves_duration():
    sr = 8000
    wav = np.random.default_rng(0).standard_normal(sr * 3).astype(np.float32)
    out = resample(wav, sr, 16000)
    assert abs(duration_s(out, 16000) - duration_s(wav, sr)) < 1e-3
