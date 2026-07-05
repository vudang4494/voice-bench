"""Adapter PhoWhisper (VinAI) — Whisper fine-tune cho tiếng Việt, SOTA trên
Common Voice VN / VIVOS / VLSP. Dùng transformers pipeline.

model_id mặc định 'vinai/PhoWhisper-large' — chỉnh qua config nếu casing khác.
"""
from __future__ import annotations

import logging
import numpy as np

from .base import ASREngine
from ..interfaces import ASRResult, LatencyBreakdown
from ..timing import measure
from ..audio import duration_s

logger = logging.getLogger(__name__)


class PhoWhisperASR(ASREngine):
    name = "phowhisper"

    def __init__(self, model_id: str = "vinai/PhoWhisper-large", device: str = "cuda"):
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise ImportError("pip install transformers torch") from e
        import time
        t0 = time.perf_counter()
        dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
        self._pipe = pipeline("automatic-speech-recognition", model=model_id, device=dev)
        self._load_s = time.perf_counter() - t0

    def transcribe(self, wav: np.ndarray, sr: int) -> ASRResult:
        dur = duration_s(wav, sr)
        # pipeline nhận dict {'array','sampling_rate'} -> tự resample về 16k.
        inp = {"array": wav, "sampling_rate": sr}
        with measure() as t:
            out = self._pipe(inp)
            text = (out.get("text") or "").strip()
        lat = LatencyBreakdown(total_s=t[0], media_dur_s=dur, model_load_s=self._load_s)
        return ASRResult(text=text, latency=lat, audio_dur_s=dur)
