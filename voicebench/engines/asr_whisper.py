"""Adapter faster-whisper (CT2). Dùng cho cả whisper gốc lẫn PhoWhisper bản
convert CT2 (vd diepho/PhoWhisper-small-ct2) — đường serving chính trên CPU/Mac.

Decode params đi qua config (asr.kwargs), đã tune cho long-form trên Mac Mini M4
(xem ROADMAP T7 + runs/tune_longform_*.json): whisper mất nội dung ở ranh giới
cửa sổ 30s khi decode tuần tự; vad_filter cắt theo khoảng lặng thật + không
carry context lỗi giữa các đoạn (condition_on_previous_text=False) sửa việc này.
"""
from __future__ import annotations

import logging
import numpy as np

from .base import ASREngine
from ..interfaces import ASRResult, LatencyBreakdown
from ..timing import measure
from ..audio import duration_s, resample

logger = logging.getLogger(__name__)


class FasterWhisperASR(ASREngine):
    name = "faster-whisper"

    def __init__(self, model_id: str = "large-v3", device: str = "cuda",
                 compute_type: str = "int8_float16", language: str = "vi",
                 beam_size: int = 5, vad_filter: bool = False,
                 vad_parameters: dict | None = None,
                 condition_on_previous_text: bool = True,
                 longform_min_s: float | None = None,
                 longform: dict | None = None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError("pip install faster-whisper") from e
        import time
        t0 = time.perf_counter()
        self._model = WhisperModel(model_id, device=device, compute_type=compute_type)
        self._load_s = time.perf_counter() - t0
        self._lang = language
        self._beam = int(beam_size)
        self._vad = bool(vad_filter)
        self._vad_params = vad_parameters
        self._condition = bool(condition_on_previous_text)
        # Profile long-form: audio >= longform_min_s dùng bộ params riêng
        # (override từng key, key thiếu kế thừa default). None = tắt.
        self._longform_min_s = (float(longform_min_s)
                                if longform_min_s is not None else None)
        lf = dict(longform or {})
        self._lf_beam = int(lf.get("beam_size", beam_size))
        self._lf_vad = bool(lf.get("vad_filter", vad_filter))
        self._lf_vad_params = lf.get("vad_parameters", vad_parameters)
        self._lf_condition = bool(lf.get("condition_on_previous_text",
                                         condition_on_previous_text))

    def transcribe(self, wav: np.ndarray, sr: int) -> ASRResult:
        dur = duration_s(wav, sr)
        # Chọn profile theo độ dài: decode tuần tự nhiều cửa sổ 30s mất nội dung
        # ở ranh giới (T7), nên request dài cần params khác request ngắn.
        # Tất cả state chỉ đọc sau __init__ -> thread-safe với threadpool service.
        if self._longform_min_s is not None and dur >= self._longform_min_s:
            beam, vad, vad_params, cond = (self._lf_beam, self._lf_vad,
                                           self._lf_vad_params,
                                           self._lf_condition)
        else:
            beam, vad, vad_params, cond = (self._beam, self._vad,
                                           self._vad_params, self._condition)
        # faster-whisper KHÔNG resample ndarray input (chỉ resample khi decode từ
        # file) -> phải tự đưa về 16k. Resample nằm TRONG cửa sổ đo: là chi phí
        # serving thật của mọi request audio không-16k.
        with measure() as t:
            wav16 = resample(wav, sr, 16000)
            kwargs: dict = {"language": self._lang, "beam_size": beam,
                            "condition_on_previous_text": cond}
            if vad:
                kwargs["vad_filter"] = True
                if vad_params:
                    kwargs["vad_parameters"] = vad_params
            segments, _ = self._model.transcribe(wav16, **kwargs)
            text = " ".join(s.text for s in segments).strip()
        lat = LatencyBreakdown(total_s=t[0], media_dur_s=dur, model_load_s=self._load_s)
        return ASRResult(text=text, latency=lat, audio_dur_s=dur)
