"""Adapter ChunkFormer (khanhld) — encoder chunk-based CTC cho ASR tiếng Việt,
ứng viên T9 thay PhoWhisper: claim WER tốt hơn với ~110M params, không bị floor
padding 30s của whisper (kỳ vọng latency clip ngắn thấp hơn hẳn) và long-form
decode "endless" không mất nội dung ở ranh giới cửa sổ.

API của package nhận ĐƯỜNG DẪN file (không nhận ndarray) -> phải ghi wav tạm;
chi phí ghi nằm TRONG cửa sổ đo — là chi phí per-request thật khi serving nhận
audio bytes, giữ so sánh công bằng với faster-whisper (resample cũng trong đo).
"""
from __future__ import annotations

import logging
import numpy as np

from .base import ASREngine
from ..interfaces import ASRResult, LatencyBreakdown
from ..timing import measure
from ..audio import duration_s, resample, save_wav

logger = logging.getLogger(__name__)


class ChunkFormerASR(ASREngine):
    name = "chunkformer"

    def __init__(self, model_id: str = "khanhld/chunkformer-large-vie",
                 device: str = "cpu", chunk_size: int = 64,
                 left_context_size: int = 128, right_context_size: int = 128,
                 longform_min_s: float | None = 30.0,
                 max_silence_duration: float = 0.5,
                 total_batch_duration: int = 1800):
        try:
            from chunkformer import ChunkFormerModel
        except ImportError as e:
            raise ImportError("pip install chunkformer") from e
        import time
        import torch
        t0 = time.perf_counter()
        self._model = ChunkFormerModel.from_pretrained(model_id)
        self._model.eval()
        if device == "cuda" and torch.cuda.is_available():
            self._model.cuda()
        self._load_s = time.perf_counter() - t0
        self._chunk = int(chunk_size)
        self._lctx = int(left_context_size)
        self._rctx = int(right_context_size)
        # Clip ngắn dùng batch_decode; audio >= longform_min_s chuyển
        # endless_decode (cache trượt, không giới hạn độ dài). None = luôn batch.
        self._longform_min_s = (float(longform_min_s)
                                if longform_min_s is not None else None)
        self._max_sil = float(max_silence_duration)
        self._batch_dur = int(total_batch_duration)

    def transcribe(self, wav: np.ndarray, sr: int) -> ASRResult:
        import os
        import tempfile
        dur = duration_s(wav, sr)
        with measure() as t:
            wav16 = resample(wav, sr, 16000)
            # mkstemp + đóng fd trước khi ghi/đọc theo path: mở lại file đang mở
            # không portable (Windows); chi phí ghi/xoá vẫn TRONG cửa sổ đo.
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                save_wav(tmp, wav16, 16000)
                if (self._longform_min_s is not None
                        and dur >= self._longform_min_s):
                    text = self._model.endless_decode(
                        tmp, chunk_size=self._chunk,
                        left_context_size=self._lctx,
                        right_context_size=self._rctx,
                        total_batch_duration=self._batch_dur,
                        return_timestamps=False,
                        max_silence_duration=self._max_sil)
                else:
                    text = self._model.batch_decode(
                        [tmp], chunk_size=self._chunk,
                        left_context_size=self._lctx,
                        right_context_size=self._rctx,
                        total_batch_duration=self._batch_dur)[0]
            finally:
                os.unlink(tmp)
            text = str(text).strip()
        lat = LatencyBreakdown(total_s=t[0], media_dur_s=dur, model_load_s=self._load_s)
        return ASRResult(text=text, latency=lat, audio_dur_s=dur)
