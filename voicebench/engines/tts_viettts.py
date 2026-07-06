"""Adapter VietTTS (dangvansam) qua HTTP — server tương thích OpenAI TTS API.
Chạy server trước: `viettts server --host 0.0.0.0 --port 8298`.

Ưu điểm: tách process, không nhét lib TTS vào harness; đo latency = round-trip
HTTP (đã gồm network local, chấp nhận được vì cùng máy).

LƯU Ý: engine này KHÔNG voice-clone — ref_wav bị bỏ qua, giọng chọn bằng kwarg
`voice`. supports_cloning=False để run_benchmark tự ghi speaker_sim = null.
"""
from __future__ import annotations

import io
import logging
import numpy as np

from .base import TTSEngine
from ..interfaces import TTSResult, LatencyBreakdown
from ..timing import measure

logger = logging.getLogger(__name__)


class VietTTSHttp(TTSEngine):
    name = "viettts-http"
    supports_cloning = False  # bỏ qua ref_wav — speaker_sim phải là null

    def __init__(self, base_url: str = "http://localhost:8298",
                 voice: str = "0", api_key: str = "viet-tts", timeout: float = 60.0,
                 health_timeout: float = 3.0):
        root = base_url.rstrip("/")
        self._url = root + "/v1/audio/speech"
        self._voice = voice
        self._key = api_key
        self._timeout = timeout
        # Health check ngay lúc build: chết ở sample 1 với ConnectionError thô
        # là quá muộn (giữa run benchmark dài). Server trả gì cũng được, miễn
        # là kết nối được trong health_timeout.
        import requests
        try:
            requests.get(root, timeout=health_timeout)
        except requests.RequestException as e:
            raise RuntimeError(
                f"VietTTS server không chạy tại {root} — khởi động trước bằng: "
                f"viettts server --host 0.0.0.0 --port 8298") from e

    def synthesize(self, text: str, ref_wav: np.ndarray, ref_sr: int) -> TTSResult:
        import requests, soundfile as sf
        payload = {"model": "tts-1", "input": text, "voice": self._voice,
                   "response_format": "wav"}
        headers = {"Authorization": f"Bearer {self._key}"}
        with measure() as t:  # CPU path -> measure = wall time HTTP
            r = requests.post(self._url, json=payload, headers=headers,
                              timeout=self._timeout)
            r.raise_for_status()
            wav, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        dur = len(wav) / sr if sr else 0.0
        lat = LatencyBreakdown(total_s=t[0], media_dur_s=dur)
        return TTSResult(audio=wav, sample_rate=int(sr), latency=lat, out_dur_s=dur)
