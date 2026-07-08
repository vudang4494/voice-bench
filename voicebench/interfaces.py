"""Result dataclasses dùng chung giữa engines và metrics.

Thiết kế forward-compatible: LatencyBreakdown mang sẵn field streaming
(ttfa_s, first_token_s) ở dạng Optional -> chuyển sang streaming chỉ cần
adapter điền thêm, harness không đổi.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


@dataclass
class LatencyBreakdown:
    # Tất cả thời gian tính bằng giây (s). model_load_s TÁCH RIÊNG khỏi inference.
    total_s: float                          # wall time của inference (đã cuda-sync)
    media_dur_s: float                      # độ dài audio in (ASR) hoặc out (TTS)
    model_load_s: float = 0.0               # cold-start / load weights — KHÔNG gộp vào total
    ttfa_s: Optional[float] = None          # time-to-first-audio (TTS streaming)
    first_token_s: Optional[float] = None   # time-to-first-token (ASR streaming)

    @property
    def rtf(self) -> float:
        """Real-time factor = processing / media duration. Voice UI cần << 1."""
        return self.total_s / self.media_dur_s if self.media_dur_s > 0 else float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rtf"] = self.rtf
        return d


@dataclass
class ASRResult:
    text: str
    latency: LatencyBreakdown
    audio_dur_s: float
    # Segments VAD (mỗi phần tử {start, end, text}) — cho sentiment long-form chấm
    # theo câu thay vì gộp 1 text rồi cắt cụt ở 256 token. None = engine không cấp
    # (harness/WER chỉ dùng `text`, nên đây là bổ sung KHÔNG phá tương thích cũ).
    segments: Optional[list] = None


@dataclass
class TTSResult:
    audio: np.ndarray            # waveform float32 [-1, 1], mono
    sample_rate: int
    latency: LatencyBreakdown
    out_dur_s: float

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0


@dataclass
class SampleRecord:
    """1 dòng trong results.jsonl — raw, đủ để re-aggregate về sau."""
    sample_id: str
    ref_text: str                # ground-truth transcript
    asr_text: str                # ASR output (lần 1, trên ref audio)
    roundtrip_text: str          # ASR output (lần 2, trên audio TTS sinh ra)
    asr_latency: dict
    tts_latency: dict
    speaker_sim: Optional[float] = None
    mos: Optional[float] = None
    extra: dict = field(default_factory=dict)
