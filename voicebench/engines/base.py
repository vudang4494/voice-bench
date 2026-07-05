"""Interface trừu tượng cho ASR / TTS. Adapter cụ thể kế thừa và lazy-import lib nặng."""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import numpy as np

from ..interfaces import ASRResult, TTSResult

logger = logging.getLogger(__name__)


class ASREngine(ABC):
    name: str = "asr-base"

    @abstractmethod
    def transcribe(self, wav: np.ndarray, sr: int) -> ASRResult:
        """audio -> text + latency (đã cuda-sync trong timing.measure)."""
        ...

    def warmup(self, wav: np.ndarray, sr: int, n: int = 2) -> None:
        """Chạy vài lần bỏ đi để loại cold-start/CUDA-init khỏi số đo."""
        for _ in range(n):
            self.transcribe(wav, sr)


class TTSEngine(ABC):
    name: str = "tts-base"

    @abstractmethod
    def synthesize(self, text: str, ref_wav: np.ndarray, ref_sr: int) -> TTSResult:
        """text (+ ref voice để clone) -> waveform + latency."""
        ...

    def warmup(self, text: str, ref_wav: np.ndarray, ref_sr: int, n: int = 2) -> None:
        for _ in range(n):
            self.synthesize(text, ref_wav, ref_sr)
