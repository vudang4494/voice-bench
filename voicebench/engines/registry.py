"""Map tên (trong config) -> class engine. Lazy: chỉ import class được chọn."""
from __future__ import annotations

from typing import Any

_ASR = {
    "phowhisper": ("voicebench.engines.asr_phowhisper", "PhoWhisperASR"),
    "faster-whisper": ("voicebench.engines.asr_whisper", "FasterWhisperASR"),
    "chunkformer": ("voicebench.engines.asr_chunkformer", "ChunkFormerASR"),
}
_TTS = {
    "vixtts": ("voicebench.engines.tts_vixtts", "ViXTTS"),
    "viettts-http": ("voicebench.engines.tts_viettts", "VietTTSHttp"),
}


def _build(table: dict, name: str, kwargs: dict) -> Any:
    if name not in table:
        raise KeyError(f"Engine '{name}' chưa đăng ký. Có: {list(table)}")
    import importlib
    mod, cls = table[name]
    klass = getattr(importlib.import_module(mod), cls)
    return klass(**(kwargs or {}))


def build_asr(name: str, kwargs: dict | None = None):
    return _build(_ASR, name, kwargs or {})


def build_tts(name: str, kwargs: dict | None = None):
    return _build(_TTS, name, kwargs or {})
