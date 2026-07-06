"""Contract TTS engine (T4): capability flag voice-clone + health check VietTTS.

Không cần model thật — chỉ kiểm tra contract lớp adapter.
"""
import pytest

from voicebench.engines.base import TTSEngine


def test_tts_engine_default_supports_cloning():
    assert TTSEngine.supports_cloning is True


def test_viettts_declares_no_cloning():
    # Class attr — không cần server để đọc capability.
    from voicebench.engines.tts_viettts import VietTTSHttp
    assert VietTTSHttp.supports_cloning is False


def test_viettts_health_check_actionable_error():
    """Server chưa chạy -> lỗi RuntimeError chỉ rõ cách khởi động, trong ~3s,
    thay vì ConnectionError thô ở sample 1 giữa run benchmark."""
    from voicebench.engines.tts_viettts import VietTTSHttp
    with pytest.raises(RuntimeError, match="viettts server"):
        # Port 1 (reserved) -> connection refused ngay lập tức trên localhost.
        VietTTSHttp(base_url="http://127.0.0.1:1", health_timeout=1.0)
