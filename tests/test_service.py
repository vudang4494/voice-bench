"""Service HTTP: decode helper + endpoints với engine stub (không cần model thật)."""
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voicebench.engines.base import ASREngine
from voicebench.interfaces import ASRResult, LatencyBreakdown
from voicebench.service import create_app, decode_audio_bytes


def _wav_bytes(dur_s=1.0, sr=16000, fmt="WAV"):
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format=fmt)
    return buf.getvalue()


def test_decode_wav_bytes():
    wav, sr = decode_audio_bytes(_wav_bytes(dur_s=0.5))
    assert sr == 16000
    assert abs(len(wav) - 8000) <= 2
    assert wav.dtype == np.float32


def test_decode_mp3_bytes():
    wav, sr = decode_audio_bytes(_wav_bytes(dur_s=0.5, fmt="MP3"))
    assert sr == 16000
    # mp3 pad frame: cho phép lệch nhỏ
    assert abs(len(wav) - 8000) < sr * 0.1


def test_decode_garbage_raises_400():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        decode_audio_bytes(b"not audio at all")
    assert ei.value.status_code == 400


class _StubASR(ASREngine):
    name = "stub"
    _load_s = 0.01

    def transcribe(self, wav, sr):
        dur = len(wav) / sr
        return ASRResult(
            text="xin chào việt nam",
            latency=LatencyBreakdown(total_s=0.05, media_dur_s=dur,
                                     model_load_s=self._load_s),
            audio_dur_s=dur)


@pytest.fixture()
def client():
    app = create_app(config={"warmup": 1, "asr": {"name": "stub"}},
                     asr_engine=_StubASR())
    with TestClient(app) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["asr"]["name"] == "stub"
    assert h["tts"]["available"] is False


def test_asr_endpoint_wav(client):
    r = client.post("/v1/asr", files={"file": ("a.wav", _wav_bytes(1.0))})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "xin chào việt nam"
    assert body["audio"]["sample_rate_in"] == 16000
    assert body["latency"]["infer_total_s"] == 0.05
    assert body["latency"]["server_total_s"] >= body["latency"]["decode_s"]
    assert body["latency"]["rtf"] == pytest.approx(0.05 / 1.0, rel=1e-3)


def test_asr_endpoint_empty_file(client):
    r = client.post("/v1/asr", files={"file": ("a.wav", b"")})
    assert r.status_code == 400


def test_tts_unavailable_503(client):
    r = client.post("/v1/tts", data={"text": "xin chào"})
    assert r.status_code == 503
    assert "tts" in r.json()["detail"].lower()


class _StubTTS:
    name = "stub-tts"

    def synthesize(self, text, ref_wav, ref_sr):
        from voicebench.interfaces import TTSResult, LatencyBreakdown
        wav = np.zeros(2400, dtype=np.float32)
        return TTSResult(audio=wav, sample_rate=24000,
                         latency=LatencyBreakdown(total_s=0.01, media_dur_s=0.1),
                         out_dur_s=0.1)

    def warmup(self, text, ref_wav, ref_sr, n=2):
        pass


@pytest.fixture()
def client_tts():
    app = create_app(config={"warmup": 0, "asr": {"name": "stub"}},
                     asr_engine=_StubASR(), tts_engine=_StubTTS())
    with TestClient(app) as c:
        yield c


def test_upload_size_limit_413():
    app = create_app(config={"warmup": 0, "asr": {"name": "stub"},
                             "max_upload_mb": 1},
                     asr_engine=_StubASR())
    with TestClient(app) as c:
        big = _wav_bytes(dur_s=40.0)  # ~1.28MB wav 16-bit? float64 -> lớn hơn 1MB chắc chắn
        assert len(big) > 1024 * 1024
        r = c.post("/v1/asr", files={"file": ("big.wav", big)})
        assert r.status_code == 413


def test_tts_missing_text_422(client_tts):
    # multipart form: thiếu field text -> FastAPI validation 422
    r = client_tts.post("/v1/tts", data={})
    assert r.status_code == 422


def test_tts_blank_text_400(client_tts):
    r = client_tts.post("/v1/tts", data={"text": "   "})
    assert r.status_code == 400


def test_tts_server_path_rejected(client_tts):
    # Contract siết T4: KHÔNG còn nhận path server-side — ref_audio phải là file
    # upload; gửi string path bị validation từ chối (422), không bao giờ mở path.
    r = client_tts.post("/v1/tts", data={"text": "xin chào",
                                         "ref_audio": "/etc/hosts"})
    assert r.status_code == 422


def test_tts_bad_ref_audio_400(client_tts):
    r = client_tts.post("/v1/tts", data={"text": "xin chào"},
                        files={"ref_audio": ("x.wav", b"khong phai audio",
                                             "audio/wav")})
    assert r.status_code == 400


def test_tts_ok_with_stub(client_tts):
    r = client_tts.post("/v1/tts", data={"text": "xin chào"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"


def test_tts_ok_with_ref_upload(client_tts):
    r = client_tts.post("/v1/tts", data={"text": "xin chào"},
                        files={"ref_audio": ("ref.wav", _wav_bytes(0.5))})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"


def test_health_not_blocked_by_slow_asr():
    """Endpoint def (threadpool): /health phải trả lời khi ASR đang bận."""
    import threading
    import time as _time

    class _SlowASR(_StubASR):
        def transcribe(self, wav, sr):
            _time.sleep(1.0)
            return super().transcribe(wav, sr)

    app = create_app(config={"warmup": 0, "asr": {"name": "stub"}},
                     asr_engine=_SlowASR())
    with TestClient(app) as c:
        t = threading.Thread(target=lambda: c.post(
            "/v1/asr", files={"file": ("a.wav", _wav_bytes(0.3))}))
        t.start()
        _time.sleep(0.2)  # chắc chắn ASR đang chạy
        t0 = _time.perf_counter()
        h = c.get("/health")
        dt = _time.perf_counter() - t0
        t.join()
        assert h.status_code == 200
        assert dt < 0.5, f"/health bị chặn {dt:.2f}s bởi ASR đang chạy"
