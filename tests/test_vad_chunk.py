"""Test EXTERNAL VAD CHUNKING (scripts/rules.vad_chunk.yaml) — engine-free (mock).

Khoá các rule không cần model/service:
  VAD-MONO-1  timestamp tuyệt đối, đơn điệu, trong biên (assembler _abs_segments)
  VAD-BND-1   clamp trong cửa sổ VAD + bỏ segment rỗng
  VAD-DUR-1   VadOptions.max_speech_duration_s = max_chunk_s (< 30s)
  VAD-COMPAT-1 cờ tắt / audio ngắn → path 1-call CŨ y hệt

Dùng object.__new__ + FakeModel (như test_engines_measure.py) — không tải model.
"""
from collections import namedtuple

import numpy as np

from voicebench.engines.asr_whisper import FasterWhisperASR, _abs_segments

# seg giả lập của faster-whisper (attr .start/.end/.text/... — GIÂY CỤC BỘ trong clip)
_Seg = namedtuple("_Seg", "start end text avg_logprob no_speech_prob compression_ratio")


def _seg(start, end, text):
    return {"start": start, "end": end, "text": text,
            "avg_logprob": -0.1, "no_speech_prob": 0.0, "compression_ratio": 1.2}


# ── PURE assembler _abs_segments (VAD-MONO-1, VAD-BND-1) ─────────────────────
def test_abs_segments_offset_tuyet_doi():
    """START cục bộ + offset biên cửa sổ VAD → START tuyệt đối đúng thời điểm."""
    wl = [
        {"win_start": 0, "win_end": 16000 * 5, "segs": [_seg(0.5, 1.0, "a")]},
        {"win_start": 16000 * 10, "win_end": 16000 * 15, "segs": [_seg(0.5, 1.0, "b")]},
    ]
    out = _abs_segments(wl, sr=16000)
    assert len(out) == 2
    assert abs(out[0]["start"] - 0.5) < 1e-9                  # w0=0 + 0.5
    assert abs(out[1]["start"] - 10.5) < 1e-9                # w0=10 + 0.5
    assert [s["text"] for s in out] == ["a", "b"]


def test_abs_segments_tile_end_trong_cua_so():
    """END dồn của whisper được TILE: end_i = start_{i+1}, end cuối = biên cửa sổ
    (không tràn qua lặng VAD). Sửa blip 0.02s → timeline liền mạch."""
    wl = [{"win_start": 0, "win_end": 16000 * 10, "segs": [
        _seg(1.0, 1.02, "câu một"),      # end dồn (0.02s) → tile tới start câu 2
        _seg(4.0, 4.01, "câu hai"),      # end dồn → tile tới w1
    ]}]
    out = _abs_segments(wl, sr=16000)
    assert abs(out[0]["start"] - 1.0) < 1e-9 and abs(out[0]["end"] - 4.0) < 1e-9   # → start kế
    assert abs(out[1]["start"] - 4.0) < 1e-9 and abs(out[1]["end"] - 10.0) < 1e-9  # → w1
    # KHÔNG tràn qua lặng VAD: mọi end ≤ biên cửa sổ
    assert all(s["end"] <= 10.0 + 1e-9 for s in out)


def test_abs_segments_don_dieu():
    """start/end KHÔNG giảm toàn cục dù segment sau (cửa sổ lệch) trả local nhỏ."""
    wl = [
        {"win_start": 0, "win_end": 16000 * 5, "segs": [_seg(0.0, 3.0, "x")]},
        # cửa sổ 2 bắt đầu 1s (chồng lấn bất thường), local [0,1] → abs [1,2] < last_end 3
        {"win_start": 16000 * 1, "win_end": 16000 * 5, "segs": [_seg(0.0, 1.0, "y")]},
    ]
    out = _abs_segments(wl, sr=16000)
    starts = [s["start"] for s in out]
    ends = [s["end"] for s in out]
    assert starts == sorted(starts)         # đơn điệu
    assert ends == sorted(ends)
    assert out[1]["start"] >= out[0]["end"] - 1e-9   # không lùi


def test_abs_segments_clamp_trong_bien_cua_so():
    """Timestamp cục bộ tràn quá biên cửa sổ → clamp về [win_start, win_end]."""
    wl = [{"win_start": 0, "win_end": 16000 * 3,       # cửa sổ 3s
           "segs": [_seg(5.0, 6.0, "tràn")]}]          # local vượt cửa sổ
    out = _abs_segments(wl, sr=16000)
    assert out[0]["start"] <= 3.0 + 1e-9 and out[0]["end"] <= 3.0 + 1e-9


def test_abs_segments_bo_segment_rong():
    wl = [{"win_start": 0, "win_end": 16000 * 5,
           "segs": [_seg(0.0, 1.0, "   "), _seg(1.0, 2.0, "thật")]}]
    out = _abs_segments(wl, sr=16000)
    assert [s["text"] for s in out] == ["thật"]        # rỗng bị loại, text strip


def test_abs_segments_giu_metric_chat_luong():
    wl = [{"win_start": 0, "win_end": 16000 * 5,
           "segs": [{"start": 0.0, "end": 1.0, "text": "m", "avg_logprob": -0.7,
                     "no_speech_prob": 0.3, "compression_ratio": 2.1}]}]
    out = _abs_segments(wl, sr=16000)
    assert out[0]["avg_logprob"] == -0.7 and out[0]["no_speech_prob"] == 0.3
    assert out[0]["compression_ratio"] == 2.1


# ── engine routing (VAD-COMPAT-1, VAD-DUR-1) ─────────────────────────────────
class _FakeChunkModel:
    """Mỗi lần transcribe trả 1 segment cục bộ [0.5,1.0], text đánh số theo call."""
    def __init__(self):
        self.calls = []

    def transcribe(self, clip, **kwargs):
        self.calls.append({"n": len(clip), "kwargs": kwargs})
        i = len(self.calls)
        return iter([_Seg(0.5, 1.0, f"chunk{i}", -0.1, 0.0, 1.2)]), None


def _make_vad_engine(model, vad_chunk=True, min_duration_s=30.0):
    eng = object.__new__(FasterWhisperASR)
    eng._model = model
    eng._lang = "vi"; eng._load_s = 0.0
    eng._beam = 5; eng._vad = False; eng._vad_params = None; eng._condition = True
    eng._longform_min_s = 30.0
    eng._lf_beam = 5; eng._lf_vad = True; eng._lf_vad_params = None
    eng._lf_condition = False
    eng._vad_chunk = vad_chunk
    eng._vc_max_chunk_s = 28.0; eng._vc_min_silence_ms = 500
    eng._vc_speech_pad_ms = 200; eng._vc_min_duration_s = min_duration_s
    return eng


def test_vad_chunk_path_timestamp_tuyet_doi(monkeypatch):
    """Audio dài + cờ bật → mỗi cửa sổ VAD decode riêng, timestamp += offset biên."""
    captured = {}

    def fake_gst(audio, vad_options=None, sampling_rate=16000, **kw):
        captured["max_speech_duration_s"] = vad_options.max_speech_duration_s
        return [{"start": 0, "end": 16000 * 5},
                {"start": 16000 * 10, "end": 16000 * 15}]
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps", fake_gst)

    model = _FakeChunkModel()
    eng = _make_vad_engine(model)
    res = eng.transcribe(np.zeros(16000 * 35, dtype=np.float32), 16000)  # 35s ≥ 30
    segs = res.segments
    assert len(segs) == 2
    assert abs(segs[0]["start"] - 0.5) < 1e-6                 # cửa sổ 1 offset 0
    assert abs(segs[1]["start"] - 10.5) < 1e-6               # cửa sổ 2 offset 10s
    assert res.text == "chunk1 chunk2"
    # VAD-DUR-1: cap < 30s được truyền vào VadOptions
    assert captured["max_speech_duration_s"] == 28.0
    # 2 lần decode, mỗi lần beam/cond profile long-form, KHÔNG bật vad_filter lại
    assert len(model.calls) == 2
    for c in model.calls:
        assert c["kwargs"]["beam_size"] == 5
        assert c["kwargs"]["condition_on_previous_text"] is False
        assert "vad_filter" not in c["kwargs"]
        # VAD-DET-1: temperature=0 tắt thang fallback sample ngẫu nhiên → tất định
        assert c["kwargs"]["temperature"] == 0


def test_vad_chunk_bo_segment_lap(monkeypatch):
    """temp=0 tắt fallback phục hồi lặp → engine tự BỎ segment compression cao
    (chữ ký lặp) khỏi text+segments (VAD-TEXT-1: raw text không phình vì lặp)."""
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps",
                        lambda *a, **k: [{"start": 0, "end": 16000 * 5}])

    class _LoopModel:
        def transcribe(self, clip, **kwargs):
            return iter([
                _Seg(0.5, 1.0, "câu thật", -0.2, 0.0, 1.3),          # giữ
                _Seg(1.0, 2.0, "lặp lặp lặp lặp", -0.3, 0.0, 3.1),   # cr>2.4 → bỏ
            ]), None
    eng = _make_vad_engine(_LoopModel())
    res = eng.transcribe(np.zeros(16000 * 35, dtype=np.float32), 16000)
    assert [s["text"] for s in res.segments] == ["câu thật"]        # segment lặp bị loại
    assert "lặp" not in res.text


def test_vad_chunk_khong_speech_fallback_1call(monkeypatch):
    """VAD không thấy speech → decode 1-call trên toàn audio (trung thực, path cũ)."""
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps",
                        lambda *a, **k: [])
    model = _FakeChunkModel()
    eng = _make_vad_engine(model)
    eng.transcribe(np.zeros(16000 * 35, dtype=np.float32), 16000)
    assert len(model.calls) == 1                              # 1 call duy nhất
    assert model.calls[0]["n"] == 16000 * 35                 # trên TOÀN audio
    assert model.calls[0]["kwargs"]["vad_filter"] is True    # profile long-form
    # VAD-DET-1: fallback VẪN thuộc path vad_chunk → phải temperature=0 (tất định),
    # KHÔNG được thừa kế thang [0.0..1.0] mặc định (review bắt lỗ hổng này).
    assert model.calls[0]["kwargs"]["temperature"] == 0


def test_vad_chunk_off_di_path_cu(monkeypatch):
    """Cờ TẮT (mặc định) → luôn path 1-call, KHÔNG gọi VAD (VAD-COMPAT-1)."""
    called = {"gst": 0}
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps",
                        lambda *a, **k: called.__setitem__("gst", called["gst"] + 1) or [])
    model = _FakeChunkModel()
    eng = _make_vad_engine(model, vad_chunk=False)
    eng.transcribe(np.zeros(16000 * 35, dtype=np.float32), 16000)
    assert called["gst"] == 0                                 # VAD không được gọi
    assert len(model.calls) == 1


def test_vad_chunk_audio_ngan_di_path_cu(monkeypatch):
    """Audio < min_duration_s → path 1-call dù cờ bật (VAD-COMPAT-1)."""
    called = {"gst": 0}
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps",
                        lambda *a, **k: called.__setitem__("gst", called["gst"] + 1) or [])
    model = _FakeChunkModel()
    eng = _make_vad_engine(model, vad_chunk=True, min_duration_s=30.0)
    eng.transcribe(np.zeros(16000 * 5, dtype=np.float32), 16000)  # 5s < 30
    assert called["gst"] == 0
    assert len(model.calls) == 1
