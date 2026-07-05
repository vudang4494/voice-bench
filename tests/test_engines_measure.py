"""T2 fix: đo lường của engine — faster-whisper phải resample về 16k trước khi
đưa vào model; ViXTTS phải tính conditioning latents VÀO total_s.

Test bằng mock model (không cần faster-whisper/Coqui/GPU): tạo instance qua
object.__new__ rồi gắn attribute — transcribe/synthesize không import lib nặng.
"""
import time

import numpy as np

from voicebench.engines.asr_whisper import FasterWhisperASR
from voicebench.engines.tts_vixtts import ViXTTS


class _FakeWhisperModel:
    def __init__(self):
        self.received = None
        self.kwargs = None

    def transcribe(self, wav, **kwargs):
        self.received = wav
        self.kwargs = kwargs
        return iter(()), None


def _make_fw():
    eng = object.__new__(FasterWhisperASR)
    eng._model = _FakeWhisperModel()
    eng._lang = "vi"
    eng._load_s = 0.0
    eng._beam = 5
    eng._vad = False
    eng._vad_params = None
    eng._condition = True
    eng._longform_min_s = None
    return eng


def test_faster_whisper_resamples_441k_to_16k():
    eng = _make_fw()
    sr, dur = 44100, 1.5
    wav = np.zeros(int(sr * dur), dtype=np.float32)
    res = eng.transcribe(wav, sr)
    # model phải nhận đúng số mẫu 16k, KHÔNG phải waveform 44.1k thô
    assert abs(len(eng._model.received) - int(16000 * dur)) <= 2
    assert abs(res.audio_dur_s - dur) < 1e-6  # duration tính trên input gốc
    assert abs(res.latency.media_dur_s - dur) < 1e-6


def test_faster_whisper_16k_passthrough():
    eng = _make_fw()
    wav = np.zeros(16000, dtype=np.float32)
    eng.transcribe(wav, 16000)
    assert eng._model.received is wav  # không copy/resample thừa


class _FakeXttsModel:
    COND_S = 0.03
    INFER_S = 0.03

    def get_conditioning_latents(self, audio_path):
        time.sleep(self.COND_S)
        return "gpt_lat", "spk_emb"

    def inference(self, text, language, gpt_cond_latent, speaker_embedding):
        time.sleep(self.INFER_S)
        return {"wav": np.zeros(2400, dtype=np.float32)}


def test_vixtts_conditioning_counted_in_total_s():
    eng = object.__new__(ViXTTS)
    eng._model = _FakeXttsModel()
    eng._sr = 24000
    eng._lang = "vi"
    eng._norm = False
    eng._load_s = 0.0
    ref = np.zeros(16000, dtype=np.float32)
    res = eng.synthesize("xin chào", ref, 16000)
    # total_s phải >= conditioning + inference (cả hai trong cửa sổ đo)
    assert res.latency.total_s >= _FakeXttsModel.COND_S + _FakeXttsModel.INFER_S
    assert res.sample_rate == 24000
    assert res.out_dur_s == 2400 / 24000


def test_faster_whisper_decode_params_passed():
    eng = _make_fw()
    eng._beam = 1
    eng._vad = True
    eng._vad_params = {"min_silence_duration_ms": 300}
    eng._condition = False
    eng.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    k = eng._model.kwargs
    assert k["beam_size"] == 1
    assert k["vad_filter"] is True
    assert k["vad_parameters"] == {"min_silence_duration_ms": 300}
    assert k["condition_on_previous_text"] is False


def test_faster_whisper_no_vad_key_when_off():
    eng = _make_fw()
    eng.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert "vad_filter" not in eng._model.kwargs


def _add_longform(eng, min_s=30.0):
    eng._longform_min_s = min_s
    eng._lf_beam = 5
    eng._lf_vad = True
    eng._lf_vad_params = {"min_silence_duration_ms": 300, "speech_pad_ms": 400}
    eng._lf_condition = False
    return eng


def test_longform_profile_applied_over_threshold():
    eng = _add_longform(_make_fw())
    eng.transcribe(np.zeros(16000 * 31, dtype=np.float32), 16000)  # 31s >= 30s
    k = eng._model.kwargs
    assert k["vad_filter"] is True
    assert k["vad_parameters"]["min_silence_duration_ms"] == 300
    assert k["condition_on_previous_text"] is False


def test_short_clip_keeps_default_profile():
    eng = _add_longform(_make_fw())
    eng.transcribe(np.zeros(16000 * 2, dtype=np.float32), 16000)  # 2s < 30s
    k = eng._model.kwargs
    assert "vad_filter" not in k
    assert k["condition_on_previous_text"] is True
    assert k["beam_size"] == 5
