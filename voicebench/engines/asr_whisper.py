"""Adapter faster-whisper (CT2). Dùng cho cả whisper gốc lẫn PhoWhisper bản
convert CT2 (vd diepho/PhoWhisper-small-ct2) — đường serving chính trên CPU/Mac.

Decode params đi qua config (asr.kwargs), đã tune cho long-form trên Mac Mini M4
(xem ROADMAP T7 + runs/tune_longform_*.json): whisper mất nội dung ở ranh giới
cửa sổ 30s khi decode tuần tự; vad_filter cắt theo khoảng lặng thật + không
carry context lỗi giữa các đoạn (condition_on_previous_text=False) sửa việc này.

EXTERNAL VAD CHUNKING (opt-in `vad_chunk`, xem scripts/rules.vad_chunk.yaml):
PhoWhisper fine-tune câu ngắn nên timestamp segment NỘI BỘ trên audio dài KHÔNG
tin cậy (dồn ~0.3s + segment bịa-37.9s). Bật vad_chunk → audio dài được cắt theo
VAD (Silero bundle sẵn trong faster-whisper) thành cửa sổ < 30s, transcribe TỪNG
cửa sổ RIÊNG, gán timestamp TUYỆT ĐỐI theo biên VAD. Kết quả: segments[].start/end
ĐÁNG TIN (đơn điệu, đúng thời điểm) cho timeline sentiment. Cờ TẮT = hành vi cũ
y hệt (VAD-COMPAT-1).
"""
from __future__ import annotations

import logging
import numpy as np

from .base import ASREngine
from ..interfaces import ASRResult, LatencyBreakdown
from ..timing import measure
from ..audio import duration_s, resample

logger = logging.getLogger(__name__)

# Ngưỡng compression_ratio (mặc định của whisper) để loại segment LẶP trong path
# VAD-chunk: temp=0 tắt thang temperature fallback (vốn tự phục hồi lặp), nên
# engine tự bỏ segment có chữ ký lặp (bù đúng phần fallback từng làm). = ngưỡng
# LF_HALLUC_COMPRESSION_MAX của bridge → raw ASR text nhất quán với sentiment.
_VAD_CHUNK_CR_MAX = 2.4


def _abs_segments(windows_local: list, sr: int = 16000) -> list:
    """Ghép segment cục bộ của từng cửa sổ VAD thành segment TUYỆT ĐỐI, timestamp
    đơn điệu + trong biên cửa sổ VAD (nền của VAD-MONO-1/BND-1). PURE — test được
    không cần model.

    windows_local: [{"win_start": sample, "win_end": sample, "segs": [seg,...]}]
      với seg = {"start","end" (giây CỤC BỘ trong clip), "text","avg_logprob",
      "no_speech_prob","compression_ratio"}. Cửa sổ giả định đã sort theo thời gian.

    START tuyệt đối = biên VAD (w0) + start cục bộ whisper — ĐÁNG TIN (whisper bắt
    onset tốt). END cục bộ whisper hay DỒN (start≈end) trên PhoWhisper → thay vì
    giữ span 0.02s vô dụng, TILE end trong cửa sổ: end_i = start_{i+1}, end cuối =
    w1. Không bao giờ tràn QUA khoảng lặng VAD (w1 là biên nói thật) → timeline
    liền mạch, đặt câu đúng khoảnh khắc, mà không bịa quá vùng nói.

    Trả list segment {start,end (giây TUYỆT ĐỐI), text (stripped), + 3 metric}.
    """
    out: list = []
    prev_end = 0.0
    for w in windows_local:
        w0 = float(w["win_start"]) / sr
        w1 = float(w["win_end"]) / sr
        # 1) START tuyệt đối cho từng seg non-empty: clamp trong [floor, w1], floor
        #    đảm bảo đơn điệu TOÀN CỤC (>= end segment trước + >= w0).
        segs: list = []
        floor = max(w0, prev_end)
        for s in w["segs"]:
            txt = (s.get("text") or "").strip()
            if not txt:
                continue                      # bỏ segment rỗng (giữ timeline sạch)
            a = min(max(w0 + float(s["start"]), floor), w1)
            floor = a                          # start sau không lùi trước start trước
            segs.append({"start": a, "text": txt,
                         "avg_logprob": float(s["avg_logprob"]),
                         "no_speech_prob": float(s["no_speech_prob"]),
                         "compression_ratio": float(s["compression_ratio"])})
        # 2) TILE end trong cửa sổ (không tràn qua lặng VAD)
        for i, seg in enumerate(segs):
            end = segs[i + 1]["start"] if i + 1 < len(segs) else w1
            seg["end"] = max(end, seg["start"])
            out.append(seg)
        if segs:
            prev_end = segs[-1]["end"]
    return out


class FasterWhisperASR(ASREngine):
    name = "faster-whisper"

    def __init__(self, model_id: str = "large-v3", device: str = "cuda",
                 compute_type: str = "int8_float16", language: str = "vi",
                 beam_size: int = 5, vad_filter: bool = False,
                 vad_parameters: dict | None = None,
                 condition_on_previous_text: bool = True,
                 longform_min_s: float | None = None,
                 longform: dict | None = None,
                 vad_chunk: bool = False,
                 vad_chunk_params: dict | None = None):
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
        # EXTERNAL VAD CHUNKING (opt-in). Mặc định TẮT = no-op tuyệt đối
        # (VAD-COMPAT-1). Params đồng bộ scripts/rules.vad_chunk.yaml.
        self._vad_chunk = bool(vad_chunk)
        vc = dict(vad_chunk_params or {})
        self._vc_max_chunk_s = float(vc.get("max_chunk_s", 28.0))
        self._vc_min_silence_ms = int(vc.get("min_silence_ms", 500))
        self._vc_speech_pad_ms = int(vc.get("speech_pad_ms", 200))
        self._vc_min_duration_s = float(vc.get("min_duration_s", 30.0))

    def transcribe(self, wav: np.ndarray, sr: int) -> ASRResult:
        dur = duration_s(wav, sr)
        # Chọn profile theo độ dài: decode tuần tự nhiều cửa sổ 30s mất nội dung
        # ở ranh giới (T7), nên request dài cần params khác request ngắn.
        # Tất cả state chỉ đọc sau __init__ -> thread-safe với threadpool service.
        is_long = self._longform_min_s is not None and dur >= self._longform_min_s
        if is_long:
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
            # ── EXTERNAL VAD CHUNKING: audio dài + cờ bật → cắt theo VAD, decode
            #    từng cửa sổ < 30s → timestamp tuyệt đối đáng tin (VAD-*). Cờ tắt
            #    hoặc audio ngắn → path 1-call CŨ (VAD-COMPAT-1). getattr phòng
            #    engine dựng qua object.__new__ (test cũ) thiếu attr → mặc định TẮT.
            vad_chunk = getattr(self, "_vad_chunk", False)
            vc_min_dur = getattr(self, "_vc_min_duration_s", 30.0)
            if vad_chunk and dur >= vc_min_dur:
                text, segments = self._transcribe_vad_chunks(
                    wav16, beam=self._lf_beam, cond=self._lf_condition)
            else:
                text, segments = self._transcribe_single(
                    wav16, beam=beam, vad=vad, vad_params=vad_params, cond=cond)
        lat = LatencyBreakdown(total_s=t[0], media_dur_s=dur, model_load_s=self._load_s)
        return ASRResult(text=text, latency=lat, audio_dur_s=dur, segments=segments)

    def _transcribe_single(self, wav16: np.ndarray, beam: int, vad: bool,
                           vad_params: dict | None, cond: bool,
                           temperature=None) -> tuple:
        """Path 1-call CŨ (giữ nguyên byte-exact — VAD-COMPAT-1). Trả (text, segments).
        temperature=None → KHÔNG truyền (giữ thang fallback mặc định, byte-exact cũ);
        =0 → tất định (dùng cho fallback của path vad_chunk, xem _transcribe_vad_chunks)."""
        kwargs: dict = {"language": self._lang, "beam_size": beam,
                        "condition_on_previous_text": cond}
        if vad:
            kwargs["vad_filter"] = True
            if vad_params:
                kwargs["vad_parameters"] = vad_params
        if temperature is not None:
            kwargs["temperature"] = temperature
        seg_iter, _ = self._model.transcribe(wav16, **kwargs)
        # faster-whisper trả generator -> vật chất hoá 1 lần (dùng cho cả text
        # lẫn segments). `text` giữ NGUYÊN cách ghép cũ (raw s.text) để WER
        # tái lập bit-exact; segments strip text cho sentiment sạch đầu vào.
        segs_raw = list(seg_iter)
        text = " ".join(s.text for s in segs_raw).strip()
        # Kèm chỉ số chất lượng chuẩn của whisper (avg_logprob/no_speech_prob/
        # compression_ratio) để bridge lọc hallucination — PhoWhisper trên
        # audio dài hay bịa nội dung ở đuôi. timestamp segment KHÔNG tin cậy
        # (PhoWhisper fine-tune câu ngắn) nên sentiment long-form gộp theo ĐỘ
        # DÀI TEXT, không dùng dur; start/end giữ lại best-effort để tham khảo.
        segments = [{"start": float(s.start), "end": float(s.end),
                     "text": s.text.strip(),
                     "avg_logprob": float(s.avg_logprob),
                     "no_speech_prob": float(s.no_speech_prob),
                     "compression_ratio": float(s.compression_ratio)}
                    for s in segs_raw]
        return text, segments

    def _transcribe_vad_chunks(self, wav16: np.ndarray, beam: int, cond: bool) -> tuple:
        """Cắt audio theo VAD (Silero) thành cửa sổ ≤ max_chunk_s < 30s, transcribe
        TỪNG cửa sổ RIÊNG, gán timestamp TUYỆT ĐỐI theo biên VAD → segments đáng
        tin (VAD-MONO/DUR/BND/COV). Không collect_chunks rồi 1-call: tránh lỗi
        ranh giới 30s (T7) + không context-bleed qua join nhân tạo. Trả (text, segments).
        """
        from faster_whisper.vad import get_speech_timestamps, VadOptions

        opts = VadOptions(min_silence_duration_ms=self._vc_min_silence_ms,
                          speech_pad_ms=self._vc_speech_pad_ms,
                          max_speech_duration_s=self._vc_max_chunk_s)
        # get_speech_timestamps: cửa sổ [start,end] theo SAMPLE, đã đệm speech_pad,
        # đã tách speech > max_speech_duration_s tại điểm lặng, non-overlap, sorted.
        windows = get_speech_timestamps(wav16, opts, sampling_rate=16000)
        if not windows:
            # Không phát hiện speech → decode 1-call (trung thực, không bịa segment).
            # temperature=0 BẮT BUỘC: bất biến VAD-DET-1 là "KHÔNG path decode nào
            # chạm temperature>0". Fallback này vẫn thuộc path vad_chunk nên phải
            # tất định — thiếu temperature=0 sẽ thừa kế thang [0.0..1.0] mặc định →
            # sample → phá determinism trên clip dài near-silent (review bắt).
            return self._transcribe_single(
                wav16, beam=beam, vad=self._lf_vad,
                vad_params=self._lf_vad_params, cond=cond, temperature=0)
        windows_local: list = []
        for w in windows:
            s0, s1 = int(w["start"]), int(w["end"])
            clip = wav16[s0:s1]
            if len(clip) == 0:
                continue
            # Chunk đã sạch (VAD-bounded) → KHÔNG bật vad_filter lại (thừa, có thể
            # tái dồn timestamp). Decode với beam/cond profile long-form.
            # temperature=0 (TẮT thang fallback [0.0..1.0]) để TẤT ĐỊNH (VAD-DET-1):
            # ở temp>0 faster-whisper SAMPLE (sampling_temperature, best_of) — CT2
            # không nhận seed mỗi call nên mẫu rút KHÁC nhau mỗi lần chạy (đã đo:
            # temp=0.6 → 3 output khác; set_random_seed KHÔNG cứu). Ở temp=0 là
            # beam/ARGMAX, KHÔNG sample → tất định (int8 GEMM cộng dồn int32, bằng
            # bit giữa các lần chạy, KHÔNG phải nguồn nhiễu). Bất biến: KHÔNG path
            # decode nào (kể cả fallback không-speech ở trên) được chạm temp>0.
            seg_iter, _ = self._model.transcribe(
                clip, language=self._lang, beam_size=beam,
                condition_on_previous_text=cond, temperature=0)
            # temp=0 mất khả năng phục hồi lặp của fallback → engine tự BỎ segment
            # có chữ ký lặp (compression_ratio cao) đúng như fallback từng làm.
            local = [{"start": float(s.start), "end": float(s.end), "text": s.text,
                      "avg_logprob": float(s.avg_logprob),
                      "no_speech_prob": float(s.no_speech_prob),
                      "compression_ratio": float(s.compression_ratio)}
                     for s in seg_iter
                     if float(s.compression_ratio) <= _VAD_CHUNK_CR_MAX]
            windows_local.append({"win_start": s0, "win_end": s1, "segs": local})
        segments = _abs_segments(windows_local, sr=16000)
        text = " ".join(s["text"] for s in segments).strip()
        return text, segments
