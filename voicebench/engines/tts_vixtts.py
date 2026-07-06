"""Adapter viXTTS (thinhlpg) — XTTS-v2 fine-tune tiếng Việt, voice cloning từ
ref audio. Cần Coqui TTS + checkpoint viXTTS + vinorm/underthesea để normalize
text VN trước khi synth.

ĐỊNH NGHĨA LATENCY (áp dụng cho mọi TTS voice-clone sau này): total_s = TOÀN BỘ
chi phí per-request cho cold speaker, GỒM get_conditioning_latents (forward GPU
qua speaker encoder + GPT conditioner) + inference. Loại conditioning ra ngoài
là báo thấp giả tạo — production mỗi request giọng mới đều trả chi phí này.

ttfa_s hiện None (batch). Khi bật streaming của XTTS -> điền ttfa ở đây.
"""
from __future__ import annotations

import logging
import numpy as np

from .base import TTSEngine
from ..interfaces import TTSResult, LatencyBreakdown
from ..timing import measure

logger = logging.getLogger(__name__)


class ViXTTS(TTSEngine):
    name = "vixtts"

    def __init__(self, model_path: str, config_path: str, device: str = "cuda",
                 language: str = "vi", normalize_vi_text: bool = True):
        try:
            import torch
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as e:
            raise ImportError("pip install coqui-tts  # fork idiap còn maintain, "
                              "TTS 0.22 gốc xung khắc transformers mới") from e
        import time
        t0 = time.perf_counter()
        cfg = XttsConfig(); cfg.load_json(config_path)
        self._model = Xtts.init_from_config(cfg)
        self._model.load_checkpoint(cfg, checkpoint_dir=model_path, use_deepspeed=False)
        if device == "cuda" and torch.cuda.is_available():
            self._model.cuda()
        # XttsConfig KHÔNG có output_sample_rate top-level (TTS 0.22): nằm ở
        # cfg.audio.output_sample_rate (và cfg.model_args.output_sample_rate).
        sr = (getattr(getattr(cfg, "audio", None), "output_sample_rate", None)
              or getattr(getattr(cfg, "model_args", None), "output_sample_rate", None))
        if sr is None:
            logger.warning("Không đọc được output_sample_rate từ XttsConfig "
                           "(cfg.audio/cfg.model_args) -> fallback 24000")
            sr = 24000
        self._sr = int(sr)
        self._lang = language
        self._norm = normalize_vi_text
        # coqui-tts chưa có 'vi' trong whitelist tokenizer (fork thinhlpg cho
        # viXTTS thêm đúng chỗ này) -> monkeypatch instance: tiền xử lý tối
        # thiểu tương đương multilingual_cleaners nhưng không đụng bảng
        # abbreviations/symbols (KeyError với 'vi'). Chữ số do vinorm lo (khi
        # có); ở đây không expand số.
        tok = getattr(self._model, "tokenizer", None)
        if tok is not None and hasattr(tok, "preprocess_text"):
            import re as _re
            tok.char_limits.setdefault("vi", 250)
            _orig = tok.preprocess_text

            def _preprocess(txt: str, lang: str, _orig=_orig):
                if lang.split("-")[0] == "vi":
                    return _re.sub(r"\s+", " ", txt.replace('"', "").lower()).strip()
                return _orig(txt, lang)

            tok.preprocess_text = _preprocess
        self._load_s = time.perf_counter() - t0

    def _prep_text(self, text: str) -> str:
        if not self._norm:
            return text
        try:
            from vinorm import TTSnorm
            return TTSnorm(text)
        except ImportError:
            logger.warning("Thiếu 'vinorm' -> dùng text thô (chữ số/viết tắt có thể đọc sai)")
            return text
        except OSError as e:
            # vinorm 2.x ship binary Linux x86-64 — trên macOS arm64 exec fail.
            # Không được chết cả synth vì normalizer: fallback text thô.
            logger.warning("vinorm không chạy được trên platform này (%s) -> text thô", e)
            return text

    def synthesize(self, text: str, ref_wav: np.ndarray, ref_sr: int) -> TTSResult:
        import tempfile
        from ..audio import save_wav
        text = self._prep_text(text)
        # Đo CẢ conditioning lẫn inference (định nghĩa total_s ở docstring module).
        # XTTS lấy conditioning latents từ file ref -> lưu tạm.
        with measure() as t:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                save_wav(f.name, ref_wav, ref_sr)
                gpt_lat, spk_emb = self._model.get_conditioning_latents(
                    audio_path=[f.name])
            out = self._model.inference(
                text=text, language=self._lang,
                gpt_cond_latent=gpt_lat, speaker_embedding=spk_emb,
            )
            wav = np.asarray(out["wav"], dtype=np.float32)
        dur = len(wav) / self._sr
        lat = LatencyBreakdown(total_s=t[0], media_dur_s=dur, model_load_s=self._load_s)
        return TTSResult(audio=wav, sample_rate=self._sr, latency=lat, out_dur_s=dur)
