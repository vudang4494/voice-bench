"""HTTP service STT/TTS trên engine registry của voice-bench.

Thiết kế:
- Engine hoán đổi qua config YAML (giống run_benchmark) — service chỉ là lớp HTTP mỏng.
- Latency trả về tách lớp: decode_s (đọc file upload) + latency engine
  (total_s đã device-sync, model_load_s riêng, rtf) + server_total_s.
  Client tự đo wall time HTTP để có lớp network/overhead.
- Warmup lúc startup (loại cold-start khỏi request đầu).
- TTS: chỉ bật khi config có mục `tts` (vd configs/service.tts.yaml); không có
  -> 503 nói rõ lý do. /v1/tts nhận multipart form: field `text` + file
  `ref_audio` (tùy chọn, để voice-clone) — KHÔNG nhận path server-side.

Chạy:
    venv/bin/uvicorn voicebench.service:app --host 127.0.0.1 --port 8386
    (config qua env VOICEBENCH_SERVICE_CONFIG, mặc định configs/service.yaml)
"""
from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .audio import duration_s
from .engines import build_asr, build_tts
from .interfaces import ASRResult

logger = logging.getLogger("voicebench.service")

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "configs" / "service.yaml")


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode wav/mp3/flac/ogg... -> (float32 mono, sr gốc).

    soundfile (libsndfile>=1.2 đọc được mp3) trước; fallback PyAV (m4a/webm...).
    KHÔNG resample ở đây — engine tự resample trong cửa sổ đo (chi phí thật).
    """
    try:
        import soundfile as sf
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr)
    except Exception:  # noqa: BLE001 — thử decoder rộng hơn
        pass
    try:
        import av
        chunks: list[np.ndarray] = []
        sr = None
        with av.open(io.BytesIO(data)) as container:
            resampler = av.AudioResampler(format="flt", layout="mono")
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    sr = rf.sample_rate
                    chunks.append(rf.to_ndarray().reshape(-1))
        if not chunks or not sr:
            raise ValueError("PyAV không decode được frame nào")
        return np.concatenate(chunks).astype(np.float32), int(sr)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Không decode được audio: {e}") from e


def _load_config(path: str | None) -> dict:
    cfg_path = path or os.environ.get("VOICEBENCH_SERVICE_CONFIG", DEFAULT_CONFIG)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {cfg_path} phải là YAML mapping, được {type(cfg).__name__}")
    return cfg


def create_app(config: dict | None = None, asr_engine=None, tts_engine=None) -> FastAPI:
    """App factory. asr_engine/tts_engine truyền sẵn để test (bỏ qua config build)."""
    cfg = config if config is not None else _load_config(None)
    app = FastAPI(title="voicebench service", version="0.1.0")
    state = {"asr": asr_engine, "tts": tts_engine, "warmed": False,
             "asr_cfg": cfg.get("asr", {}), "tts_cfg": cfg.get("tts")}

    @app.on_event("startup")
    def _startup() -> None:
        if state["asr"] is None:
            a = state["asr_cfg"]
            if not a:
                raise RuntimeError("Config thiếu mục 'asr'")
            logger.info("Build ASR %s ...", a.get("name"))
            state["asr"] = build_asr(a["name"], a.get("kwargs"))
        if state["tts"] is None and state["tts_cfg"]:
            t = state["tts_cfg"]
            logger.info("Build TTS %s ...", t.get("name"))
            state["tts"] = build_tts(t["name"], t.get("kwargs"))
        n_warm = int(cfg.get("warmup", 2))
        if n_warm > 0:
            logger.info("Warmup ASR x%d ...", n_warm)
            silence = np.zeros(16000 * 2, dtype=np.float32)
            state["asr"].warmup(silence, 16000, n=n_warm)
            if state["tts"] is not None:
                state["tts"].warmup("xin chào", silence, 16000, n=n_warm)
        state["warmed"] = True
        logger.info("Service sẵn sàng.")

    @app.get("/health")
    def health() -> dict:
        asr = state["asr"]
        return {
            "status": "ok" if state["warmed"] else "starting",
            "asr": {
                "name": getattr(asr, "name", None),
                "model": state["asr_cfg"].get("kwargs", {}).get("model_id"),
                "model_load_s": round(getattr(asr, "_load_s", float("nan")), 3)
                if asr is not None else None,
                "warmed": state["warmed"],
            },
            "tts": {"available": state["tts"] is not None,
                    "reason": None if state["tts"] is not None
                    else "Chưa cấu hình TTS engine (dùng config có mục 'tts', "
                         "vd configs/service.tts.yaml)"},
        }

    max_upload = int(float(cfg.get("max_upload_mb", 100)) * 1024 * 1024)

    # CHÚ Ý: endpoint là `def` (KHÔNG async) có chủ đích — engine.transcribe/
    # synthesize là CPU-bound blocking; FastAPI chạy `def` handler trong
    # threadpool nên /health và request khác không bị đứng. `async def` ở đây
    # sẽ chặn toàn bộ event loop trong suốt inference (đã đo: /health 2.5s).
    @app.post("/v1/asr")
    def asr_endpoint(file: UploadFile = File(...)) -> dict:
        if state["asr"] is None:
            raise HTTPException(status_code=503, detail="ASR chưa sẵn sàng")
        data = file.file.read(max_upload + 1)
        if len(data) > max_upload:
            raise HTTPException(status_code=413,
                                detail=f"File vượt giới hạn {max_upload // (1024*1024)}MB")
        if not data:
            raise HTTPException(status_code=400, detail="File rỗng")
        t0 = time.perf_counter()
        wav, sr = decode_audio_bytes(data)
        decode_s = time.perf_counter() - t0
        if len(wav) == 0:
            raise HTTPException(status_code=400, detail="Audio 0 mẫu sau decode")
        result: ASRResult = state["asr"].transcribe(wav, sr)
        server_total_s = time.perf_counter() - t0
        return {
            "text": result.text,
            "audio": {"duration_s": round(duration_s(wav, sr), 3),
                      "sample_rate_in": sr, "bytes": len(data),
                      "filename": file.filename},
            "latency": {
                "decode_s": round(decode_s, 4),
                "infer_total_s": round(result.latency.total_s, 4),
                "rtf": round(result.latency.rtf, 4),
                "model_load_s": round(result.latency.model_load_s, 3),
                "server_total_s": round(server_total_s, 4),
            },
            "engine": getattr(state["asr"], "name", None),
        }

    # multipart form (KHÔNG nhận path server-side — client upload ref audio như
    # /v1/asr upload file; nhận path tùy ý là lỗ đọc file nội bộ, ROADMAP T4).
    @app.post("/v1/tts")
    def tts_endpoint(text: str = Form(...),
                     ref_audio: UploadFile | None = File(None)) -> Response:
        if state["tts"] is None:
            raise HTTPException(
                status_code=503,
                detail="TTS chưa khả dụng trên host này: chạy config có mục 'tts' "
                       "(vd configs/service.tts.yaml, checkpoint: "
                       "scripts/download_models.py --tts).")
        text = text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="'text' phải khác rỗng")
        if ref_audio is not None:
            data = ref_audio.file.read(max_upload + 1)
            if len(data) > max_upload:
                raise HTTPException(status_code=413,
                                    detail=f"ref_audio vượt giới hạn "
                                           f"{max_upload // (1024*1024)}MB")
            if not data:
                raise HTTPException(status_code=400, detail="ref_audio rỗng")
            ref_wav, ref_sr = decode_audio_bytes(data)  # 400 nếu không decode được
            if len(ref_wav) == 0:
                raise HTTPException(status_code=400,
                                    detail="ref_audio 0 mẫu sau decode")
        else:
            ref_wav, ref_sr = np.zeros(16000, dtype=np.float32), 16000
        r = state["tts"].synthesize(text, ref_wav, ref_sr)
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, r.audio, r.sample_rate, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav",
                        headers={"X-Infer-Total-S": f"{r.latency.total_s:.4f}",
                                 "X-RTF": f"{r.latency.rtf:.4f}"})

    return app


def app_factory() -> FastAPI:
    """Cho `uvicorn voicebench.service:app_factory --factory`."""
    return create_app()


# Instance mặc định cho `uvicorn voicebench.service:app`. Guard để import module
# (vd chỉ lấy decode_audio_bytes trong test) không chết khi config thiếu/hỏng.
try:
    app = create_app()
except Exception as e:  # noqa: BLE001
    logger.warning("Không tạo được app mặc định (%s) — dùng create_app()/app_factory.", e)
    app = None
