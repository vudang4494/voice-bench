"""Orchestrator: chạy pipeline round-trip trên 1 dataset, ghi per-sample JSONL.

Luồng mỗi mẫu:
  ref_audio --[ASR#1]--> asr_text        (đo ASR acc = WER(ref_text, asr_text))
  ref_text  --[TTS clone ref voice]--> out_audio
  out_audio --[ASR#2]--> roundtrip_text  (đo TTS intelligibility)
  speaker_sim(ref_audio, out_audio), mos(out_audio)

MẶC ĐỊNH tts_input = ref_text (ground truth) để CÔ LẬP lỗi TTS khỏi lỗi ASR#1.
Đổi sang 'asr_text' nếu muốn đo đúng pipeline echo thực tế (lỗi tích luỹ).

Reproducibility: dump run_meta.json (config + hardware + versions + commit + seed).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from .engines import build_asr, build_tts
from .audio import load_wav, save_wav
from .interfaces import SampleRecord

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voicebench.run")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _hardware() -> dict:
    hw = {"platform": sys.platform}
    try:
        import torch
        hw["torch"] = torch.__version__
        hw["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        hw["torch"] = None
    return hw


def load_manifest(path: str) -> list[dict]:
    """JSONL: {'id','audio','text'} mỗi dòng."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(config_path: str) -> Path:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    out_dir = Path(cfg.get("output_dir", "runs")) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dump = out_dir / "tts_audio"
    if cfg.get("save_tts_audio", False):
        audio_dump.mkdir(exist_ok=True)

    logger.info("Build ASR=%s TTS=%s", cfg["asr"]["name"], cfg["tts"]["name"])
    asr = build_asr(cfg["asr"]["name"], cfg["asr"].get("kwargs"))
    tts = build_tts(cfg["tts"]["name"], cfg["tts"].get("kwargs"))

    manifest = load_manifest(cfg["dataset"]["manifest"])
    if cfg["dataset"].get("limit"):
        manifest = manifest[: int(cfg["dataset"]["limit"])]
    logger.info("Dataset: %d mẫu", len(manifest))

    tts_src = cfg.get("tts_input_source", "ref_text")  # ref_text | asr_text
    do_spk = cfg.get("speaker_sim", True)
    # Engine không voice-clone (vd viettts-http bỏ qua ref_wav): speaker_sim
    # vô nghĩa — tự tắt qua capability flag, ghi null cho MỌI sample.
    if do_spk and not getattr(tts, "supports_cloning", True):
        logger.warning("TTS '%s' không voice-clone -> speaker_sim = null", tts.name)
        do_spk = False
    do_mos = cfg.get("mos", False)
    # keep_tone không dùng ở đây: chấm điểm nằm ở report.py, đọc lại từ
    # run_meta.json (config được dump nguyên vẹn bên dưới).

    # Warmup trên mẫu đầu (loại cold-start/CUDA-init khỏi số đo).
    if manifest:
        w_wav, w_sr = load_wav(manifest[0]["audio"])
        n_warm = int(cfg.get("warmup", 2))
        logger.info("Warmup x%d ...", n_warm)
        asr.warmup(w_wav, w_sr, n=n_warm)
        tts.warmup(manifest[0]["text"], w_wav, w_sr, n=n_warm)

    results_path = out_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(manifest):
            try:
                ref_wav, ref_sr = load_wav(row["audio"])
                r_asr = asr.transcribe(ref_wav, ref_sr)
                tts_text = row["text"] if tts_src == "ref_text" else r_asr.text
                r_tts = tts.synthesize(tts_text, ref_wav, ref_sr)
                r_rt = asr.transcribe(r_tts.audio, r_tts.sample_rate)

                spk = None
                if do_spk:
                    from .metrics.speaker_sim import speaker_similarity
                    spk = speaker_similarity(ref_wav, ref_sr,
                                             r_tts.audio, r_tts.sample_rate)
                mos = None
                if do_mos:
                    from .metrics.mos import predict_mos
                    mos = predict_mos(r_tts.audio, r_tts.sample_rate)

                if cfg.get("save_tts_audio", False):
                    save_wav(str(audio_dump / f"{row['id']}.wav"),
                             r_tts.audio, r_tts.sample_rate)

                rec = SampleRecord(
                    sample_id=str(row["id"]),
                    ref_text=row["text"],
                    asr_text=r_asr.text,
                    roundtrip_text=r_rt.text,
                    asr_latency=r_asr.latency.to_dict(),
                    tts_latency=r_tts.latency.to_dict(),
                    speaker_sim=spk,
                    mos=mos,
                    extra={"tts_input_source": tts_src},
                )
                fout.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                fout.flush()
                if (i + 1) % 10 == 0:
                    logger.info("  %d/%d", i + 1, len(manifest))
            except Exception as e:  # noqa: BLE001 — log mẫu lỗi, không chết cả run
                logger.exception("Mẫu %s lỗi: %s", row.get("id"), e)

    meta = {
        "seed": seed, "config": cfg, "hardware": _hardware(),
        "git_commit": _git_commit(), "timestamp": time.time(),
        "n_samples": len(manifest),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Xong -> %s", results_path)
    return results_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
