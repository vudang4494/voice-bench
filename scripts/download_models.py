"""Tải trước model ASR từ HuggingFace (mirror vudang449) về cache local.

Không bắt buộc — faster-whisper tự tải khi chạy lần đầu — nhưng chạy script này
sau khi clone repo để tách bước tải (chậm, cần mạng) khỏi bước chạy benchmark.

    venv/bin/python scripts/download_models.py            # small (serving mặc định, ~919MB)
    venv/bin/python scripts/download_models.py --large    # thêm large (accuracy, ~2.9GB)
    venv/bin/python scripts/download_models.py --tts      # viXTTS -> models/viXTTS (~1.9GB)
    venv/bin/python scripts/download_models.py --all      # tất cả (kèm chunkformer)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "small": "vudang449/PhoWhisper-small-ct2",       # configs/service.yaml (mặc định)
    "large": "vudang449/PhoWhisper-large-ct2",       # configs/service.large.yaml
    "chunkformer": "khanhld/chunkformer-large-vie",  # configs/service.chunkformer.yaml
}
TTS_REPO = "capleaf/viXTTS"  # configs/service.tts.yaml — cần local_dir cố định


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--large", action="store_true", help="tải bản large thay vì small")
    ap.add_argument("--chunkformer", action="store_true",
                    help="tải ChunkFormer (ứng viên serving mới)")
    ap.add_argument("--tts", action="store_true", help="tải viXTTS về models/viXTTS")
    ap.add_argument("--all", action="store_true", help="tải tất cả")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Thiếu huggingface_hub — cài deps trước: pip install -r requirements.txt")
        return 1

    if args.all:
        keys = list(MODELS)
    else:
        keys = ([k for k, on in (("large", args.large),
                                 ("chunkformer", args.chunkformer)) if on]
                or ([] if args.tts else ["small"]))

    for k in keys:
        repo = MODELS[k]
        print(f"Tải {k}: {repo} ...")
        path = snapshot_download(repo)
        print(f"  -> {path}")

    if args.tts or args.all:
        dst = ROOT / "models/viXTTS"
        print(f"Tải tts: {TTS_REPO} -> {dst} ...")
        snapshot_download(TTS_REPO, local_dir=str(dst))
        print(f"  -> {dst}")

    print("Xong. Chạy service: venv/bin/uvicorn voicebench.service:app --port 8386")
    return 0


if __name__ == "__main__":
    sys.exit(main())
