"""Tải trước model ASR từ HuggingFace (mirror vudang449) về cache local.

Không bắt buộc — faster-whisper tự tải khi chạy lần đầu — nhưng chạy script này
sau khi clone repo để tách bước tải (chậm, cần mạng) khỏi bước chạy benchmark.

    venv/bin/python scripts/download_models.py            # small (serving mặc định, ~919MB)
    venv/bin/python scripts/download_models.py --large    # thêm large (accuracy, ~2.9GB)
    venv/bin/python scripts/download_models.py --all      # cả hai
"""
from __future__ import annotations

import argparse
import sys

MODELS = {
    "small": "vudang449/PhoWhisper-small-ct2",  # configs/service.yaml (mặc định)
    "large": "vudang449/PhoWhisper-large-ct2",  # configs/service.large.yaml
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--large", action="store_true", help="tải bản large thay vì small")
    ap.add_argument("--all", action="store_true", help="tải cả small lẫn large")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Thiếu huggingface_hub — cài deps trước: pip install -r requirements.txt")
        return 1

    if args.all:
        keys = ["small", "large"]
    elif args.large:
        keys = ["large"]
    else:
        keys = ["small"]

    for k in keys:
        repo = MODELS[k]
        print(f"Tải {k}: {repo} ...")
        path = snapshot_download(repo)
        print(f"  -> {path}")
    print("Xong. Chạy service: venv/bin/uvicorn voicebench.service:app --port 8386")
    return 0


if __name__ == "__main__":
    sys.exit(main())
