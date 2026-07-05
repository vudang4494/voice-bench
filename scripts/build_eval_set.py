"""Build bộ eval từ VIVOS test (mirror quocanh34/viet_vivos, HF dataset viewer).

Lấy mẫu STRIDE trên toàn split (760 dòng) thay vì lấy tuần tự từ đầu — tránh
bias speaker (đầu split chỉ có 2/19 speakers; xem review 2026-07-05).

Sinh ra trong data/eval/:
- clips/<id>.wav            : từng clip 16k mono (cho eval per-utterance)
- eval_3min.manifest.jsonl  : {"id","audio","dur_s","text"} mỗi dòng
- eval_3min.wav / .mp3      : ghép clips + khoảng lặng GAP (cho eval long-form)
- eval_3min.txt             : transcript ghép (ref cho long-form)

Dùng:
    venv/bin/python scripts/build_eval_set.py [--target-s 178] [--gap-s 0.35] [--seed 42]
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import subprocess
import urllib.request

import numpy as np
import soundfile as sf

BASE = ("https://datasets-server.huggingface.co/rows"
        "?dataset=quocanh34%2Fviet_vivos&config=default&split=test")
SR = 16000
N_TOTAL = 760  # số dòng split test của mirror


def fetch_rows(offset: int, length: int) -> list[dict]:
    with urllib.request.urlopen(f"{BASE}&offset={offset}&length={length}",
                                timeout=90) as r:
        return json.load(r)["rows"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-s", type=float, default=178.0)
    ap.add_argument("--gap-s", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/eval")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Stride qua toàn split: mỗi block 100 dòng lấy vài dòng đầu (offset ngẫu
    # nhiên theo seed) -> phủ đều các speaker (split xếp theo speaker).
    rng = np.random.default_rng(args.seed)
    picks_per_block = 8
    clips, texts, manifest, total = [], [], [], 0.0
    for block in range(0, N_TOTAL, 100):
        if total >= args.target_s:
            break
        rows = fetch_rows(block, 100)
        idx = rng.choice(len(rows), size=min(picks_per_block, len(rows)),
                         replace=False)
        for i in sorted(int(x) for x in idx):
            row = rows[i]["row"]
            txt = row["transcription"].strip()
            if not txt:
                continue
            url = row["audio"][0]["src"]
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    wav, sr = sf.read(io.BytesIO(resp.read()),
                                      dtype="float32", always_2d=False)
            except Exception as e:  # noqa: BLE001
                print("skip (dl err):", e)
                continue
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != SR:
                print("skip sr", sr)
                continue
            cid = f"vivos-test-r{block + i:04d}"
            sf.write(clip_dir / f"{cid}.wav", wav, SR)
            clips.append(wav)
            texts.append(txt)
            manifest.append({"id": cid, "audio": f"{args.out_dir}/clips/{cid}.wav",
                             "dur_s": round(len(wav) / SR, 2), "text": txt})
            total += len(wav) / SR
            if total >= args.target_s:
                break
        print(f"block {block}: {len(clips)} clips, {total:.1f}s")

    gap = np.zeros(int(args.gap_s * SR), dtype=np.float32)
    parts = []
    for c in clips:
        parts += [c, gap]
    full = np.concatenate(parts[:-1])
    sf.write(out_dir / "eval_3min.wav", full, SR)
    (out_dir / "eval_3min.txt").write_text(" ".join(texts) + "\n", encoding="utf-8")
    with open(out_dir / "eval_3min.manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(out_dir / "eval_3min.wav"),
                    "-codec:a", "libmp3lame", "-q:a", "3",
                    str(out_dir / "eval_3min.mp3")], check=True)
    print(f"XONG: {len(clips)} clips, speech {total:.1f}s, "
          f"file ghép {len(full) / SR:.1f}s ({len(full) / SR / 60:.2f} min)")


if __name__ == "__main__":
    main()
