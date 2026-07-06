"""Dựng eval set NGOÀI DOMAIN (OOD) từ VietMed test split — hội thoại y tế thật
(Host/Doctor/Patient, nhiều accent), KHÁC hẳn VIVOS đọc sách sạch mà PhoWhisper
đã train. Mục đích: biết WER THẬT của serving config ngoài vùng lạc quan VIVOS.

Lấy qua HuggingFace datasets-server API (không cần tải cả dataset):
stride-sample `--n` clip trải đều test split (3437 rows) để phủ speaker/accent.

    venv/bin/python scripts/build_eval_ood.py            # 50 clips -> data/eval_ood/
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://datasets-server.huggingface.co/rows"
DATASET = "leduckhai/VietMed"


def fetch_page(offset: int, length: int = 100) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default",
                                "split": "test", "offset": offset, "length": length})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
        return json.load(r)["rows"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--total", type=int, default=3437, help="số rows của test split")
    ap.add_argument("--out", default=str(ROOT / "data/eval_ood"))
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=29.0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)

    # Stride qua toàn split: mỗi clip lấy từ 1 offset cách đều nhau.
    stride = args.total // args.n
    entries, skipped = [], 0
    i = 0
    while len(entries) < args.n and i < args.n * 2:  # dư quota để bù clip bị loại
        offset = (i * stride) % args.total
        i += 1
        try:
            row = fetch_page(offset, 1)[0]["row"]
        except Exception as e:  # noqa: BLE001 — log-và-bỏ-qua như run loop chính
            print(f"  bỏ offset {offset}: {e}")
            skipped += 1
            continue
        dur, text = row.get("duration") or 0, (row.get("text") or "").strip()
        if not (args.min_dur <= dur <= args.max_dur) or not text:
            skipped += 1
            continue
        src = row["audio"][0]["src"]
        cid = f"vietmed_{row.get('utterance_id', f'off{offset}')}"
        wav_path = out / "clips" / f"{cid}.wav"
        with urllib.request.urlopen(src, timeout=120) as r:
            wav_path.write_bytes(r.read())
        entries.append({"id": cid,
                        "audio": str(wav_path.relative_to(ROOT)),
                        "text": text,
                        "duration": dur,
                        "accent": row.get("accent"),
                        "role": row.get("role")})
        if len(entries) % 10 == 0:
            print(f"  {len(entries)}/{args.n} clips")

    mpath = out / "ood.manifest.jsonl"
    mpath.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
                     + "\n", encoding="utf-8")
    total_dur = sum(e["duration"] for e in entries)
    print(f"Xong: {len(entries)} clips ({total_dur:.0f}s), bỏ {skipped} — {mpath}")

    (out / "README.md").write_text(
        "# Eval OOD — nguồn gốc & license\n\n"
        f"{len(entries)} clip stride-sampled từ **VietMed** test split "
        "([leduckhai/VietMed](https://huggingface.co/datasets/leduckhai/VietMed)) — "
        "hội thoại y tế thật, đa accent/vai. Domain KHÁC VIVOS (đọc sách sạch) nên "
        "WER đo ở đây phản ánh out-of-domain thật của serving config.\n\n"
        "License theo dataset gốc (dùng nghiên cứu phi thương mại, giữ attribution). "
        "Tạo lại: `venv/bin/python scripts/build_eval_ood.py`.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
