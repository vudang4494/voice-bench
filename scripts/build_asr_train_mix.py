"""Build train mix v1 cho full fine-tune ASR (ST-1) — theo configs/data_rules.asr.yaml.

Nguồn v1 (đều CC-BY-4.0, xem ATTRIBUTIONS.md):
  vlsp2020_vinai_100h (~100h) + fpt_fosd (~30h) + infore1_25hours (~25h)

Thiết kế:
- Stream row-group parquet qua HfFileSystem (không tải full shard xuống disk).
- Deterministic + resumable: mỗi shard xong ghi manifest phần riêng
  (parts/<tag>-<shard>.jsonl); chạy lại bỏ qua shard đã có part. KHÔNG RNG.
- Val cắt TRƯỚC filter: split gán bằng sha1(row_id) % 100 < 1 (~1%) trên toàn
  bộ row thô — filter áp như nhau cho cả 2 phía (bài học leakage sentiment).
- Text: builder chỉ TỰ SỬA NFC + lowercase + gộp whitespace; vi phạm khác
  (chữ số, charset lạ, quá ngắn) -> DROP + đếm lý do. Drop rate > ngưỡng
  build.drop_rate_max_pct -> DỪNG, điều tra nguồn.
- Audio: resample 16k mono PCM16; ngoài [dur_min_s, dur_max_s] -> drop.
- Anti-join: text trùng eval sets của product -> drop (đếm riêng).
- Kết thúc: gộp parts -> data/manifest_train_v1.jsonl + manifest_trainval_v1.jsonl
  + sha256 + stats; verify bằng scripts/verify_asr_dataset.py trên sample.

Chạy:  venv/bin/python scripts/build_asr_train_mix.py [--sources vlsp100h,fosd,infore1]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import string
import sys
import unicodedata
from collections import Counter

import numpy as np
import soundfile as sf
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scripts.verify_asr_dataset import VN_CHARS, PUNCT, load_eval_refs, norm_text  # noqa: E402
from voicebench.metrics.text_norm import fold_spelling_variants  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "train_mix_v1")
PARTS_DIR = os.path.join(OUT_DIR, "parts")
RULES = yaml.safe_load(open(os.path.join(ROOT, "configs/data_rules.asr.yaml"),
                            encoding="utf-8"))

SOURCES = {
    "vlsp100h": {"repo": "doof-ferb/vlsp2020_vinai_100h", "license": "CC-BY-4.0"},
    "fosd": {"repo": "doof-ferb/fpt_fosd", "license": "CC-BY-4.0"},
    "infore1": {"repo": "doof-ferb/infore1_25hours", "license": "CC-BY-4.0"},
}
VAL_PCT = 1  # sha1(id) % 100 < VAL_PCT -> val


def clean_text(t: str) -> str:
    """Chuẩn hoá target train — chỉ các phép AN TOÀN, khớp normalize_vi của eval
    và output serving (lowercase, không dấu câu):
      NFC + lowercase + STRIP dấu câu + canonical biến thể dấu thanh (hóa->hoá,
      lý->lí — nhất quán target giữa các nguồn) + gộp whitespace.
    KHÔNG đụng chữ số (verbalize/drop ở row_ok vì '5' vs 'năm' làm lệch audio).
    Strip dấu câu là normalization (giống eval), KHÔNG phải drop row — FPT-FOSD
    có shard giữ nguyên dấu câu, strip là đúng chứ không phải loại bỏ audio tốt."""
    t = unicodedata.normalize("NFC", str(t)).lower()
    t = "".join(" " if c in PUNCT else c for c in t)
    t = fold_spelling_variants(t)
    return " ".join(t.split())


def row_ok(text: str, dur: float, drop: Counter) -> bool:
    r_txt, r_aud = RULES["text"], RULES["audio"]
    if not text:
        drop["text_rong"] += 1
        return False
    if any(c.isdigit() for c in text):
        drop["chu_so"] += 1
        return False
    if any(c in string.punctuation or c in "…“”‘’–—" for c in text):
        drop["dau_cau"] += 1
        return False
    if any(c not in VN_CHARS for c in text):
        drop["charset_la"] += 1
        return False
    if len(text.split()) < r_txt["min_words"]:
        drop["qua_ngan_text"] += 1
        return False
    if not (r_aud["dur_min_s"] <= dur <= r_aud["dur_max_s"]):
        drop["duration"] += 1
        return False
    words_per_s = len(text.split()) / dur
    if not (r_aud["wps_min"] <= words_per_s <= r_aud["wps_max"]):
        drop["lech_audio_text"] += 1
        return False
    return True


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g)
    return x.astype(np.float32)


def process_shard(fs, tag: str, src: dict, shard_path: str, shard_idx: int,
                  evals: set) -> dict:
    """Xử lý 1 shard parquet -> ghi wav + part jsonl. Trả stats."""
    import pyarrow.parquet as pq
    part_path = os.path.join(PARTS_DIR, f"{tag}-{shard_idx:05d}.jsonl")
    if os.path.exists(part_path + ".done"):
        return {"skipped": True}
    clips_dir = os.path.join(OUT_DIR, "clips", tag)
    os.makedirs(clips_dir, exist_ok=True)
    drop = Counter()
    rows_out = []
    n_raw = 0
    with fs.open(shard_path, "rb") as f:
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            for j, r in enumerate(pf.read_row_group(g).to_pylist()):
                n_raw += 1
                rid = f"{tag}-{shard_idx:05d}-{g:04d}-{j:04d}"
                split = "val" if int(hashlib.sha1(rid.encode()).hexdigest(), 16) % 100 < VAL_PCT else "train"
                audio_field = r["audio"] if isinstance(r.get("audio"), dict) else {"bytes": r["bytes"]}
                try:
                    x, sr = sf.read(io.BytesIO(audio_field["bytes"]))
                except Exception:
                    drop["audio_hong"] += 1
                    continue
                x = to_16k_mono(x, sr)
                dur = len(x) / 16000
                text = clean_text(r["transcription"])
                if not row_ok(text, dur, drop):
                    continue
                if len(x) and float(np.sqrt((x ** 2).mean())) < RULES["audio"]["rms_min"]:
                    drop["gan_cam"] += 1
                    continue
                if norm_text(text) in evals:
                    drop["nhiem_eval"] += 1
                    continue
                wav = os.path.join(clips_dir, f"{rid}.wav")
                sf.write(wav, x, 16000, subtype="PCM_16")
                rows_out.append({
                    "id": rid, "audio": os.path.relpath(wav, ROOT), "text": text,
                    "dur_s": round(dur, 2), "split": split,
                    "source": src["repo"], "license": src["license"],
                    # sha1 audio để dedup TOÀN CỤC lúc gộp (sống sót qua resume);
                    # field này bị bỏ khỏi manifest cuối
                    "ah": hashlib.sha1(x.tobytes()).hexdigest(),
                })
    with open(part_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {"n_raw": n_raw, "n_ok": len(rows_out), "drop": dict(drop)}
    with open(part_path + ".done", "w") as f:
        json.dump(stats, f)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="vlsp100h,fosd,infore1")
    args = ap.parse_args()
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    os.makedirs(PARTS_DIR, exist_ok=True)
    evals = load_eval_refs()

    for tag in args.sources.split(","):
        src = SOURCES[tag]
        shards = sorted(f for f in fs.ls(f"datasets/{src['repo']}/data", detail=False)
                        if f.endswith(".parquet") and "train" in f.split("/")[-1])
        print(f"== {tag}: {src['repo']} — {len(shards)} shard", flush=True)
        for i, shard in enumerate(shards):
            st = process_shard(fs, tag, src, shard, i, evals)
            if st.get("skipped"):
                print(f"  shard {i}: đã có, bỏ qua", flush=True)
                continue
            print(f"  shard {i}/{len(shards) - 1}: {st['n_ok']}/{st['n_raw']} ok, "
                  f"drop={st['drop']}", flush=True)

    # Tổng hợp stats từ .done (đúng cả khi resume)
    total = Counter()
    for p in sorted(os.listdir(PARTS_DIR)):
        if p.endswith(".done"):
            st = json.load(open(os.path.join(PARTS_DIR, p)))
            total["n_raw"] += st.get("n_raw", 0)
            for k, v in st.get("drop", {}).items():
                total[f"drop:{k}"] += v

    # Gộp parts -> dedup audio toàn cục (giữ lần xuất hiện đầu, thứ tự part ổn định)
    all_rows = []
    seen_audio: set = set()
    for p in sorted(os.listdir(PARTS_DIR)):
        if p.endswith(".jsonl"):
            with open(os.path.join(PARTS_DIR, p), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    ah = r.pop("ah", None)
                    if ah and ah in seen_audio:
                        total["drop:audio_trung"] += 1
                        continue
                    if ah:
                        seen_audio.add(ah)
                    all_rows.append(r)
    train = [r for r in all_rows if r["split"] == "train"]
    val = [r for r in all_rows if r["split"] == "val"]
    for name, rows in [("manifest_train_v1.jsonl", train), ("manifest_trainval_v1.jsonl", val)]:
        with open(os.path.join(ROOT, "data", name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_raw = max(1, total["n_raw"])
    n_drop = sum(v for k, v in total.items() if k.startswith("drop:"))
    drop_rate = 100.0 * n_drop / n_raw
    durs = np.array([r["dur_s"] for r in all_rows]) if all_rows else np.array([0.0])
    long_pct = 100.0 * float((durs >= RULES["audio"]["longform_min_s"]).sum()) / max(1, len(all_rows))
    hours = float(durs.sum()) / 3600
    stats = {
        "n_raw": total["n_raw"], "n_train": len(train), "n_val": len(val),
        "hours": round(hours, 1), "drop_rate_pct": round(drop_rate, 2),
        "drops": {k[5:]: v for k, v in total.items() if k.startswith("drop:")},
        "longform_pct": round(long_pct, 1),
    }
    with open(os.path.join(OUT_DIR, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)
    if drop_rate > RULES["build"]["drop_rate_max_pct"]:
        print(f"!! DROP RATE {drop_rate:.2f}% > {RULES['build']['drop_rate_max_pct']}% "
              "— DỪNG, điều tra nguồn trước khi dùng mix này.", flush=True)
        sys.exit(1)
    print("Tiếp theo: verify sample mix bằng scripts/verify_asr_dataset.py rồi "
          "đóng băng sha256.", flush=True)


if __name__ == "__main__":
    main()
