"""Dựng bộ eval OOD2 (spot-check) — 1 corpus KHÁC hẳn train (vlsp/fosd/infore1)
và khác cả VIVOS/VietMed, để kiểm tra generalization của model STT.

Ưu tiên Bud500 (audiobook/đọc-kể) qua `datasets` streaming; tự detect cột
audio/text qua features; xử lý torchcodec AudioDecoder (datasets 5.0). Lưu N
clip 16k mono + manifest. Clip KHÔNG version (gitignore data/eval_ood2/clips/);
manifest CÓ commit → dựng lại bằng chính script này.

Dùng cho re-validate ST-2 (runs/st2_ood2_bench.json). Chạy:
  PYTHONPATH=. venv/bin/python scripts/build_eval_ood2.py
"""
import os, sys, json
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from math import gcd

N = 40
OUT_DIR = "data/eval_ood2"
CLIPS = f"{OUT_DIR}/clips"
os.makedirs(CLIPS, exist_ok=True)

CANDIDATES = [
    ("linhtran92/viet_bud500", None, "test"),
    ("NhutP/VietSpeech", None, "test"),
    ("mozilla-foundation/common_voice_17_0", "vi", "test"),
]

TEXT_NAMES = ("transcription", "sentence", "text", "transcript", "raw_transcription")


def to_array_sr(a):
    """Trả (np.float32 mono, sr) từ dict cũ hoặc torchcodec AudioDecoder."""
    if isinstance(a, dict) and "array" in a:
        arr = np.asarray(a["array"], dtype=np.float32); sr = a["sampling_rate"]
        if arr.ndim > 1: arr = arr.mean(axis=1)
        return arr, sr
    if hasattr(a, "get_all_samples"):           # torchcodec AudioDecoder
        s = a.get_all_samples()
        arr = s.data.numpy()                    # (channels, samples)
        if arr.ndim > 1: arr = arr.mean(axis=0)
        return arr.astype(np.float32), int(s.sample_rate)
    raise ValueError(f"audio type lạ: {type(a)}")


from datasets import load_dataset
from datasets.features import Audio, Value

picked = None
for name, cfg, split in CANDIDATES:
    try:
        print(f"thử {name} [{cfg}/{split}] ...", flush=True)
        ds = load_dataset(name, cfg, split=split, streaming=True)
        feats = ds.features
        acol = next((k for k, v in feats.items() if isinstance(v, Audio)), None)
        tcands = [k for k, v in feats.items() if isinstance(v, Value) and v.dtype == "string"]
        tcol = next((k for k in tcands if k.lower() in TEXT_NAMES), tcands[0] if tcands else None)
        if not acol or not tcol:
            print(f"  bỏ: thiếu audio/text (cols={list(feats.keys())})", flush=True); continue
        print(f"  OK: audio='{acol}' text='{tcol}'", flush=True)
        picked = (name, acol, tcol, iter(ds)); break
    except Exception as e:
        print(f"  lỗi: {str(e)[:140]}", flush=True); continue

if not picked:
    print("KHÔNG bộ nào chạy được"); sys.exit(1)

name, acol, tcol, it = picked
man = open(f"{OUT_DIR}/ood2.manifest.jsonl", "w", encoding="utf-8")
saved = 0
for i, r in enumerate(it):
    if saved >= N: break
    txt = (r.get(tcol) or "").strip()
    if not txt: continue
    try:
        arr, sr = to_array_sr(r[acol])
    except Exception as e:
        print(f"  skip {i}: {e}", flush=True); continue
    if sr != 16000:
        g = gcd(int(sr), 16000)
        arr = resample_poly(arr, 16000 // g, int(sr) // g).astype(np.float32)
    wav = f"{CLIPS}/ood2_{saved:04d}.wav"
    sf.write(wav, arr, 16000)
    man.write(json.dumps({"audio": f"{OUT_DIR}/clips/ood2_{saved:04d}.wav", "text": txt}, ensure_ascii=False) + "\n")
    saved += 1
man.close()
print(f"DONE: {name} -> {saved} clip 16k @ {OUT_DIR}/ood2.manifest.jsonl", flush=True)
