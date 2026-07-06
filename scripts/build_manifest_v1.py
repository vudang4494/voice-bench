"""Build data/manifest_v1.jsonl — bộ eval tiếng Việt CHUẨN, ĐÓNG BĂNG (T5).

Thiết kế theo ROADMAP T5:
- Nguồn public license rõ, ghi source+license TỪNG SAMPLE trong manifest:
  * VIVOS test (mirror parquet `quocanh34/viet_vivos`, gốc AILAB VNU-HCM,
    CC BY-NC-SA 4.0) — clip ngắn/trung (đọc sách, 686 rows).
  * FLEURS vi_vn validation (`google/fleurs`, CC-BY-4.0) — clip trung/dài
    (câu Wikipedia có chữ số + tên riêng; 361 rows, biết duration trước khi
    tải qua num_samples; split test 691MB vượt scan limit của datasets-server).
- Phân tầng: ~1/3 ngắn (<3s), ~1/3 trung (3-10s), ~1/3 dài (10-30s);
  ưu tiên câu chứa chữ số (mục tiêu >= 30) và tên riêng/loanword (>= 20).
- Text: NFC giữ nguyên dấu; FLEURS dùng raw_transcription (verbatim, giữ chữ
  số — để T9 đo gap '5' vs 'năm'); VIVOS dùng transcription gốc.
- Audio: 16kHz mono WAV (convention voicebench/audio.py), path tương đối repo.
- Loại các row VIVOS đã nằm trong data/eval/ (gate set) — hai bộ phải rời nhau.
- QC máy: ChunkFormer local đọc lại từng clip; row KHÔNG chứa chữ số mà
  WER >= 0.6 coi là ref/audio hỏng -> loại (row có chữ số chỉ flag, vì ASR
  xuất chữ còn ref là số — gap normalization, không phải ref sai).
- Dev slice 30 câu (data/manifest_dev.jsonl, 10/bucket) TÁCH RIÊNG khỏi v1
  để iterate nhanh không nhiễm bộ chính; dev sort ngắn->dài (smoke lấy đầu).
- Đóng băng: data/manifest_v1.sha256 (hash manifest + từng clip).
  QUY TẮC: v1 không bao giờ sửa — bổ sung thì làm manifest_v2.

Chạy (từ voice-bench/, cần mạng + models/chunkformer-large-vie cho QC):
    venv/bin/python scripts/build_manifest_v1.py
    venv/bin/python scripts/build_manifest_v1.py --skip-qc   # không có model QC
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicebench.audio import resample  # noqa: E402

API = "https://datasets-server.huggingface.co/rows"
SR = 16000

VIVOS = {"dataset": "quocanh34/viet_vivos", "config": "default", "split": "test",
         "total": 686, "source": "AILAB-VNUHCM/vivos (mirror quocanh34/viet_vivos, test)",
         "license": "CC BY-NC-SA 4.0"}
FLEURS = {"dataset": "google/fleurs", "config": "vi_vn", "split": "validation",
          "total": 361, "source": "google/fleurs vi_vn (validation)",
          "license": "CC-BY-4.0"}

# Quota mỗi bucket = v1 + dev (+ dư để bù row bị QC loại).
BUCKETS = {"short": (0.0, 3.0), "medium": (3.0, 10.0), "long": (10.0, 30.0)}
V1_TARGET = {"short": 135, "medium": 135, "long": 130}
DEV_TARGET = {"short": 10, "medium": 10, "long": 10}
OVERPROVISION = 12  # mỗi bucket lấy dư chừng này để bù QC loại


def fetch_rows(ds: dict, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": ds["dataset"], "config": ds["config"],
                                "split": ds["split"], "offset": offset,
                                "length": length})
    with urllib.request.urlopen(f"{API}?{q}", timeout=90) as r:
        return json.load(r)["rows"]


def dl_audio(src: str) -> tuple[np.ndarray, int]:
    with urllib.request.urlopen(src, timeout=120) as r:
        wav, sr = sf.read(io.BytesIO(r.read()), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def dl_audio_retry(ds: dict, offset: int, src: str) -> tuple[np.ndarray, int]:
    """Asset URL của datasets-server có hạn dùng — lấy metadata trước rồi tải
    sau là dính hết hạn hàng loạt (lần build đầu mất 22 row FLEURS kiểu này).
    Fail thì fetch lại đúng row đó để lấy URL mới, thử thêm 1 lần."""
    try:
        return dl_audio(src)
    except Exception:  # noqa: BLE001
        row = fetch_rows(ds, offset, 1)[0]["row"]
        return dl_audio(row["audio"][0]["src"])


def load_cached(clips_dir: Path, cid: str) -> tuple[np.ndarray, int] | None:
    """Clip đã có trên đĩa từ lần build trước (id deterministic theo offset)
    -> đọc local, khỏi tải lại."""
    p = clips_dir / f"{cid}.wav"
    if p.exists():
        wav, sr = sf.read(p, dtype="float32", always_2d=False)
        return wav.astype(np.float32), int(sr)
    return None


def bucket_of(dur: float) -> str | None:
    for name, (lo, hi) in BUCKETS.items():
        if lo <= dur < hi:
            return name
    return None


def norm_text(t: str) -> str:
    return unicodedata.normalize("NFC", t.strip())


def tags_of(text: str) -> list[str]:
    tags = []
    if re.search(r"\d", text):
        tags.append("digits")
    # Tên riêng: từ viết hoa không đứng đầu câu (FLEURS raw giữ hoa/thường).
    if re.search(r"(?<![.!?]\s)(?<!^)\b[A-ZĐÂĂÊÔƠƯ][a-zà-ỹđ]+", text):
        tags.append("proper_name")
    # Loanword: từ chứa f/j/w/z (không có trong chính tả tiếng Việt).
    if re.search(r"\b\w*[fjwzFJWZ]\w*\b", text):
        tags.append("loanword")
    return tags


def used_gate_offsets() -> set[int]:
    """Row offset VIVOS đã dùng trong data/eval (clip vivos-test-rNNNN.wav)."""
    used = set()
    for p in (ROOT / "data/eval/clips").glob("vivos-test-r*.wav"):
        m = re.search(r"r(\d+)\.wav$", p.name)
        if m:
            used.add(int(m.group(1)))
    return used


def quota(counts: dict, name: str) -> int:
    return V1_TARGET[name] + DEV_TARGET[name] + OVERPROVISION - counts.get(name, 0)


def collect_vivos(clips_dir: Path, counts: dict) -> list[dict]:
    """Duyệt tuần tự 686 rows (thứ tự offset — split xếp theo speaker nên đi
    hết split là phủ đủ speaker); dur chỉ biết sau khi tải nên tải rồi mới xếp
    bucket, bucket đầy thì bỏ (không ghi file)."""
    skip = used_gate_offsets()
    print(f"VIVOS: loại {len(skip)} row đã nằm trong gate set data/eval/")
    out = []
    for page in range(0, VIVOS["total"], 100):
        if quota(counts, "short") <= 0 and quota(counts, "medium") <= 0:
            break
        try:
            rows = fetch_rows(VIVOS, page, min(100, VIVOS["total"] - page))
        except Exception as e:  # noqa: BLE001 — log-và-bỏ-qua như run loop chính
            print(f"  bỏ page {page}: {e}")
            continue
        for j, r in enumerate(rows):
            off = page + j
            if off in skip:
                continue
            text = norm_text(r["row"].get("transcription") or "")
            if not text:
                continue
            cid = f"vivos-r{off:04d}"
            cached = load_cached(clips_dir, cid)
            try:
                wav, sr = cached or dl_audio_retry(VIVOS, off,
                                                   r["row"]["audio"][0]["src"])
            except Exception as e:  # noqa: BLE001
                print(f"  bỏ row {off} (dl): {e}")
                continue
            if sr != SR:
                wav = resample(wav, sr, SR)
            dur = len(wav) / SR
            b = bucket_of(dur)
            if b is None or b == "long" or quota(counts, b) <= 0:
                continue
            sf.write(clips_dir / f"{cid}.wav", wav, SR, subtype="PCM_16")
            counts[b] = counts.get(b, 0) + 1
            out.append({"id": cid, "audio": f"data/manifest_v1/clips/{cid}.wav",
                        "text": text, "source": VIVOS["source"],
                        "license": VIVOS["license"],
                        "duration_s": round(dur, 2), "bucket": b,
                        "tags": tags_of(text)})
        print(f"  page {page}: short {counts.get('short', 0)} "
              f"medium {counts.get('medium', 0)}")
    return out


def collect_fleurs(clips_dir: Path, counts: dict) -> list[dict]:
    """Metadata trước (num_samples -> dur), chọn xong mới tải audio (không tải
    thừa). Ưu tiên row chứa chữ số cho đủ quota digits, còn lại theo offset."""
    metas = []
    for page in range(0, FLEURS["total"], 100):
        try:
            rows = fetch_rows(FLEURS, page, min(100, FLEURS["total"] - page))
        except Exception as e:  # noqa: BLE001
            print(f"  bỏ page {page}: {e}")
            continue
        for j, r in enumerate(rows):
            row = r["row"]
            text = norm_text(row.get("raw_transcription")
                             or row.get("transcription") or "")
            dur = (row.get("num_samples") or 0) / 16000
            b = bucket_of(dur)
            if not text or b is None or b == "short":
                continue
            metas.append({"off": page + j, "text": text, "dur": dur, "bucket": b,
                          "src": row["audio"][0]["src"],
                          "gender": row.get("gender")})
    # Deterministic: trong mỗi bucket, row có chữ số đứng trước, rồi theo offset.
    # Bucket medium thường đã đầy từ VIVOS (không có chữ số — text verbalize
    # sẵn) -> vẫn LUÔN vớt row medium chứa chữ số của FLEURS (hiếm, ~4 row)
    # để đạt quota digits; split_dev giữ digit rows khi trim.
    chosen = []
    for b in ("long", "medium"):
        cand = [m for m in metas if m["bucket"] == b]
        cand.sort(key=lambda m: (0 if re.search(r"\d", m["text"]) else 1, m["off"]))
        take = cand[: max(0, quota(counts, b))]
        if b == "medium":
            extra = [m for m in cand if m not in take
                     and re.search(r"\d", m["text"])][:8]
            take += extra
        chosen += take
    print(f"FLEURS: chọn {len(chosen)} rows "
          f"(long {sum(1 for m in chosen if m['bucket'] == 'long')}, "
          f"medium {sum(1 for m in chosen if m['bucket'] == 'medium')}) — tải audio...")
    out = []
    for i, m in enumerate(chosen):
        cid = f"fleurs-r{m['off']:04d}"
        cached = load_cached(clips_dir, cid)
        try:
            wav, sr = cached or dl_audio_retry(FLEURS, m["off"], m["src"])
        except Exception as e:  # noqa: BLE001
            print(f"  bỏ fleurs {m['off']} (dl): {e}")
            continue
        if sr != SR:
            wav = resample(wav, sr, SR)
        sf.write(clips_dir / f"{cid}.wav", wav, SR, subtype="PCM_16")
        counts[m["bucket"]] = counts.get(m["bucket"], 0) + 1
        out.append({"id": cid, "audio": f"data/manifest_v1/clips/{cid}.wav",
                    "text": m["text"], "source": FLEURS["source"],
                    "license": FLEURS["license"],
                    "duration_s": round(len(wav) / SR, 2), "bucket": m["bucket"],
                    "tags": tags_of(m["text"]),
                    "gender": m["gender"]})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(chosen)}")
    return out


def qc_asr(entries: list[dict]) -> dict:
    """Cross-check máy: ChunkFormer local đọc lại từng clip, WER per-utt so ref.
    Trả {id: {wer, hyp}}. (VIVOS/FLEURS có thể nằm trong train data của model
    QC — ở đây chỉ dùng để bắt ref/audio LỆCH NHAU, không phải đo model.)"""
    from voicebench.engines.registry import build_asr
    from voicebench.metrics.wer import corpus_wer
    # Cache theo id từ lần build trước (clip deterministic theo offset).
    old_path = ROOT / "data/manifest_v1/qc_asr.json"
    res = (json.loads(old_path.read_text(encoding="utf-8"))
           if old_path.exists() else {})
    todo = [e for e in entries if e["id"] not in res]
    print(f"QC: {len(res)} id đã có cache, chạy mới {len(todo)} — "
          "load ChunkFormer local...")
    if not todo:
        return res
    asr = build_asr("chunkformer", {"model_id": "models/chunkformer-large-vie",
                                    "device": "cpu"})
    for i, e in enumerate(todo):
        wav, sr = sf.read(ROOT / e["audio"], dtype="float32")
        r = asr.transcribe(wav, sr)
        w = corpus_wer([e["text"]], [r.text])
        res[e["id"]] = {"wer": round(w["wer"], 4), "hyp": r.text}
        if (i + 1) % 50 == 0:
            print(f"  QC {i + 1}/{len(todo)}")
    return res


def split_dev(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Tách dev 10/bucket đều khắp danh sách (stride), phần còn lại trim về
    V1_TARGET. Deterministic: entries đã theo thứ tự nguồn/offset."""
    v1, dev = [], []
    for b in BUCKETS:
        rows = [e for e in entries if e["bucket"] == b]
        # Row có chữ số lên đầu (sort ổn định giữ thứ tự cũ trong nhóm) để
        # bước trim về V1_TARGET không bao giờ cắt trúng digit rows (quota >=30).
        rows.sort(key=lambda e: 0 if "digits" in e["tags"] else 1)
        n_dev = min(DEV_TARGET[b], len(rows))
        step = max(1, len(rows) // n_dev) if n_dev else 1
        dev_idx = set(range(0, step * n_dev, step))
        dev += [rows[i] for i in sorted(dev_idx)]
        rest = [rows[i] for i in range(len(rows)) if i not in dev_idx]
        v1 += rest[: V1_TARGET[b]]
    dev.sort(key=lambda e: e["duration_s"])  # smoke limit=N lấy clip ngắn trước
    return v1, dev


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-qc", action="store_true",
                    help="bỏ QC ASR (không có models/chunkformer-large-vie)")
    ap.add_argument("--qc-wer-drop", type=float, default=0.6,
                    help="row không chứa chữ số có QC WER >= ngưỡng này bị loại")
    args = ap.parse_args()

    out_dir = ROOT / "data/manifest_v1"
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    counts: dict = {}
    entries = collect_vivos(clips_dir, counts)
    entries += collect_fleurs(clips_dir, counts)
    print(f"Tổng tải: {len(entries)} clips — {counts}")

    qc = {}
    if not args.skip_qc:
        qc = qc_asr(entries)
        dropped = [e for e in entries
                   if "digits" not in e["tags"]
                   and qc[e["id"]]["wer"] >= args.qc_wer_drop]
        for e in dropped:
            print(f"  QC LOẠI {e['id']} (WER {qc[e['id']]['wer']:.0%}): "
                  f"ref={e['text'][:50]!r} hyp={qc[e['id']]['hyp'][:50]!r}")
            (ROOT / e["audio"]).unlink()
        entries = [e for e in entries if e not in dropped]
        (out_dir / "qc_asr.json").write_text(
            json.dumps(qc, ensure_ascii=False, indent=1), encoding="utf-8")

    v1, dev = split_dev(entries)
    # Dọn clip thừa (over-provision không được chọn) để checksum chỉ phủ file dùng.
    keep = {e["id"] for e in v1 + dev}
    for p in clips_dir.glob("*.wav"):
        if p.stem not in keep:
            p.unlink()

    mpath = ROOT / "data/manifest_v1.jsonl"
    dpath = ROOT / "data/manifest_dev.jsonl"
    mpath.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in v1)
                     + "\n", encoding="utf-8")
    dpath.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in dev)
                     + "\n", encoding="utf-8")

    # Đóng băng: checksum manifest + từng clip (sha256).
    lines = [f"{sha256_file(mpath)}  data/manifest_v1.jsonl",
             f"{sha256_file(dpath)}  data/manifest_dev.jsonl"]
    for p in sorted(clips_dir.glob("*.wav")):
        lines.append(f"{sha256_file(p)}  {p.relative_to(ROOT)}")
    (ROOT / "data/manifest_v1.sha256").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")

    # Listen-check log: 20 id stride đều khắp v1 — máy điền hyp, người nghe
    # xác nhận cột cuối (PENDING cho tới khi user nghe).
    picks = [v1[i] for i in range(0, len(v1), max(1, len(v1) // 20))][:20]
    lc = ["# Listen-check manifest_v1 — 20 mẫu ngẫu-định (stride)",
          "",
          "Máy đã cross-check bằng ChunkFormer local (cột QC). Người nghe mở",
          "clip, so với ref, điền PASS/FAIL cột cuối. Row FAIL -> loại ở v2.",
          "", "| id | dur | ref | QC hyp | QC WER | Người nghe |",
          "|---|---|---|---|---|---|"]
    for e in picks:
        q = qc.get(e["id"], {})
        lc.append(f"| {e['id']} | {e['duration_s']}s | {e['text'][:60]} | "
                  f"{q.get('hyp', '(skip-qc)')[:60]} | {q.get('wer', '—')} | PENDING |")
    (out_dir / "listen_check.md").write_text("\n".join(lc) + "\n", encoding="utf-8")

    # Thống kê nghiệm thu (verify của T5).
    n = len(v1)
    stats = {b: sum(1 for e in v1 if e["bucket"] == b) for b in BUCKETS}
    n_dig = sum(1 for e in v1 if "digits" in e["tags"])
    n_pn = sum(1 for e in v1 if {"proper_name", "loanword"} & set(e["tags"]))
    print(f"\nXONG: v1 = {n} câu {stats} | dev = {len(dev)}")
    print(f"  digits: {n_dig} (cần >=30) | proper_name/loanword: {n_pn} (cần >=20)")
    for b, c in stats.items():
        share = c / n
        flag = "OK" if 0.25 <= share <= 0.40 else "LỆCH!"
        print(f"  bucket {b}: {c} ({share:.0%}) {flag}")
    print(f"  -> {mpath}\n  -> {dpath}\n  -> data/manifest_v1.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
