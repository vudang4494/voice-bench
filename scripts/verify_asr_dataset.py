"""Verify dataset ASR train theo bộ rules chuẩn (configs/data_rules.asr.yaml).

Nguyên tắc (như mọi gate trong repo): FAIL = điều tra data/nguồn, KHÔNG nới
ngưỡng trong file rules. Chạy TRƯỚC mọi lần fine-tune; dataset chưa PASS thì
không đưa vào train mix.

Input: manifest json (list) hoặc jsonl, mỗi row {id, audio, text}. Với dataset
khổng lồ, verify chạy trên SAMPLE đại diện (>= sample_min_n, lấy trải đều
nhiều shard) — các check text/audio là full-scan trên sample; QC-by-ASR chạy
trên sample đó luôn (full 1M clip không khả thi).

Chạy:
  venv/bin/python scripts/verify_asr_dataset.py --manifest <sample.json> \
      [--rules configs/data_rules.asr.yaml] [--qc-engine chunkformer] [--report out.md]
Exit 0 = PASS đủ mọi gate; 1 = có FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
import unicodedata
from collections import Counter

import numpy as np
import soundfile as sf
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:  # chạy theo path `python scripts/...` vẫn import được voicebench
    sys.path.insert(0, ROOT)

# Charset tiếng Việt chuẩn (lowercase) + loanword f/j/w/z; space là ký tự hợp lệ
VN_CHARS = set("aăâbcdđeêghiklmnoôơpqrstuưvxyfjwz") | set(
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ") | {" "}
PUNCT = set(string.punctuation) | set("…“”‘’–—")
WEIRD_WS = ["\t", " ", "​", "‌", "\r", "\n"]

# Các manifest eval của product — text train KHÔNG được trùng (anti-join)
EVAL_MANIFESTS = [
    "data/eval/eval_3min.manifest.jsonl",
    "data/eval_ood/ood.manifest.jsonl",
    "data/manifest_v1.jsonl",
    "data/manifest_dev.jsonl",
]


def norm_text(t: str) -> str:
    """Chuẩn hoá để so trùng: NFC + lowercase + bỏ punct + gộp space."""
    t = unicodedata.normalize("NFC", str(t)).lower()
    t = "".join(c for c in t if c not in PUNCT)
    return " ".join(t.split())


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            rows = json.load(f)
        else:
            rows = [json.loads(l) for l in f if l.strip()]
    # chấp nhận text_raw (sampler) hoặc text (manifest chuẩn)
    for r in rows:
        if "text" not in r and "text_raw" in r:
            r["text"] = r["text_raw"]
    return rows


def load_eval_refs() -> set[str]:
    refs = set()
    for rel in EVAL_MANIFESTS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                refs.add(norm_text(json.loads(line)["text"]))
    return refs


class Verifier:
    def __init__(self, rules: dict):
        self.rules = rules
        self.results: list[tuple[str, bool, str]] = []

    def gate(self, rule_id: str, ok: bool, detail: str):
        self.results.append((rule_id, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {rule_id}: {detail}")

    # ---------- TEXT ----------
    def check_text(self, rows):
        r = self.rules["text"]
        n = len(rows)
        empty = upper = punct = digit = nfc_bad = ws = short = charset_bad = 0
        bad_chars = Counter()
        for row in rows:
            t = str(row["text"])
            if not t.strip():
                empty += 1
                continue
            if any(unicodedata.category(c) == "Lu" for c in t):
                upper += 1
            if any(c in PUNCT for c in t):
                punct += 1
            if re.search(r"[0-9]", t):
                digit += 1
            if t != unicodedata.normalize("NFC", t):
                nfc_bad += 1
            if any(w in t for w in WEIRD_WS):
                ws += 1
            low = unicodedata.normalize("NFC", t).lower()
            bad = {c for c in low if c not in VN_CHARS and not c.isdigit() and c not in PUNCT}
            if bad:
                charset_bad += 1
                bad_chars.update(bad)
            if len(low.split()) < r["min_words"]:
                short += 1

        def p(x):
            return 100.0 * x / n

        self.gate("TXT-empty", p(empty) <= r["empty_rows_max_pct"],
                  f"{p(empty):.2f}% ref rỗng (<= {r['empty_rows_max_pct']}%)")
        self.gate("TXT-uppercase", p(upper) <= r["uppercase_rows_max_pct"],
                  f"{p(upper):.2f}% row có chữ HOA (<= {r['uppercase_rows_max_pct']}%)")
        self.gate("TXT-punct", p(punct) <= r["punct_rows_max_pct"],
                  f"{p(punct):.2f}% row có dấu câu (<= {r['punct_rows_max_pct']}%)")
        self.gate("TXT-digit", p(digit) <= r["digit_rows_max_pct"],
                  f"{p(digit):.2f}% row có chữ số (<= {r['digit_rows_max_pct']}%)")
        self.gate("TXT-nfc", p(nfc_bad) <= r["nfc_bad_rows_max_pct"],
                  f"{p(nfc_bad):.2f}% row không NFC (<= {r['nfc_bad_rows_max_pct']}%)")
        self.gate("TXT-whitespace", p(ws) <= r["weird_ws_rows_max_pct"],
                  f"{p(ws):.2f}% row có whitespace lạ (<= {r['weird_ws_rows_max_pct']}%)")
        self.gate("TXT-charset", p(charset_bad) <= r["charset_bad_rows_max_pct"],
                  f"{p(charset_bad):.2f}% row có ký tự ngoài charset VN"
                  f" (<= {r['charset_bad_rows_max_pct']}%) {dict(bad_chars.most_common(5))}")
        self.gate("TXT-minwords", p(short) <= r["under_min_words_max_pct"],
                  f"{p(short):.2f}% row < {r['min_words']} từ (<= {r['under_min_words_max_pct']}%)")

    # ---------- AUDIO ----------
    def check_audio(self, rows):
        r = self.rules["audio"]
        n = len(rows)
        sr_bad = dur_lo = dur_hi = quiet = clipped = wps_bad = 0
        durs = []
        for row in rows:
            x, sr = sf.read(row["audio"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            dur = len(x) / sr
            durs.append(dur)
            if sr != r["sample_rate"]:
                sr_bad += 1
            if dur < r["dur_min_s"]:
                dur_lo += 1
            if dur > r["dur_max_s"]:
                dur_hi += 1
            if len(x) and float(np.sqrt((x ** 2).mean())) < r["rms_min"]:
                quiet += 1
            if len(x) and float((np.abs(x) >= 0.999).mean()) > r["clip_ratio_row_max"]:
                clipped += 1
            words = len(norm_text(row["text"]).split())
            wps = words / dur if dur > 0 else 0
            if not (r["wps_min"] <= wps <= r["wps_max"]):
                wps_bad += 1

        def p(x):
            return 100.0 * x / n

        self.gate("AUD-samplerate", p(sr_bad) <= 0.0,
                  f"{p(sr_bad):.2f}% row sai sample rate {r['sample_rate']} (phải 0%)")
        self.gate("AUD-dur", p(dur_lo + dur_hi) <= r["dur_outlier_max_pct"],
                  f"{p(dur_lo + dur_hi):.2f}% row ngoài [{r['dur_min_s']}, {r['dur_max_s']}]s"
                  f" (<= {r['dur_outlier_max_pct']}%)")
        self.gate("AUD-rms", p(quiet) <= r["quiet_rows_max_pct"],
                  f"{p(quiet):.2f}% row RMS < {r['rms_min']} (<= {r['quiet_rows_max_pct']}%)")
        self.gate("AUD-clip", p(clipped) <= r["clipped_rows_max_pct"],
                  f"{p(clipped):.2f}% row clipping > {r['clip_ratio_row_max']:.0%}"
                  f" mẫu (<= {r['clipped_rows_max_pct']}%)")
        self.gate("AUD-wps", p(wps_bad) <= r["wps_outlier_max_pct"],
                  f"{p(wps_bad):.2f}% row tốc độ từ ngoài [{r['wps_min']}, {r['wps_max']}] w/s"
                  f" (<= {r['wps_outlier_max_pct']}%) — lệch audio-text")
        # Phân bố duration (cảnh báo mix-level, không fail dataset):
        durs = np.array(durs)
        long_pct = 100.0 * float((durs >= r["longform_min_s"]).sum()) / n
        print(f"  [INFO] AUD-durdist: {long_pct:.1f}% clip >= {r['longform_min_s']}s "
              f"(rule mix-level: train mix cần >= {r['mix_longform_min_pct']}% — cân khi trộn)")

    # ---------- INTEGRITY ----------
    def check_integrity(self, rows):
        r = self.rules["integrity"]
        n = len(rows)
        texts = [norm_text(row["text"]) for row in rows]
        ahash = [hashlib.sha1(open(row["audio"], "rb").read()).hexdigest()
                 for row in rows]
        audio_dup = sum(c - 1 for c in Counter(ahash).values() if c > 1)
        # "Trùng nguy hiểm" = trùng CẢ text LẪN audio (caption spam / nhân bản 1
        # clip). Trùng text nhưng KHÁC audio (bản tin đọc lại intro/outro; nhiều
        # người đọc cùng bài) = ĐA DẠNG ÂM HỌC, tốt cho ASR — không phạt, chỉ giữ
        # backstop chống loop/synthetic. (Đo thật mix v1: text lặp 20 lần là news
        # boilerplate VLSP, 20 audio đều KHÁC nhau -> không phải spam.)
        ta = Counter(zip(texts, ahash))
        spam_extra = sum(c - 1 for c in ta.values() if c > 1)
        spam_rep_max = max(ta.values()) if ta else 0
        text_cnt = Counter(texts)
        variety_rep_max = max(text_cnt.values()) if text_cnt else 0
        variety_dup_pct = 100.0 * sum(c - 1 for c in text_cnt.values() if c > 1) / n
        cap = r.get("distinct_audio_text_repeat_max", 100)
        evals = load_eval_refs()
        contam = sum(1 for t in texts if t in evals)

        self.gate("INT-dup-text", 100.0 * spam_extra / n <= r["dup_text_max_pct"],
                  f"{100.0 * spam_extra / n:.2f}% row trùng CẢ text+audio (spam)"
                  f" (<= {r['dup_text_max_pct']}%)")
        self.gate("INT-rep-text", spam_rep_max <= r["single_text_repeat_max"],
                  f"1 cặp (text,audio) lặp tối đa {spam_rep_max} lần"
                  f" (<= {r['single_text_repeat_max']}) — bắt caption spam/nhân bản")
        self.gate("INT-dup-audio", audio_dup == 0,
                  f"{audio_dup} audio trùng hash (phải 0)")
        self.gate("INT-text-variety", variety_rep_max <= cap,
                  f"[đa dạng âm học] text lặp nhiều nhất {variety_rep_max} lần /"
                  f" {variety_dup_pct:.1f}% row trùng text (khác audio) — backstop <= {cap}")
        self.gate("INT-eval-contam", contam == 0,
                  f"{contam} row trùng text với eval sets của product (phải 0 —"
                  f" so {len(evals)} refs: VIVOS/VietMed/manifest_v1/dev)")

    # ---------- QC-BY-ASR (per-dataset, sample) ----------
    def check_qc_asr(self, rows, engine_name: str):
        r = self.rules["qc_asr"]
        if len(rows) < r["sample_min_n"]:
            self.gate("DST-qc-n", False,
                      f"sample {len(rows)} < {r['sample_min_n']} — chưa đủ để tin QC")
            return
        from voicebench.engines.registry import build_asr
        from voicebench.metrics.wer import corpus_wer
        kwargs = r.get("engines", {}).get(engine_name, {})
        asr = build_asr(engine_name, kwargs)
        refs, hyps, row_wers = [], [], []
        for row in rows:
            x, sr = sf.read(row["audio"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            hyp = asr.transcribe(x, sr).text
            refs.append(row["text"])
            hyps.append(hyp)
            row_wers.append(corpus_wer([row["text"]], [hyp])["wer"])
        agg = corpus_wer(refs, hyps)["wer"]
        bad = 100.0 * sum(1 for w in row_wers if w > 0.5) / len(row_wers)
        self.gate("DST-qc-wer", agg <= r["corpus_wer_max"],
                  f"corpus WER(ref vs {engine_name}) = {agg:.2%} (<= {r['corpus_wer_max']:.0%})")
        self.gate("DST-qc-badrows", bad <= r["rows_over_50pct_max_pct"],
                  f"{bad:.2f}% row vênh > 50% (<= {r['rows_over_50pct_max_pct']}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rules", default=os.path.join(ROOT, "configs/data_rules.asr.yaml"))
    ap.add_argument("--qc-engine", default=None,
                    help="chunkformer | faster-whisper | (bỏ trống = skip QC-by-ASR)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    rules = yaml.safe_load(open(args.rules, encoding="utf-8"))
    rows = load_manifest(args.manifest)
    print(f"Verify dataset: {args.manifest} ({len(rows)} rows) | rules: {args.rules}")

    v = Verifier(rules)
    print("== TEXT ==")
    v.check_text(rows)
    print("== AUDIO ==")
    v.check_audio(rows)
    print("== INTEGRITY ==")
    v.check_integrity(rows)
    if args.qc_engine:
        print(f"== QC-BY-ASR ({args.qc_engine}) ==")
        v.check_qc_asr(rows, args.qc_engine)

    n_pass = sum(1 for _, ok, _ in v.results if ok)
    ok_all = n_pass == len(v.results)
    print(f"\n==> {'PASS' if ok_all else 'FAIL'} ({n_pass}/{len(v.results)} gates)")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(f"# Verify ASR dataset — {'PASS' if ok_all else 'FAIL'}\n\n"
                    f"- Manifest: {args.manifest} ({len(rows)} rows)\n\n"
                    "| Gate | Kết quả | Chi tiết |\n|---|---|---|\n")
            for rid, ok, detail in v.results:
                f.write(f"| {rid} | {'✅' if ok else '❌'} | {detail} |\n")
        print(f"Report: {args.report}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
