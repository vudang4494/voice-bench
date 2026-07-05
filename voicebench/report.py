"""Đọc results.jsonl -> aggregate + bootstrap BCa CI -> in bảng markdown.

Metrics báo cáo:
  ASR#1 WER/CER          -> chất lượng ASR (trên audio thật)
  Round-trip WER/CER     -> intelligibility TTS (ASR đọc lại audio TTS)
  ΔWER = RT - ASR        -> phần lỗi QUY CHO TTS (đã trừ nền ASR)
  Latency ASR/TTS        -> median + p90 + RTF
  Speaker sim            -> mean + 95% BCa CI
  MOS (nếu có)           -> mean + 95% BCa CI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics.wer import corpus_wer, corpus_cer
from .metrics.latency import summarize_latency
from .metrics.bootstrap import bca_ci


def _load(results_path: str) -> list[dict]:
    rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict], keep_tone: bool = True) -> dict:
    refs = [r["ref_text"] for r in rows]
    asr = [r["asr_text"] for r in rows]
    rt = [r["roundtrip_text"] for r in rows]

    asr_c = corpus_wer(refs, asr, keep_tone)
    rt_c = corpus_wer(refs, rt, keep_tone)
    # CER cũng corpus-pooled như WER — mean per-sentence bias câu ngắn.
    asr_cer = corpus_cer(refs, asr, keep_tone)["cer"]
    rt_cer = corpus_cer(refs, rt, keep_tone)["cer"]

    asr_lat = summarize_latency([r["asr_latency"]["total_s"] for r in rows])
    asr_rtf = summarize_latency([r["asr_latency"]["rtf"] for r in rows])
    tts_lat = summarize_latency([r["tts_latency"]["total_s"] for r in rows])
    tts_rtf = summarize_latency([r["tts_latency"]["rtf"] for r in rows])
    ttfa_vals = [r["tts_latency"].get("ttfa_s") for r in rows
                 if r["tts_latency"].get("ttfa_s") is not None]
    tts_ttfa = summarize_latency(ttfa_vals) if ttfa_vals else None

    spk = [r["speaker_sim"] for r in rows if r.get("speaker_sim") is not None]
    mos = [r["mos"] for r in rows if r.get("mos") is not None]

    return {
        "n": len(rows),
        "keep_tone": keep_tone,
        "asr_wer": asr_c["wer"], "asr_cer": asr_cer,
        "rt_wer": rt_c["wer"], "rt_cer": rt_cer,
        "delta_wer": rt_c["wer"] - asr_c["wer"],
        "asr_errors": {k: asr_c[k] for k in ("substitutions", "deletions", "insertions")},
        "asr_latency": asr_lat, "asr_rtf": asr_rtf,
        "tts_latency": tts_lat, "tts_rtf": tts_rtf, "tts_ttfa": tts_ttfa,
        "speaker_sim": bca_ci(spk) if spk else None,
        "mos": bca_ci(mos) if mos else None,
    }


def to_markdown(agg: dict) -> str:
    L = []
    L.append(f"# voice-bench report (n={agg['n']})\n")
    L.append("## Accuracy\n")
    L.append("| Metric | WER | CER |")
    L.append("|---|---|---|")
    L.append(f"| ASR#1 (audio thật) | {agg['asr_wer']:.4f} | {agg['asr_cer']:.4f} |")
    L.append(f"| Round-trip (audio TTS) | {agg['rt_wer']:.4f} | {agg['rt_cer']:.4f} |")
    L.append(f"| **ΔWER (quy cho TTS)** | **{agg['delta_wer']:+.4f}** | — |")
    e = agg["asr_errors"]
    L.append(f"\nASR errors: S={e['substitutions']} D={e['deletions']} I={e['insertions']}\n")

    L.append("## Latency\n")
    L.append("| Stage | median (s) | p90 (s) | RTF median |")
    L.append("|---|---|---|---|")
    L.append(f"| ASR | {agg['asr_latency']['median_s']:.3f} | "
             f"{agg['asr_latency']['p90_s']:.3f} | {agg['asr_rtf']['median_s']:.3f} |")
    L.append(f"| TTS | {agg['tts_latency']['median_s']:.3f} | "
             f"{agg['tts_latency']['p90_s']:.3f} | {agg['tts_rtf']['median_s']:.3f} |")
    if agg["tts_ttfa"]:
        L.append(f"| TTS TTFA | {agg['tts_ttfa']['median_s']:.3f} | "
                 f"{agg['tts_ttfa']['p90_s']:.3f} | — |")

    L.append("\n## Fidelity\n")
    if agg["speaker_sim"]:
        s = agg["speaker_sim"]
        L.append(f"- Speaker sim: **{s['point']:.4f}** "
                 f"(95% BCa CI [{s['low']:.4f}, {s['high']:.4f}], n={s['n']})")
        L.append("  - ⚠️ chỉ diễn giải được khi so với same/different-speaker baseline")
    if agg["mos"]:
        m = agg["mos"]
        L.append(f"- MOS (UTMOS): **{m['point']:.3f}** "
                 f"(95% BCa CI [{m['low']:.3f}, {m['high']:.3f}])")
    return "\n".join(L)


def _default_keep_tone(results_path: str) -> bool:
    """Mặc định lấy keep_tone từ run_meta.json cạnh results (config lúc chạy bench)."""
    meta_path = Path(results_path).parent / "run_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return bool(meta["config"].get("keep_tone", True))
    except (OSError, KeyError, ValueError):
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", "-r", required=True)
    ap.add_argument("--keep-tone", action=argparse.BooleanOptionalAction, default=None,
                    help="mặc định: đọc từ run_meta.json (fallback true); "
                         "--no-keep-tone để chấm không phân biệt dấu thanh")
    ap.add_argument("--out", "-o", default=None, help="ghi markdown ra file")
    args = ap.parse_args()
    keep_tone = (args.keep_tone if args.keep_tone is not None
                 else _default_keep_tone(args.results))
    rows = _load(args.results)
    agg = aggregate(rows, keep_tone=keep_tone)
    md = to_markdown(agg)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    Path(args.results).with_suffix(".agg.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
