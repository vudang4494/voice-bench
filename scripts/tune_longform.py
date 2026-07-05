"""T7: tune decode params long-form cho faster-whisper trên Mac Mini M4.

Vấn đề đo được (baseline 2026-07-05): PhoWhisper-small-ct2 decode tuần tự file
3 phút MẤT nội dung ở ranh giới cửa sổ 30s → WER long-form 18.35% dù
per-utterance chỉ 6.61%. Script này chạy ma trận VAD × condition × beam trên
CÙNG audio + ref, chọn cấu hình thắng để hardcode vào configs/service.yaml.

    venv/bin/python scripts/tune_longform.py \
        --model-id diepho/PhoWhisper-small-ct2 \
        --audio data/eval/eval_3min.mp3 --ref data/eval/eval_3min.txt \
        --manifest data/eval/eval_3min.manifest.jsonl \
        --out runs/tune_longform_small.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicebench.audio import load_wav  # noqa: E402
from voicebench.engines.asr_whisper import FasterWhisperASR  # noqa: E402
from voicebench.metrics.wer import corpus_cer, corpus_wer  # noqa: E402
from voicebench.service import decode_audio_bytes  # noqa: E402

VAD_TUNED = {"min_silence_duration_ms": 300, "speech_pad_ms": 400}

MATRIX = [
    # (tên, beam, vad, vad_params, condition_on_previous_text)
    ("A_base_b5_cond", 5, False, None, True),      # baseline hiện tại
    ("B_vadDef_b5_cond", 5, True, None, True),
    ("C_noCond_b5", 5, False, None, False),
    ("D_vadDef_noCond_b5", 5, True, None, False),
    ("E_vadTuned_noCond_b5", 5, True, VAD_TUNED, False),
    ("F_vadTuned_noCond_b1", 1, True, VAD_TUNED, False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="diepho/PhoWhisper-small-ct2")
    ap.add_argument("--audio", default="data/eval/eval_3min.mp3")
    ap.add_argument("--ref", default="data/eval/eval_3min.txt")
    ap.add_argument("--manifest", default=None,
                    help="nếu có: chạy per-utterance cho cấu hình thắng")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wav, sr = decode_audio_bytes(Path(args.audio).read_bytes())
    ref = Path(args.ref).read_text(encoding="utf-8").strip()
    dur = len(wav) / sr

    print(f"Load {args.model_id} (cpu int8) ...")
    eng = FasterWhisperASR(model_id=args.model_id, device="cpu",
                           compute_type="int8", language="vi")
    # warmup 1 lần cho công bằng
    import numpy as np
    eng.warmup(np.zeros(sr * 2, dtype=np.float32), sr, n=1)

    results = []
    for name, beam, vad, vad_p, cond in MATRIX:
        eng._beam, eng._vad, eng._vad_params, eng._condition = beam, vad, vad_p, cond
        r = eng.transcribe(wav, sr)
        w = corpus_wer([ref], [r.text])
        c = corpus_cer([ref], [r.text])
        row = {"config": name, "beam": beam, "vad": vad, "vad_params": vad_p,
               "condition": cond, "wer": round(w["wer"], 4),
               "cer": round(c["cer"], 4), "S": w["substitutions"],
               "D": w["deletions"], "I": w["insertions"],
               "infer_s": round(r.latency.total_s, 2),
               "rtf": round(r.latency.rtf, 4), "hyp_words": len(r.text.split()),
               "hyp": r.text}
        results.append(row)
        print(f"{name:24s} WER {row['wer']:.2%} CER {row['cer']:.2%} "
              f"S={row['S']:3d} D={row['D']:3d} I={row['I']:2d} "
              f"| {row['infer_s']:6.1f}s RTF {row['rtf']:.3f}")

    best = min(results, key=lambda r: (r["wer"], r["infer_s"]))
    print(f"\nTHẮNG: {best['config']} — WER {best['wer']:.2%}, "
          f"{best['infer_s']:.1f}s (baseline A: WER {results[0]['wer']:.2%}, "
          f"{results[0]['infer_s']:.1f}s)")

    utt = None
    if args.manifest:
        print("\nXác nhận per-utterance với cấu hình thắng (không được tệ đi)...")
        eng._beam = best["beam"]
        eng._vad, eng._vad_params = best["vad"], best["vad_params"]
        eng._condition = best["condition"]
        entries = [json.loads(l) for l in
                   Path(args.manifest).read_text(encoding="utf-8").splitlines()
                   if l.strip()]
        refs, hyps, infers = [], [], []
        for e in entries:
            cw, csr = load_wav(e["audio"])
            rr = eng.transcribe(cw, csr)
            refs.append(e["text"])
            hyps.append(rr.text)
            infers.append(rr.latency.total_s)
        w = corpus_wer(refs, hyps)
        infers.sort()
        utt = {"wer": round(w["wer"], 4),
               "cer": round(corpus_cer(refs, hyps)["cer"], 4),
               "infer_p50_s": round(infers[len(infers) // 2], 3),
               "infer_p90_s": round(infers[int(0.9 * (len(infers) - 1))], 3)}
        print(f"Per-utterance ({best['config']}): WER {utt['wer']:.2%} "
              f"CER {utt['cer']:.2%} | p50 {utt['infer_p50_s']:.2f}s "
              f"p90 {utt['infer_p90_s']:.2f}s")

    if args.out:
        out = {"model_id": args.model_id, "audio": args.audio,
               "audio_dur_s": round(dur, 1),
               "platform": platform.platform(), "results": results,
               "best": best["config"], "per_utterance_best": utt,
               "timestamp": time.time()}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
