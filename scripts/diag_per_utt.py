"""T7 chẩn đoán: config thắng long-form F (beam1+VAD+noCond) tụt per-utterance
(13.03% vs baseline 6.61%). Tách nguyên nhân beam1 vs VAD vs noCond bằng cách
chạy per-utterance 3 cấu hình trung gian trên cùng manifest:

    E: beam5 + vadTuned + noCond  -> nếu tệ: VAD là thủ phạm
    G: beam5 + noVAD   + noCond  -> cô lập ảnh hưởng noCond (kỳ vọng ~baseline)
    H: beam1 + noVAD   + cond    -> nếu tệ: beam1 là thủ phạm

Mốc đã biết: baseline (b5,noVAD,cond) 6.61%; F (b1,vadTuned,noCond) 13.03%.
Kết quả quyết định params hardcode cho profile long-form trong configs/service.yaml.

    venv/bin/python scripts/diag_per_utt.py --out runs/diag_per_utt_small.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicebench.audio import load_wav  # noqa: E402
from voicebench.engines.asr_whisper import FasterWhisperASR  # noqa: E402
from voicebench.metrics.wer import corpus_cer, corpus_wer  # noqa: E402

VAD_TUNED = {"min_silence_duration_ms": 300, "speech_pad_ms": 400}

MATRIX = [
    # (tên, beam, vad, vad_params, condition_on_previous_text)
    ("E_vadTuned_noCond_b5", 5, True, VAD_TUNED, False),
    ("G_noVad_noCond_b5", 5, False, None, False),
    ("H_noVad_cond_b1", 1, False, None, True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="diepho/PhoWhisper-small-ct2")
    ap.add_argument("--manifest", default="data/eval/eval_3min.manifest.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    entries = [json.loads(l) for l in
               Path(args.manifest).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    clips = [(load_wav(e["audio"]), e["text"]) for e in entries]

    print(f"Load {args.model_id} (cpu int8), {len(clips)} clips ...")
    eng = FasterWhisperASR(model_id=args.model_id, device="cpu",
                           compute_type="int8", language="vi")
    import numpy as np
    eng.warmup(np.zeros(16000 * 2, dtype=np.float32), 16000, n=1)

    results = []
    for name, beam, vad, vad_p, cond in MATRIX:
        eng._beam, eng._vad, eng._vad_params, eng._condition = beam, vad, vad_p, cond
        refs, hyps, infers = [], [], []
        for (cw, csr), ref in clips:
            r = eng.transcribe(cw, csr)
            refs.append(ref)
            hyps.append(r.text)
            infers.append(r.latency.total_s)
        w = corpus_wer(refs, hyps)
        infers.sort()
        row = {"config": name, "beam": beam, "vad": vad, "vad_params": vad_p,
               "condition": cond, "wer": round(w["wer"], 4),
               "cer": round(corpus_cer(refs, hyps)["cer"], 4),
               "S": w["substitutions"], "D": w["deletions"],
               "I": w["insertions"],
               "infer_p50_s": round(infers[len(infers) // 2], 3),
               "infer_p90_s": round(infers[int(0.9 * (len(infers) - 1))], 3)}
        results.append(row)
        print(f"{name:24s} WER {row['wer']:.2%} CER {row['cer']:.2%} "
              f"S={row['S']:3d} D={row['D']:3d} I={row['I']:2d} "
              f"| p50 {row['infer_p50_s']:.2f}s p90 {row['infer_p90_s']:.2f}s")

    if args.out:
        out = {"model_id": args.model_id, "manifest": args.manifest,
               "anchors": {"baseline_b5_noVad_cond": 0.0661,
                           "F_vadTuned_noCond_b1": 0.1303},
               "results": results}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
