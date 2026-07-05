"""So sánh các model CT2 trên CÙNG harness + CÙNG decode params serving chuẩn
(đọc từ configs/service.yaml — T7 tuned) để vẽ đường biên WER vs latency trên
Mac Mini M4. Mỗi model đo cả 2 chế độ benchmark chuẩn:
  - per-utterance: 50 clips data/eval (corpus WER/CER, infer p50/p90)
  - long-form: file 3:16 / 1 request (WER, RTF)

    venv/bin/python scripts/bench_models.py \
        --models diepho/PhoWhisper-tiny-ct2 diepho/PhoWhisper-base-ct2 \
        --out runs/bench_models_tiny_base.json
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from voicebench.audio import load_wav  # noqa: E402
from voicebench.engines.asr_whisper import FasterWhisperASR  # noqa: E402
from voicebench.metrics.wer import corpus_cer, corpus_wer  # noqa: E402
from voicebench.service import decode_audio_bytes  # noqa: E402


def bench_one(model_id: str, kwargs: dict, clips, lf_wav, lf_sr, lf_ref) -> dict:
    print(f"\n== {model_id} ==")
    eng = FasterWhisperASR(**{**kwargs, "model_id": model_id})
    eng.warmup(np.zeros(16000 * 2, dtype=np.float32), 16000, n=2)

    refs, hyps, infers = [], [], []
    for (cw, csr), ref in clips:
        r = eng.transcribe(cw, csr)
        refs.append(ref)
        hyps.append(r.text)
        infers.append(r.latency.total_s)
    w, c = corpus_wer(refs, hyps), corpus_cer(refs, hyps)
    infers.sort()
    p50 = infers[len(infers) // 2]
    p90 = infers[int(0.9 * (len(infers) - 1))]

    rl = eng.transcribe(lf_wav, lf_sr)
    wl = corpus_wer([lf_ref], [rl.text])

    row = {"model_id": model_id, "model_load_s": round(eng._load_s, 2),
           "utt": {"wer": round(w["wer"], 4), "cer": round(c["cer"], 4),
                   "S": w["substitutions"], "D": w["deletions"],
                   "I": w["insertions"], "infer_p50_s": round(p50, 3),
                   "infer_p90_s": round(p90, 3),
                   "empty_hyps": sum(1 for h in hyps if not h.strip())},
           "longform": {"wer": round(wl["wer"], 4),
                        "infer_s": round(rl.latency.total_s, 1),
                        "rtf": round(rl.latency.rtf, 4),
                        "D": wl["deletions"]}}
    print(f"  per-utt : WER {w['wer']:.2%} CER {c['cer']:.2%} "
          f"| p50 {p50:.2f}s p90 {p90:.2f}s | rỗng {row['utt']['empty_hyps']}")
    print(f"  longform: WER {wl['wer']:.2%} | {rl.latency.total_s:.1f}s "
          f"RTF {rl.latency.rtf:.3f}")
    del eng
    gc.collect()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["diepho/PhoWhisper-tiny-ct2",
                             "diepho/PhoWhisper-base-ct2"])
    ap.add_argument("--service-config", default=str(ROOT / "configs/service.yaml"))
    ap.add_argument("--manifest", default=str(ROOT / "data/eval/eval_3min.manifest.jsonl"))
    ap.add_argument("--audio", default=str(ROOT / "data/eval/eval_3min.mp3"))
    ap.add_argument("--ref", default=str(ROOT / "data/eval/eval_3min.txt"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kwargs = yaml.safe_load(Path(args.service_config)
                            .read_text(encoding="utf-8"))["asr"]["kwargs"]
    print(f"Decode params dùng chung (từ {Path(args.service_config).name}): "
          f"beam={kwargs.get('beam_size')} vad={kwargs.get('vad_filter')} "
          f"cond={kwargs.get('condition_on_previous_text')}")

    entries = [json.loads(l) for l in
               Path(args.manifest).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    clips = [(load_wav(str(ROOT / e["audio"])), e["text"]) for e in entries]
    lf_wav, lf_sr = decode_audio_bytes(Path(args.audio).read_bytes())
    lf_ref = Path(args.ref).read_text(encoding="utf-8").strip()

    rows = [bench_one(m, kwargs, clips, lf_wav, lf_sr, lf_ref)
            for m in args.models]

    if args.out:
        out = {"platform": platform.platform(), "decode_params": kwargs,
               "n_clips": len(clips), "results": rows}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
