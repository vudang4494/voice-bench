"""Đánh giá service STT: latency (client/server/infer/RTF) + accuracy (WER/CER).

HAI CHẾ ĐỘ (số WER của 2 chế độ KHÔNG so sánh trực tiếp với nhau):
1. Long-form (--audio + --ref): 1 file dài qua 1 request. WER gồm cả lỗi
   ranh giới ghép clip/cửa sổ 30s của whisper — đo robustness long-form,
   KHÔNG so được với số per-utterance chuẩn VIVOS.
2. Per-utterance (--manifest): mỗi clip 1 request, WER/CER corpus-pooled
   trên từng cặp (ref, hyp) — đúng chuẩn benchmark; latency có median/p90
   per-request (metric service thực).

Cách dùng (service phải đang chạy):
    venv/bin/python scripts/eval_service.py --audio data/eval/eval_3min.mp3 \
        --ref data/eval/eval_3min.txt --runs 3 --out runs/service_eval.json
    venv/bin/python scripts/eval_service.py --manifest data/eval/eval_3min.manifest.jsonl \
        --out runs/service_eval_utt.json

Quy ước đo (theo CLAUDE.md):
- Latency báo median/p90, KHÔNG mean. (n=2 thì median = trung bình 2 giá trị —
  in kèm min/max; nên chạy n>=3.)
- WER/CER corpus-pooled, keep_tone=true (mặc định) VÀ keep_tone=false
  (tách lỗi dấu thanh khỏi lỗi âm tiết).
- JSON output kèm meta (hardware, versions, git commit) — reproducibility.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voicebench.metrics.wer import corpus_cer, corpus_wer  # noqa: E402


def _meta() -> dict:
    """Hardware/versions/commit — quy ước run_meta.json (CLAUDE.md invariant 7)."""
    meta = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version()}
    try:
        meta["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        pass
    for mod in ("faster_whisper", "ctranslate2", "soundfile"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            meta[mod] = None
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        meta["git_commit"] = "unknown"
    return meta


def _acc(refs: list[str], hyps: list[str]) -> dict:
    acc = {}
    for kt in (True, False):
        w = corpus_wer(refs, hyps, keep_tone=kt)
        c = corpus_cer(refs, hyps, keep_tone=kt)
        acc[f"keep_tone={kt}"] = {
            "wer": round(w["wer"], 4), "cer": round(c["cer"], 4),
            "S": w["substitutions"], "D": w["deletions"], "I": w["insertions"],
            "ref_words": w["ref_words"],
        }
    return acc


def _post_audio(url: str, name: str, data: bytes) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/asr", files={"file": (name, data)}, timeout=1800)
    client_s = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), client_s


def _p(vals: list[float], q: float) -> float:
    s = sorted(vals)
    k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[int(k)]


def eval_manifest(args, health) -> None:
    """Per-utterance: mỗi clip 1 request — WER pooled chuẩn + latency p50/p90."""
    entries = [json.loads(l) for l in
               Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    root = Path(args.manifest).resolve().parent.parent.parent  # repo root (data/eval/..)
    refs, hyps, lats, dur_total = [], [], [], 0.0
    for i, e in enumerate(entries):
        p = Path(e["audio"])
        if not p.exists():
            p = root / e["audio"]
        body, client_s = _post_audio(args.url, p.name, p.read_bytes())
        refs.append(e["text"])
        hyps.append(body["text"])
        lat = body["latency"]
        lat["client_total_s"] = client_s
        lats.append(lat)
        dur_total += body["audio"]["duration_s"]
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(entries)} clips ...")
    infer = [x["infer_total_s"] for x in lats]
    server = [x["server_total_s"] for x in lats]
    rtf = [x["rtf"] for x in lats]
    acc = _acc(refs, hyps)
    print("\n== KẾT QUẢ (per-utterance) ==")
    print(f"{len(entries)} clips | speech {dur_total:.1f}s | tổng infer {sum(infer):.1f}s")
    print(f"Latency per-request: infer p50 {statistics.median(infer):.2f}s "
          f"p90 {_p(infer, 0.9):.2f}s | server p50 {statistics.median(server):.2f}s "
          f"| RTF p50 {statistics.median(rtf):.3f} (chú ý: whisper pad mỗi clip lên "
          f"cửa sổ 30s nên RTF clip ngắn cao hơn long-form)")
    for kt, a in acc.items():
        print(f"Accuracy ({kt}): WER {a['wer']:.2%} | CER {a['cer']:.2%} "
              f"| S={a['S']} D={a['D']} I={a['I']} / {a['ref_words']} từ")
    if args.out:
        out = {"mode": "per-utterance", "manifest": args.manifest,
               "n_clips": len(entries), "speech_dur_s": round(dur_total, 1),
               "engine": health["asr"], "meta": _meta(),
               "latency_per_request": {
                   "infer_p50_s": statistics.median(infer), "infer_p90_s": _p(infer, 0.9),
                   "server_p50_s": statistics.median(server), "server_p90_s": _p(server, 0.9),
                   "rtf_p50": statistics.median(rtf), "rtf_p90": _p(rtf, 0.9)},
               "accuracy": acc,
               "pairs": [{"id": e["id"], "ref": r, "hyp": h}
                          for e, r, h in zip(entries, refs, hyps)],
               "timestamp": time.time()}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nĐã ghi {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8386")
    ap.add_argument("--audio", help="file audio dài (chế độ long-form)")
    ap.add_argument("--ref", help="file text ground-truth (long-form)")
    ap.add_argument("--manifest", help="manifest jsonl (chế độ per-utterance)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None, help="ghi JSON kết quả")
    args = ap.parse_args()

    health = requests.get(f"{args.url}/health", timeout=10).json()
    if health.get("status") != "ok":
        raise SystemExit(f"Service chưa sẵn sàng: {health}")
    print(f"Engine: {health['asr']['name']} | model: {health['asr']['model']} "
          f"| model_load_s: {health['asr']['model_load_s']}")

    if args.manifest:
        eval_manifest(args, health)
        return
    if not (args.audio and args.ref):
        raise SystemExit("Cần --manifest HOẶC (--audio và --ref)")

    audio_bytes = Path(args.audio).read_bytes()
    ref_text = Path(args.ref).read_text(encoding="utf-8").strip()

    runs = []
    for i in range(args.runs):
        body, client_s = _post_audio(args.url, Path(args.audio).name, audio_bytes)
        body["client_total_s"] = round(client_s, 4)
        runs.append(body)
        lat = body["latency"]
        print(f"run {i+1}: client={client_s:.2f}s server={lat['server_total_s']:.2f}s "
              f"infer={lat['infer_total_s']:.2f}s decode={lat['decode_s']:.3f}s "
              f"RTF={lat['rtf']:.3f}")

    hyp = runs[0]["text"]
    for i, b in enumerate(runs[1:], 2):
        if b["text"] != hyp:
            print(f"⚠️ run {i} cho text KHÁC run 1 (non-determinism?)")

    dur = runs[0]["audio"]["duration_s"]
    med = {k: statistics.median(r["latency"][k] for r in runs)
           for k in ("decode_s", "infer_total_s", "rtf", "server_total_s")}
    med["client_total_s"] = statistics.median(r["client_total_s"] for r in runs)
    infers = [r["latency"]["infer_total_s"] for r in runs]

    acc = _acc([ref_text], [hyp])

    print("\n== KẾT QUẢ (long-form, 1 request/file) ==")
    print(f"Audio: {dur:.1f}s | ref {acc['keep_tone=True']['ref_words']} từ "
          f"| hyp {len(hyp.split())} từ | {args.runs} runs")
    n_note = " (n=2: median = trung bình 2 run)" if args.runs == 2 else ""
    print(f"Latency (median{n_note}, min {min(infers):.2f}s max {max(infers):.2f}s): "
          f"client {med['client_total_s']:.2f}s | "
          f"server {med['server_total_s']:.2f}s | infer {med['infer_total_s']:.2f}s | "
          f"decode {med['decode_s']:.3f}s | RTF {med['rtf']:.3f}")
    for kt, a in acc.items():
        print(f"Accuracy ({kt}): WER {a['wer']:.2%} | CER {a['cer']:.2%} "
              f"| S={a['S']} D={a['D']} I={a['I']}")
    tone_err = acc["keep_tone=True"]["wer"] - acc["keep_tone=False"]["wer"]
    print(f"Phần WER quy cho lỗi dấu thanh (ước lượng): {tone_err:.2%}")
    print("Lưu ý: WER long-form gồm cả lỗi ranh giới ghép clip — số chuẩn "
          "benchmark dùng chế độ --manifest (per-utterance).")

    if args.out:
        out = {
            "mode": "long-form", "audio": args.audio, "audio_dur_s": dur,
            "ref": args.ref, "engine": health["asr"], "meta": _meta(),
            "runs": runs, "latency_median": med, "accuracy": acc,
            "hyp_text": hyp, "timestamp": time.time(),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
