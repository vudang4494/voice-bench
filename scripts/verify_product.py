"""Verify product qua quality gates (configs/gates.yaml) -> output chuẩn PASS/FAIL.

Chạy product qua ĐƯỜNG HTTP THẬT (TestClient trên create_app từ configs/service.yaml,
engine thật, warmup thật) với 3 nhóm rules:
  1. unit_tests  — pytest -q phải 100% pass (verify skill Mức 1).
  2. functional  — test case chức năng: health, wav thật, im lặng, bytes rác,
                   file rỗng, TTS 503 contract.
  3. benchmark   — per-utterance 50 clips (corpus WER/CER + p50/p90) + long-form
                   3:16 (WER + RTF) + determinism (lặp lại phải ra cùng text).

Output: bảng PASS/FAIL ra stdout + runs/verify_<ts>/verify_report.md + .json.
Exit code 0 = mọi gate PASS; 1 = có gate FAIL (dùng được cho CI).

    venv/bin/python scripts/verify_product.py
    venv/bin/python scripts/verify_product.py --skip-pytest   # khi vừa chạy pytest tay
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from voicebench.metrics.wer import corpus_cer, corpus_wer  # noqa: E402
from voicebench.metrics.text_norm import normalize_vi  # noqa: E402

GATES = []  # list[dict]: {group, name, passed, measured, threshold}

# Câu smoke TTS: chữ thuần không chữ số/viết tắt — vinorm không chạy trên macOS
# (binary Linux) nên text vào thẳng XTTS; round-trip WER chỉ có nghĩa với chữ thuần.
TTS_SMOKE_TEXT = ("Xin chào, đây là bài kiểm tra tổng hợp giọng nói tiếng Việt "
                  "của voice bench, chạy trên máy tính để bàn nhỏ.")


def gate(group: str, name: str, passed: bool, measured, threshold) -> bool:
    GATES.append({"group": group, "name": name, "passed": bool(passed),
                  "measured": measured, "threshold": threshold})
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {group}/{name}: {measured} (ngưỡng: {threshold})")
    return bool(passed)


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


def _p(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[len(s) // 2] if q == 50 else s[int(0.9 * (len(s) - 1))]


def _clip(path: str) -> Path:
    """Đường dẫn clip trong manifest là tương đối so với voice-bench/."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run_unit_tests(g: dict) -> None:
    print("\n== 1. Unit tests ==")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(không có output)"])[-1]
    gate("unit_tests", "must_pass", r.returncode == 0, tail,
         "pytest exit 0" if g.get("must_pass", True) else "bỏ qua")


def run_functional(client, _g: dict, sample_wav_path: str) -> None:
    # _g giữ chỗ cho việc bật/tắt từng TC sau này; hiện mọi TC đều bắt buộc.
    print("\n== 2. Functional test cases (HTTP) ==")
    h = client.get("/health")
    ok = h.status_code == 200 and h.json().get("status") == "ok"
    gate("functional", "health_ok", ok,
         f"HTTP {h.status_code}, status={h.json().get('status')}", "200 + ok")

    r = client.post("/v1/asr", files={"file": ("c.wav",
                    _clip(sample_wav_path).read_bytes(), "audio/wav")})
    j = r.json() if r.status_code == 200 else {}
    lat_keys = {"decode_s", "infer_total_s", "rtf", "server_total_s"}
    ok = (r.status_code == 200 and j.get("text", "").strip() != ""
          and lat_keys.issubset(j.get("latency", {}).keys())
          and all(j["latency"][k] >= 0 for k in lat_keys))
    gate("functional", "wav_ok", ok,
         f"HTTP {r.status_code}, text={j.get('text', '')[:30]!r}...",
         "200 + text khác rỗng + đủ latency")

    r = client.post("/v1/asr", files={"file": ("sil.wav",
                    _wav_bytes(np.zeros(32000, dtype=np.float32), 16000), "audio/wav")})
    gate("functional", "silence_ok",
         r.status_code == 200 and isinstance(r.json().get("text"), str),
         f"HTTP {r.status_code}, text={r.json().get('text', None)!r}"
         if r.status_code == 200 else f"HTTP {r.status_code}", "200, không crash")

    r = client.post("/v1/asr", files={"file": ("x.bin", b"day khong phai audio" * 100,
                                               "application/octet-stream")})
    gate("functional", "garbage_400", r.status_code == 400,
         f"HTTP {r.status_code}", "400 (không 500)")

    r = client.post("/v1/asr", files={"file": ("e.wav", b"", "audio/wav")})
    gate("functional", "empty_400", r.status_code == 400,
         f"HTTP {r.status_code}", "400")

    r = client.post("/v1/tts", data={"text": "xin chào"})
    gate("functional", "tts_503_when_unconfigured", r.status_code == 503,
         f"HTTP {r.status_code}", "503 kèm lý do (config không có mục tts)")


def run_benchmark(client, gates_cfg: dict, manifest: str, audio: str,
                  ref_txt: str) -> dict:
    print("\n== 3a. Benchmark per-utterance (50 clips) ==")
    g = gates_cfg["per_utterance"]
    entries = [json.loads(l) for l in
               Path(manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    refs, hyps, infers = [], [], []
    for e in entries:
        r = client.post("/v1/asr", files={"file": (Path(e["audio"]).name,
                        _clip(e["audio"]).read_bytes(), "audio/wav")})
        r.raise_for_status()
        j = r.json()
        refs.append(e["text"])
        hyps.append(j["text"])
        infers.append(j["latency"]["infer_total_s"])
    w = corpus_wer(refs, hyps)
    c = corpus_cer(refs, hyps)
    w_nt = corpus_wer(refs, hyps, keep_tone=False)
    p50, p90 = _p(infers, 50), _p(infers, 90)
    n_empty = sum(1 for h in hyps if not h.strip())
    gate("per_utterance", "wer_max", w["wer"] <= g["wer_max"],
         f"{w['wer']:.2%}", f"<= {g['wer_max']:.2%}")
    gate("per_utterance", "cer_max", c["cer"] <= g["cer_max"],
         f"{c['cer']:.2%}", f"<= {g['cer_max']:.2%}")
    gate("per_utterance", "infer_p50_max_s", p50 <= g["infer_p50_max_s"],
         f"{p50:.2f}s", f"<= {g['infer_p50_max_s']}s")
    gate("per_utterance", "infer_p90_max_s", p90 <= g["infer_p90_max_s"],
         f"{p90:.2f}s", f"<= {g['infer_p90_max_s']}s")
    gate("per_utterance", "empty_hyp_max", n_empty <= g["empty_hyp_max"],
         f"{n_empty} clip rỗng", f"<= {g['empty_hyp_max']}")

    print("\n== 3b. Benchmark long-form (1 file / 1 request) ==")
    gl = gates_cfg["longform"]
    r = client.post("/v1/asr", files={"file": (Path(audio).name,
                    Path(audio).read_bytes(), "audio/mpeg")})
    r.raise_for_status()
    j = r.json()
    ref = Path(ref_txt).read_text(encoding="utf-8").strip()
    wl = corpus_wer([ref], [j["text"]])
    rtf = j["latency"]["rtf"]
    gate("longform", "wer_max", wl["wer"] <= gl["wer_max"],
         f"{wl['wer']:.2%}", f"<= {gl['wer_max']:.2%}")
    gate("longform", "rtf_max", rtf <= gl["rtf_max"],
         f"RTF {rtf:.3f}", f"<= {gl['rtf_max']}")

    print("\n== 3c. Determinism (lặp lại phải ra cùng text) ==")
    gd = gates_cfg["determinism"]
    n = int(gd["repeat_n"])
    mismatches = []
    for e in entries[:n]:
        r = client.post("/v1/asr", files={"file": (Path(e["audio"]).name,
                        _clip(e["audio"]).read_bytes(), "audio/wav")})
        r.raise_for_status()
        t2 = r.json()["text"]
        t1 = hyps[entries.index(e)]
        if normalize_vi(t1) != normalize_vi(t2):
            mismatches.append({"clip": e["audio"], "run1": t1, "run2": t2})
    gate("determinism", "require_identical_text",
         len(mismatches) == 0 or not gd["require_identical_text"],
         f"{n - len(mismatches)}/{n} giống hệt", f"{n}/{n}")

    return {"per_utterance": {"n_clips": len(entries), "wer": round(w["wer"], 4),
                              "cer": round(c["cer"], 4),
                              "wer_no_tone": round(w_nt["wer"], 4),
                              "S": w["substitutions"], "D": w["deletions"],
                              "I": w["insertions"],
                              "infer_p50_s": round(p50, 3),
                              "infer_p90_s": round(p90, 3),
                              "empty_hyps": n_empty},
            "longform": {"wer": round(wl["wer"], 4), "rtf": round(rtf, 4),
                         "infer_s": j["latency"]["infer_total_s"],
                         "S": wl["substitutions"], "D": wl["deletions"],
                         "I": wl["insertions"]},
            "determinism": {"repeat_n": n, "mismatches": mismatches}}


def run_tts_gates(client, g: dict, ref_clip: str) -> dict:
    """Gate TTS: contract lỗi + synth thật (voice-clone từ 1 clip eval) +
    nghe lại qua chính ASR của service (round-trip intelligibility)."""
    import soundfile as sf
    print("\n== 4. TTS gates (synth + round-trip qua ASR của service) ==")

    r = client.post("/v1/tts", data={})
    gate("tts", "missing_text_422", r.status_code == 422,
         f"HTTP {r.status_code}", "422 (multipart thiếu field text)")

    r = client.post("/v1/tts", data={"text": "xin chào"},
                    files={"ref_audio": ("x.wav", b"khong phai audio" * 10,
                                         "audio/wav")})
    gate("tts", "ref_garbage_400", r.status_code == 400,
         f"HTTP {r.status_code}", "400 (ref không decode được)")

    ref = _clip(ref_clip)
    r = client.post("/v1/tts", data={"text": TTS_SMOKE_TEXT},
                    files={"ref_audio": (ref.name, ref.read_bytes(), "audio/wav")})
    ok = (r.status_code == 200
          and r.headers.get("content-type", "").startswith("audio/wav"))
    gate("tts", "synth_ok", ok, f"HTTP {r.status_code}", "200 + audio/wav")
    if not ok:
        return {}

    wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    dur = len(wav) / sr
    rms = float(np.sqrt(np.mean(wav ** 2)))
    rtf = float(r.headers.get("x-rtf", "nan"))
    gate("tts", "min_dur_s", dur >= g["min_dur_s"],
         f"{dur:.1f}s audio", f">= {g['min_dur_s']}s")
    gate("tts", "min_rms", rms >= g["min_rms"],
         f"RMS {rms:.3f}", f">= {g['min_rms']} (không câm)")
    gate("tts", "rtf_max", rtf <= g["rtf_max"],
         f"RTF {rtf:.2f}", f"<= {g['rtf_max']}")

    ra = client.post("/v1/asr", files={"file": ("tts_out.wav", r.content,
                                                "audio/wav")})
    ra.raise_for_status()
    hyp = ra.json()["text"]
    w = corpus_wer([TTS_SMOKE_TEXT], [hyp])
    gate("tts", "roundtrip_wer_max", w["wer"] <= g["roundtrip_wer_max"],
         f"{w['wer']:.2%} (nghe lại: {hyp[:50]!r}...)",
         f"<= {g['roundtrip_wer_max']:.0%}")
    return {"tts": {"out_dur_s": round(dur, 1), "rms": round(rms, 3),
                    "rtf": round(rtf, 3), "roundtrip_wer": round(w["wer"], 4),
                    "roundtrip_text": hyp}}


def write_report(out_dir: Path, meta: dict, numbers: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_fail = sum(1 for g in GATES if not g["passed"])
    verdict = "PASS" if n_fail == 0 else f"FAIL ({n_fail} gate)"
    (out_dir / "verify_results.json").write_text(json.dumps(
        {"meta": meta, "verdict": verdict, "gates": GATES, "numbers": numbers},
        ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Verify product — {verdict}", "",
             f"- Thời điểm: {meta['timestamp']}",
             f"- Máy: {meta['platform']}",
             f"- Config: {meta['service_config']} | Gates: {meta['gates_config']}",
             f"- Model: {meta['model_id']}", "",
             "| Gate | Đo được | Ngưỡng | Kết quả |", "|---|---|---|---|"]
    for g in GATES:
        lines.append(f"| {g['group']}/{g['name']} | {g['measured']} | "
                     f"{g['threshold']} | {'✅ PASS' if g['passed'] else '❌ FAIL'} |")
    if "per_utterance" in numbers:
        pu, lf = numbers["per_utterance"], numbers["longform"]
        lines += ["", "## Số benchmark chuẩn", "",
                  f"- Per-utterance ({pu['n_clips']} clips): WER {pu['wer']:.2%} "
                  f"(bỏ tone {pu['wer_no_tone']:.2%}), CER {pu['cer']:.2%}, "
                  f"S/D/I {pu['S']}/{pu['D']}/{pu['I']}, "
                  f"infer p50 {pu['infer_p50_s']}s p90 {pu['infer_p90_s']}s",
                  f"- Long-form: WER {lf['wer']:.2%}, RTF {lf['rtf']}, "
                  f"infer {lf['infer_s']}s, S/D/I {lf['S']}/{lf['D']}/{lf['I']}"]
    if "tts" in numbers:
        t = numbers["tts"]
        lines += ["", "## Số TTS", "",
                  f"- Synth: {t['out_dur_s']}s audio, RMS {t['rms']}, "
                  f"RTF {t['rtf']}",
                  f"- Round-trip WER {t['roundtrip_wer']:.2%} — nghe lại: "
                  f"{t['roundtrip_text']!r}"]
    (out_dir / "verify_report.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    print(f"\n==> {verdict} — report: {out_dir}/verify_report.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-config", default=str(ROOT / "configs/service.yaml"))
    ap.add_argument("--gates", default=str(ROOT / "configs/gates.yaml"))
    ap.add_argument("--manifest", default=str(ROOT / "data/eval/eval_3min.manifest.jsonl"))
    ap.add_argument("--audio", default=str(ROOT / "data/eval/eval_3min.mp3"))
    ap.add_argument("--ref", default=str(ROOT / "data/eval/eval_3min.txt"))
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    gates_cfg = yaml.safe_load(Path(args.gates).read_text(encoding="utf-8"))
    print(f"Gates profile: {gates_cfg.get('profile')}")

    if not args.skip_pytest and "unit_tests" in gates_cfg:
        run_unit_tests(gates_cfg["unit_tests"])

    from fastapi.testclient import TestClient
    from voicebench.service import create_app
    cfg = yaml.safe_load(Path(args.service_config).read_text(encoding="utf-8"))
    print(f"\nBuild app + engine thật ({cfg['asr']['kwargs'].get('model_id')}) ...")
    app = create_app(config=cfg)
    numbers = {}
    entries0 = json.loads(Path(args.manifest).read_text(encoding="utf-8")
                          .splitlines()[0])
    # Mỗi nhóm gate chạy khi gates config CÓ section tương ứng — profile TTS
    # (gates.tts.yaml) chỉ khai báo section tts, khỏi lặp lại benchmark ASR
    # đã được gates.yaml mặc định rào.
    with TestClient(app) as client:  # __enter__ chạy startup: build engine + warmup
        if "functional" in gates_cfg:
            run_functional(client, gates_cfg["functional"], entries0["audio"])
        if "per_utterance" in gates_cfg:
            numbers = run_benchmark(client, gates_cfg, args.manifest, args.audio,
                                    args.ref)
        if "tts" in gates_cfg:
            numbers.update(run_tts_gates(client, gates_cfg["tts"],
                                         entries0["audio"]))

    import faster_whisper
    meta = {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "faster_whisper": faster_whisper.__version__,
            "service_config": args.service_config, "gates_config": args.gates,
            "model_id": cfg["asr"]["kwargs"].get("model_id"),
            "asr_kwargs": cfg["asr"]["kwargs"]}
    out_dir = ROOT / "runs" / f"verify_{time.strftime('%Y%m%d_%H%M%S')}"
    write_report(out_dir, meta, numbers)
    sys.exit(0 if all(g["passed"] for g in GATES) else 1)


if __name__ == "__main__":
    main()
