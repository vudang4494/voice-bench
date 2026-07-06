"""Smoke test TTS viXTTS trên CPU: synth 1 câu + round-trip qua ASR serving để
kiểm tra intelligibility. Đây là lần chạy TTS thật đầu tiên sau khi thông deps
(coqui-tts fork thay TTS 0.22 — T4).

    venv/bin/python scripts/smoke_tts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicebench.audio import load_wav, save_wav  # noqa: E402
from voicebench.engines.registry import build_asr, build_tts  # noqa: E402
from voicebench.metrics.wer import corpus_wer  # noqa: E402

# Tránh chữ số/viết tắt: vinorm không chạy trên macOS (binary Linux) nên text
# vào thẳng XTTS — câu smoke phải là chữ thuần để round-trip WER có nghĩa.
TEXT = ("Xin chào, đây là bài kiểm tra tổng hợp giọng nói tiếng Việt "
        "của voice bench, chạy trên máy tính để bàn nhỏ.")


def main() -> int:
    print("Load viXTTS (CPU)...")
    tts = build_tts("vixtts", {
        "model_path": str(ROOT / "models/viXTTS"),
        "config_path": str(ROOT / "models/viXTTS/config.json"),
        "device": "cpu", "language": "vi",
    })
    print(f"  model_load_s: {tts._load_s:.1f}s")

    ref_wav, ref_sr = load_wav(str(ROOT / "models/viXTTS/vi_sample.wav"))
    r = tts.synthesize(TEXT, ref_wav, ref_sr)
    out = ROOT / "runs/tts_smoke.wav"
    out.parent.mkdir(exist_ok=True)
    save_wav(str(out), r.audio, r.sample_rate)
    rtf = r.latency.total_s / r.out_dur_s if r.out_dur_s else float("inf")
    print(f"  synth: {r.latency.total_s:.1f}s cho {r.out_dur_s:.1f}s audio "
          f"(RTF {rtf:.2f}) @ {r.sample_rate}Hz -> {out}")

    print("Round-trip qua ASR serving (small)...")
    asr = build_asr("faster-whisper", {
        "model_id": "vudang449/PhoWhisper-small-ct2", "device": "cpu",
        "compute_type": "int8", "language": "vi", "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 300, "speech_pad_ms": 400},
        "condition_on_previous_text": False,
    })
    rt = asr.transcribe(r.audio, r.sample_rate)
    w = corpus_wer([TEXT], [rt.text])
    wf = corpus_wer([TEXT], [rt.text], fold_variants=True)
    print(f"  nghe lại: {rt.text}")
    print(f"  round-trip WER: {w['wer']:.2%} (fold {wf['wer']:.2%}) "
          f"— S{w['substitutions']} D{w['deletions']} I{w['insertions']}")
    ok = w["wer"] <= 0.35  # smoke: chỉ cần nghe ra đại ý, chưa phải gate chất lượng
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
