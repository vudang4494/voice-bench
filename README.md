# voice-bench

Harness đo **Latency + Accuracy** cho voice pipeline tiếng Việt (STT + TTS round-trip), kèm **service HTTP STT** đã tune sẵn cho CPU Apple Silicon. Offline batch, engine hoán đổi qua config.

Model weights **không nằm trong repo** — tải từ HuggingFace ([vudang449/PhoWhisper-small-ct2](https://huggingface.co/vudang449/PhoWhisper-small-ct2), [vudang449/PhoWhisper-large-ct2](https://huggingface.co/vudang449/PhoWhisper-large-ct2), mirror của PhoWhisper/VinAI bản CTranslate2).

## Quickstart (clone về là chạy)

```bash
git clone https://github.com/vudang4494/voice-bench && cd voice-bench
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt

venv/bin/python scripts/download_models.py      # tải model small từ HF (~919MB; --all để thêm large)
venv/bin/python -m pytest -q                     # unit tests, <5s, không cần model

# Verify product qua quality gates (tải model + benchmark chuẩn + test chức năng, ~3-5 phút)
venv/bin/python scripts/verify_product.py

# Chạy service STT
venv/bin/uvicorn voicebench.service:app --port 8386
curl -F "file=@audio.wav" http://localhost:8386/v1/asr
```

Cần accuracy cao nhất (chậm ~5×): `VOICEBENCH_SERVICE_CONFIG=configs/service.large.yaml venv/bin/uvicorn voicebench.service:app --port 8386`.

## Số đo chuẩn (Mac Mini M4 24GB, CPU int8, eval VIVOS-test 50 clips + long-form 3:16)

| Model | WER clip ngắn | p50/p90 | WER long-form / RTF | Kết luận |
|---|---|---|---|---|
| PhoWhisper-tiny-ct2 | 14.13% | 0.41s / 0.46s | 16.51% / 0.036 | loại (accuracy kém) |
| PhoWhisper-base-ct2 | 14.31% | 0.62s / 0.67s | 12.29% / 0.059 | loại (xóa nội dung) |
| **PhoWhisper-small-ct2** | **6.24%** (CER 3.17%) | **1.51s / 1.65s** | 9.17% / 0.143 | **serving mặc định** |
| PhoWhisper-medium-ct2 | 6.97% | 4.26s / 4.58s | 8.26% / 0.368 | loại (thua small, chậm hơn) |
| **PhoWhisper-large-ct2** | **5.50%** | 7.65s / 8.12s | **5.50%** / 0.693 | **profile accuracy** |

Decode params đã tune (hardcode trong `configs/service.yaml`): beam 5 + VAD (min_silence 300ms, pad 400ms) + `condition_on_previous_text=False`. Gotchas đã đo được:

- **Đừng hạ beam xuống 1**: clip ngắn tiếng Việt hallucinate insertions (WER 12.7–13%).
- **`condition_on_previous_text=True` là thủ phạm chính mất nội dung long-form** ở ranh giới cửa sổ 30s (18.35% → 11.56% khi tắt).
- **Params không chuyển giao giữa các cỡ model**: params tune trên small làm large tệ đi — `service.large.yaml` dùng profile theo độ dài (clip ngắn giữ mặc định, file ≥30s mới bật VAD + tắt condition).

## Quality gates

Rules chuẩn ở `configs/gates.yaml` (ngưỡng WER/CER/latency/determinism từ baseline đo thật + headroom). `scripts/verify_product.py` chạy product qua HTTP thật: unit tests → 6 test case chức năng → benchmark chuẩn → determinism; exit 0/1 dùng được cho CI. Nguyên tắc: **gate FAIL = điều tra, không nới ngưỡng**.

## Ý tưởng đo round-trip (phần TTS)

```
ref_audio ──[ASR#1]──► asr_text          → đo ASR: WER(ref_text, asr_text)
ref_text  ──[TTS clone ref voice]──► out_audio
out_audio ──[ASR#2]──► roundtrip_text    → đo TTS intelligibility
                       speaker_sim(ref_audio, out_audio) + MOS(out_audio)
```

**Cô lập lỗi**: round-trip WER trộn lỗi ASR + TTS. Mặc định TTS đọc `ref_text` nên `ΔWER = round-trip_WER − ASR_WER` là phần lỗi quy cho TTS đã trừ nền ASR. Đổi `tts_input_source: asr_text` để đo echo tích luỹ thật. (TTS engines: viXTTS / VietTTS — đang chờ fix xung khắc deps Coqui TTS ↔ transformers mới.)

```bash
venv/bin/python -m voicebench.run_benchmark -c configs/default.yaml
venv/bin/python -m voicebench.report -r runs/<timestamp>/results.jsonl -o report.md
```

## Metrics & invariants đo lường

| Nhóm | Metric | Ghi chú |
|---|---|---|
| Accuracy | WER/CER corpus-pooled | gộp S/D/I chia tổng từ — KHÔNG mean WER từng câu. CER quan trọng cho VN (lỗi tone); chuẩn hoá giữ dấu thanh |
| Latency | median + p90 + RTF, TTFA (khi streaming) | KHÔNG mean (đuôi lệch phải); CUDA sync 2 đầu; `model_load_s` tách riêng, warmup trước khi đo |
| Fidelity | speaker_sim (ECAPA cosine) + 95% BCa CI | cần same/diff-speaker baseline mới diễn giải được |
| Naturalness | MOS (UTMOS, optional) | tương quan thô |

Mỗi run ghi `results.jsonl` (raw per-sample, re-aggregate được), `run_meta.json` (seed + hardware + versions + commit), `results.agg.json`, `report.md`.

## Kiến trúc

- `voicebench/engines/` — plugin: `registry.py` map tên config → class, lazy import (thiếu deps engine khác không sao). Thêm engine = viết class theo contract `base.py` + đăng ký.
- `voicebench/metrics/` — text_norm VN, WER/CER, latency, BCa bootstrap, speaker_sim, MOS.
- `voicebench/service.py` — FastAPI: `/v1/asr`, `/v1/tts` (503 khi chưa cấu hình), `/health`.
- `scripts/` — `download_models.py`, `verify_product.py` (gates), `bench_models.py` (so sánh model), `eval_service.py`, `tune_longform.py`, `build_eval_set.py`.

## Data eval

`data/eval/` = 50 clips VIVOS test (stride-sampled, đa speaker) + file ghép 3:16 — dẫn xuất từ [VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos) (CC BY-NC-SA 4.0, xem `data/eval/README.md`). Lưu ý: VIVOS nằm trong data train của PhoWhisper nên số đo hơi lạc quan so với audio ngoài domain.

## License & attribution

Model: PhoWhisper của [VinAI Research](https://huggingface.co/vinai) (BSD-3-Clause), bản convert CTranslate2 từ cộng đồng (diepho, kiendt), mirror giữ nguyên trọng số. Data eval: VIVOS (AILAB VNUHCM, CC BY-NC-SA 4.0).
