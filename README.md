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

Các biến thể config (chạy qua `VOICEBENCH_SERVICE_CONFIG=<file> venv/bin/uvicorn voicebench.service:app --port 8386`):

- `configs/service.yaml` — mặc định: PhoWhisper-small-ct2 (STT only).
- `configs/service.large.yaml` — accuracy VIVOS cao nhất, chậm ~5× (lưu ý: ngoài domain KHÔNG tốt hơn small, xem bảng OOD).
- `configs/service.chunkformer.yaml` — ứng viên mới: nhanh 9×, WER bằng large (tải model: `download_models.py --chunkformer`).
- `configs/service.tts.yaml` — STT + TTS viXTTS voice-clone (tải checkpoint: `download_models.py --tts`).

## Số đo chuẩn (Mac Mini M4 24GB, CPU int8, eval VIVOS-test 50 clips + long-form 3:16)

| Model | WER clip ngắn | p50/p90 | WER long-form / RTF | Kết luận |
|---|---|---|---|---|
| PhoWhisper-tiny-ct2 | 14.13% | 0.41s / 0.46s | 16.51% / 0.036 | loại (accuracy kém) |
| PhoWhisper-base-ct2 | 14.31% | 0.62s / 0.67s | 12.29% / 0.059 | loại (xóa nội dung) |
| **PhoWhisper-small-ct2** | **6.24%** (CER 3.17%) | **1.51s / 1.65s** | 9.17% / 0.143 | **serving mặc định** |
| PhoWhisper-medium-ct2 | 6.97% | 4.26s / 4.58s | 8.26% / 0.368 | loại (thua small, chậm hơn) |
| **PhoWhisper-large-ct2** | **5.50%** | 7.65s / 8.12s | **5.50%** / 0.693 | **profile accuracy** |
| **ChunkFormer-large-vie** | **5.50%** (CER 2.91%) | **0.17s / 0.18s** | 6.61% / **0.073** | **ứng viên serving mới** — `configs/service.chunkformer.yaml` |

ChunkFormer (~110M, CTC chunk-based) bằng large trên VIVOS nhưng **nhanh hơn 45×**,
không bị floor padding 30s của whisper — model duy nhất đạt <500ms với WER <10% trên M4.

## Số đo ngoài domain (VietMed test — hội thoại y tế, `data/eval_ood/`, 38 clips)

| Model | WER OOD | p50 | Ghi chú |
|---|---|---|---|
| PhoWhisper-small-ct2 | 22.20% | 2.00s | gap 3.5× so VIVOS — số "thật" ngoài domain |
| PhoWhisper-large-ct2 | 25.12% | 10.25s | **THUA small ngoài domain** (đảo thứ hạng!) |
| ChunkFormer-large-vie | 18.81%* | 0.30s | *VietMed_labeled NẰM TRONG train data của ChunkFormer — số này không chứng minh ưu thế OOD |

Bài học đã đo được: (1) WER VIVOS là cận dưới lạc quan — ngoài domain gấp ~3.5×;
(2) model to hơn KHÔNG chắc tốt hơn ngoài domain; (3) mọi model VN public đều
train trên gần hết dataset VN public — đánh giá OOD sạch cần audio tự thu.

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

**Cô lập lỗi**: round-trip WER trộn lỗi ASR + TTS. Mặc định TTS đọc `ref_text` nên `ΔWER = round-trip_WER − ASR_WER` là phần lỗi quy cho TTS đã trừ nền ASR. Đổi `tts_input_source: asr_text` để đo echo tích luỹ thật.

TTS viXTTS **đã chạy được** (qua fork `coqui-tts` của idiap — TTS 0.22 gốc xung khắc transformers mới): trên CPU M4 synth RTF ~0.96, round-trip smoke WER 8.7% (`scripts/smoke_tts.py`). Caveat macOS: `vinorm` là binary Linux nên text có chữ số/viết tắt vào thẳng XTTS (engine tự fallback + warning).

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
- `scripts/` — `download_models.py`, `verify_product.py` (gates), `bench_models.py` (so sánh model, `--engine chunkformer` được), `smoke_tts.py` (TTS round-trip), `eval_service.py`, `tune_longform.py`, `build_eval_set.py`, `build_eval_ood.py`.

## Data eval

`data/eval/` = 50 clips VIVOS test (stride-sampled, đa speaker) + file ghép 3:16 — dẫn xuất từ [VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos) (CC BY-NC-SA 4.0, xem `data/eval/README.md`). Lưu ý: VIVOS nằm trong data train của PhoWhisper (và ChunkFormer) nên số đo hơi lạc quan so với audio ngoài domain.

`data/eval_ood/` = 38 clips [VietMed](https://huggingface.co/datasets/leduckhai/VietMed) test (hội thoại y tế, đa accent) — đo out-of-domain cho dòng PhoWhisper; tạo lại bằng `scripts/build_eval_ood.py`.

## License & attribution

Model: PhoWhisper của [VinAI Research](https://huggingface.co/vinai) (BSD-3-Clause), bản convert CTranslate2 từ cộng đồng (diepho, kiendt); ChunkFormer-large-vie của [khanhld](https://huggingface.co/khanhld/chunkformer-large-vie) (CC-BY-NC-4.0); viXTTS của [capleaf](https://huggingface.co/capleaf/viXTTS). Mirror trên vudang449 giữ nguyên trọng số. Data eval: VIVOS (AILAB VNUHCM, CC BY-NC-SA 4.0), VietMed (leduckhai).
