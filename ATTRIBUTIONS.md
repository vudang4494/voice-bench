# ATTRIBUTIONS — Ghi công tác giả & license

File này là nguồn chuẩn duy nhất về ghi công trong voice-bench. Mirror trên
HuggingFace `vudang449/*` giữ nguyên trọng số gốc, chỉ phục vụ tải nhanh/ổn định.

## Models

### ChunkFormer — Khanh Le (khanhld)

- **Tác giả:** Khanh Le, Tuan Vu Ho, Dung Tran, Duc Thanh Chau — ICASSP 2025.
- **Code/kiến trúc:** <https://github.com/khanld/chunkformer> — **CC-BY-4.0**.
  Dùng qua package pip `chunkformer` trong engine `voicebench/engines/asr_chunkformer.py`.
- **Weights `chunkformer-large-vie`:** <https://huggingface.co/khanhld/chunkformer-large-vie>
  — **CC-BY-NC-4.0** (phi thương mại). Chạy local tại `models/chunkformer-large-vie/`;
  mirror `vudang449/chunkformer-large-vie` giữ nguyên từng byte.
- **Thay đổi:** không sửa model/trọng số — voice-bench chỉ đóng gói cách gọi
  (routing clip ngắn/dài, đo latency).
- **Trích dẫn:**

```bibtex
@INPROCEEDINGS{10888640,
  author={Le, Khanh and Ho, Tuan Vu and Tran, Dung and Chau, Duc Thanh},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription},
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10888640}}
```

### PhoWhisper — VinAI Research

- **Tác giả:** Thanh-Thien Le, Linh The Nguyen, Dat Quoc Nguyen (VinAI) —
  "PhoWhisper: Automatic Speech Recognition for Vietnamese", ICLR 2024 Tiny Papers.
- **Model gốc:** <https://huggingface.co/vinai> — **BSD-3-Clause**.
- **Bản convert CTranslate2** (chạy trong voice-bench): cộng đồng —
  [diepho/PhoWhisper-small-ct2](https://huggingface.co/diepho/PhoWhisper-small-ct2),
  [kiendt/PhoWhisper-large-ct2](https://huggingface.co/kiendt/PhoWhisper-large-ct2).
  Mirror: `vudang449/PhoWhisper-{small,large}-ct2` (không đổi trọng số).

### viXTTS — capleaf

- **Model:** <https://huggingface.co/capleaf/viXTTS> — fine-tune XTTS-v2 cho tiếng Việt.
- **License:** Coqui Public Model License (theo model card gốc).

## Runtime / thư viện chính

| Thư viện | Vai trò | License |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | inference PhoWhisper CT2 | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | backend int8 CPU | MIT |
| [chunkformer](https://github.com/khanld/chunkformer) | inference ChunkFormer | CC-BY-4.0 |
| [coqui-tts](https://github.com/idiap/coqui-ai-TTS) (fork idiap) | inference viXTTS | MPL-2.0 |

## Data eval

- **VIVOS** — AILAB, VNU-HCM (<https://huggingface.co/datasets/AILAB-VNUHCM/vivos>),
  **CC BY-NC-SA 4.0**. Tải qua mirror parquet cộng đồng
  [quocanh34/viet_vivos](https://huggingface.co/datasets/quocanh34/viet_vivos)
  (không đổi nội dung). Dẫn xuất: `data/eval/` (50 clips stride + file ghép 3:16)
  và phần ngắn/trung của `data/manifest_v1.jsonl`.
- **FLEURS** — Google (<https://huggingface.co/datasets/google/fleurs>, config
  `vi_vn`), **CC-BY-4.0** (Conneau et al., "FLEURS: Few-shot Learning Evaluation
  of Universal Representations of Speech", 2022). Dẫn xuất: phần trung/dài của
  `data/manifest_v1.jsonl` (validation split).
- **VietMed** — Khai Le-Duc (<https://huggingface.co/datasets/leduckhai/VietMed>),
  dùng nghiên cứu phi thương mại, giữ attribution. Dẫn xuất: `data/eval_ood/` (38 clips).
