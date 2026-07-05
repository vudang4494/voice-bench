# Data eval — nguồn gốc & license

Bộ eval chuẩn của voice-bench, dẫn xuất từ **VIVOS** (AILAB, VNU-HCM University of Science):

- `clips/` — 50 clip WAV lấy từ VIVOS **test split**, stride-sampled để phủ nhiều speaker.
- `eval_3min.mp3` / `eval_3min.wav` — 195.6s ghép từ các clip VIVOS test (đo long-form).
- `eval_3min.manifest.jsonl` — mỗi dòng `{"id", "audio", "text"}` (transcript gốc VIVOS).
- `eval_3min.txt` — transcript ghép cho file long-form (545 từ).

Tạo lại từ đầu: `venv/bin/python scripts/build_eval_set.py`.

**License: CC BY-NC-SA 4.0** — theo dataset gốc [AILAB-VNUHCM/vivos](https://huggingface.co/datasets/AILAB-VNUHCM/vivos). Phần data trong thư mục này chỉ dùng phi thương mại, giữ attribution và share-alike theo đúng license gốc.

**Caveat đo lường**: VIVOS nằm trong data train của PhoWhisper → WER đo trên bộ này là cận dưới lạc quan (in-domain). Đánh giá out-of-domain cần thêm nguồn khác (Common Voice vi, audio tự thu).
