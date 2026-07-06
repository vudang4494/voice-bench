# Eval OOD — nguồn gốc & license

38 clip stride-sampled từ **VietMed** test split ([leduckhai/VietMed](https://huggingface.co/datasets/leduckhai/VietMed)) — hội thoại y tế thật, đa accent/vai. Domain KHÁC VIVOS (đọc sách sạch) nên WER đo ở đây phản ánh out-of-domain thật của serving config.

License theo dataset gốc (dùng nghiên cứu phi thương mại, giữ attribution). Tạo lại: `venv/bin/python scripts/build_eval_ood.py`.
