"""ST-2 — full fine-tune PhoWhisper-small trên train_mix_v1 đã lọc QC (MPS, fp32).

Data đã qua verify_asr_dataset.py PASS 19/19 (dedup + drop poison WER>0.9).
Nghiệm thu SAU train bằng verify_product.py — CHỈ nhận model nếu thắng baseline
(WER 6.24% VIVOS / longform 9.17%); không thắng thì giữ nguyên model đang chạy.

Chạy smoke (đo tốc độ, 30 step, không eval):
  venv/bin/python scripts/train_st2.py --smoke
Chạy thật:
  venv/bin/python scripts/train_st2.py --max_steps 4000 --out runs/st2
"""
import os, json, argparse, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import torch, soundfile as sf
from dataclasses import dataclass
from typing import Any
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)


class ManifestDS(torch.utils.data.Dataset):
    """Đọc manifest jsonl {audio, text}; tính mel + tokenize LAZY trong __getitem__."""
    def __init__(self, path, proc):
        self.rows = [json.loads(l) for l in open(path) if l.strip()]
        self.proc = proc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x, sr = sf.read(r["audio"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        feat = self.proc.feature_extractor(x, sampling_rate=16000).input_features[0]
        labels = self.proc.tokenizer(r["text"]).input_ids
        return {"input_features": feat, "labels": labels}


@dataclass
class Collator:
    proc: Any
    dtype: Any = None
    def __call__(self, features):
        inp = [{"input_features": f["input_features"]} for f in features]
        batch = self.proc.feature_extractor.pad(inp, return_tensors="pt")
        if self.dtype is not None:
            batch["input_features"] = batch["input_features"].to(self.dtype)
        lab = [{"input_ids": f["labels"]} for f in features]
        lb = self.proc.tokenizer.pad(lab, return_tensors="pt")
        labels = lb["input_ids"].masked_fill(lb.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.proc.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]  # model tự thêm decoder_start -> cắt bos ở label
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="vinai/PhoWhisper-small")
    ap.add_argument("--train", default="data/train_mix_v1/train.jsonl")
    ap.add_argument("--val", default="data/train_mix_v1/val.jsonl")
    ap.add_argument("--out", default="runs/st2")
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--lora", action="store_true", help="fine-tune bằng LoRA adapter (nhẹ, ít quên)")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--no_gc", action="store_true", help="tắt gradient checkpointing (nhanh hơn, tốn RAM hơn)")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="bf16 = nửa compute+RAM (MPS)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    proc = WhisperProcessor.from_pretrained(args.base, language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch_dtype)
    model.config.use_cache = False
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    gc_enabled = not args.no_gc
    if args.lora:
        from peft import LoraConfig, get_peft_model
        for p in model.parameters():
            p.requires_grad_(False)
        lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                        lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
        if gc_enabled:
            model.enable_input_require_grads()  # cần cho GC + base đóng băng

    train_ds = ManifestDS(args.train, proc)
    val_ds = ManifestDS(args.val, proc)
    print(f"train {len(train_ds)} | val {len(val_ds)}", flush=True)

    ta = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=0 if args.smoke else args.warmup,
        max_steps=30 if args.smoke else args.max_steps,
        gradient_checkpointing=gc_enabled,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False, bf16=False,          # MPS: fp32 an toàn (fp16 MPS bất ổn)
        eval_strategy="no" if args.smoke else "steps",
        eval_steps=args.save_steps,
        save_strategy="no" if args.smoke else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=5 if args.smoke else 25,
        per_device_eval_batch_size=args.bs,
        predict_with_generate=False,     # eval theo loss (nhanh); WER thật đo bằng verify_product.py
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        load_best_model_at_end=(not args.smoke),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )
    trainer = Seq2SeqTrainer(model=model, args=ta, train_dataset=train_ds,
                             eval_dataset=val_ds, data_collator=Collator(proc, torch_dtype),
                             processing_class=proc)
    t = time.time()
    trainer.train()
    dt = (time.time() - t) / 60
    print(f"TRAIN DONE in {dt:.1f}m ({30 if args.smoke else args.max_steps} steps)", flush=True)
    if not args.smoke:
        best = os.path.join(args.out, "best")
        trainer.save_model(best)
        proc.save_pretrained(best)
        print("saved ->", best, flush=True)


if __name__ == "__main__":
    main()
