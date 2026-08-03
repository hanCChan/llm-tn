"""Qwen3-0.6B-Base 全参 SFT。
conda activate qwen3_tn
find data/train_data -type f -name "*.jsonl" -exec cat {} + > data/train_merged.jsonl
python -m train.sft --data data/train_merged.jsonl --out out/sft/v3 --fp32
"""

import argparse
import random

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from train.constants import MODEL_ID
from train.build_dataset import Collator, encode_example, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--per-device", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="纯 fp32 训练(最稳,T4 上慢但无 AMP 隐患)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    recs = load_jsonl(args.data)
    random.Random(0).shuffle(recs)
    encoded = [e for r in recs if (e := encode_example(r, tok)) is not None]
    print(f"dataset: {len(recs)} recs -> {len(encoded)} encoded "
          f"(dropped {len(recs) - len(encoded)} overlong)", flush=True)

    # 精度:--fp32 强制纯 fp32(最稳);否则 Ampere+ 用 bf16,Turing(T4)用 fp16-AMP。
    # fp16-AMP 需权重以 fp32 加载(autocast + GradScaler),否则 unscale 报错。
    cap8 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if args.fp32:
        model_dtype, use_bf16, use_fp16, prec = torch.float32, False, False, "fp32"
    elif cap8:
        model_dtype, use_bf16, use_fp16, prec = torch.bfloat16, True, False, "bf16"
    else:
        model_dtype, use_bf16, use_fp16, prec = torch.float32, False, True, "fp16-amp(fp32 weights)"
    print(f"precision: {prec}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=model_dtype, attn_implementation="sdpa")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.per_device,
        gradient_accumulation_steps=max(1, args.global_batch // args.per_device),
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=encoded,
        data_collator=Collator(),
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(f"{args.out}/final")
    tok.save_pretrained(f"{args.out}/final")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
