"""单阶段 A1 专用类别 Token 训练入口。

最终方案直接从 ``model/v2`` 初始化，一次完成类别 Token 和边界学习：

* 冻结整个 Qwen3 底座；
* 新增 12 个 ``<|TN_CLASS|>`` special token；
* 只训练这 12 行 tied embedding/LM-head 权重；
* 对每个 Span 的最后两个 token 和类别 token 使用 2x CE 权重。

最终训练数据为 ``data/train_short.jsonl`` 与 ``data/train_augmented.jsonl``，
模型部署在 ``model/a1/``。
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from train.build_dataset import Collator, encode_example, load_jsonl
from train.label_tokens import add_label_tokens


class BoundaryTrainer(Trainer):
    """使用边界 token 权重的因果语言模型 Trainer。

    最终 A1 对边界 token 和类别 token 传入 2.0 权重。这是通用边界监督，
    不包含测试句子或局部规则。
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)
        if loss_weights is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # labels/logits 错一位；权重同步错一位，并按有效权重之和归一化。
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = inputs["labels"][:, 1:]
        shift_weights = loss_weights[:, 1:]
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(shift_labels.shape)
        loss = (token_loss * shift_weights).sum() / shift_weights.sum().clamp_min(1e-8)
        return (loss, outputs) if return_outputs else loss


def freeze_except_label_tokens(model, label_ids):
    """冻结底座，只给新增类别 token 行保留梯度。

    Qwen3 使用 tied embeddings，输入 embedding 和输出 LM head 指向同一参数。
    backward hook 会把所有旧词表行梯度清零，所以每一步只更新 12 行。
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    embedding = model.get_input_embeddings().weight
    output_embedding = model.get_output_embeddings().weight
    if embedding.data_ptr() != output_embedding.data_ptr():
        raise RuntimeError("A1 training requires tied input/output embeddings")

    embedding.requires_grad_(True)
    label_ids_cpu = torch.tensor(sorted(label_ids), dtype=torch.long)

    def mask_embedding_grad(grad):
        index = label_ids_cpu.to(grad.device)
        mask = torch.zeros_like(grad)
        mask[index] = 1
        return grad * mask

    embedding.register_hook(mask_embedding_grad)


def main():
    parser = argparse.ArgumentParser(description="A1 label-token head training")
    parser.add_argument("--data", nargs="+", required=True,
                        help="one or more replay/augmentation JSONL files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="model/v2")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--per-device", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="positive value runs an equal-step controlled pilot")
    parser.add_argument("--optim", default=(
        "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.boundary_weight < 1:
        raise ValueError("--boundary-weight must be >= 1")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = max(1, round(args.global_batch / (args.per_device * world_size)))

    tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=True)
    label_ids = add_label_tokens(tokenizer)

    records = [
        record
        for path in args.data
        for record in load_jsonl(path)
    ]
    random.Random(0).shuffle(records)
    val_size = int(len(records) * args.val_ratio)
    train_records = records[val_size:]
    val_records = records[:val_size]
    train_encoded = [
        encoded for record in train_records
        if (encoded := encode_example(
            record, tokenizer, boundary_weight=args.boundary_weight)) is not None
    ]
    val_encoded = [
        encoded for record in val_records
        if (encoded := encode_example(
            record, tokenizer, boundary_weight=args.boundary_weight)) is not None
    ]
    print(
        f"dataset: {len(records)} -> train {len(train_encoded)}, "
        f"val {len(val_encoded)}, boundary_weight={args.boundary_weight}",
        flush=True,
    )

    cap8 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if args.fp32:
        model_dtype, use_bf16, use_fp16 = torch.float32, False, False
    elif cap8:
        model_dtype, use_bf16, use_fp16 = torch.bfloat16, True, False
    else:
        model_dtype, use_bf16, use_fp16 = torch.float32, False, True

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=model_dtype, attn_implementation="sdpa")
    if tokenizer.vocab_size > model.get_input_embeddings().num_embeddings:
        raise ValueError("tokenizer has tokens missing from the model embedding")
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    freeze_except_label_tokens(model, label_ids.values())

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        # weight decay 作用于整个 embedding 参数，会污染冻结的旧词表行。
        weight_decay=0.0,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        per_device_train_batch_size=args.per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="epoch" if val_encoded else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=bool(val_encoded),
        metric_for_best_model="eval_loss" if val_encoded else None,
        greater_is_better=False if val_encoded else None,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        seed=42,
    )
    callbacks = (
        [EarlyStoppingCallback(early_stopping_patience=args.patience)]
        if val_encoded else []
    )
    trainer = BoundaryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded or None,
        data_collator=Collator(),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        final_model = trainer.model
        final_model.config.use_cache = True
        final_model.save_pretrained(f"{args.out}/final")
        tokenizer.save_pretrained(f"{args.out}/final")
        print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
