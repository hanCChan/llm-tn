"""构造 A1 专用类别 token 训练序列。

最终 A1 训练序列
================
v2 completion::

    span<TAB>CLASS:ctx
    span<TAB>CLASS:ctx<EOS>

A1 completion::

    span<|TN_CLASS|>span<|TN_CLASS|><EOS>

prompt 与 v2 相同：``<|fim_prefix|>原文<|fim_suffix|>``。A1 不再使用 TAB、
换行和 ``:ctx``，类别 special token 同时承担字段边界与类别语义。

边界加权
========
A1 对每条 edit 的最后两个普通 token 和类别 token 设置 ``boundary_weight``。
类别 token 提前生成会把句点等收尾字符留在 token 之后，这是专用 token 协议
最常见的 Spoken 回退来源。最终方案统一使用边界权重 2。
"""

import json

from train.constants import EOS_ID, SEP, TN_PREFIX
from train.label_tokens import LABEL_TOKENS, render_label_token_output


MAX_LEN = 512


def _build_boundary_weights(completion_ids: list[int], prompt_len: int,
                            tokenizer, boundary_weight: float) -> list[float]:
    """构造与 labels 对齐的 token 权重。

    prompt 权重为 0；completion 普通 token 权重为 1；每个类别 token 及其前
    最多两个 span 内 token 权重为 ``boundary_weight``。若前一个位置已经是
    上一条 edit 的类别 token，则停止向前扩散，避免跨 span 污染。
    """
    weights = [0.0] * prompt_len + [1.0] * len(completion_ids)
    label_ids = {
        tokenizer.convert_tokens_to_ids(token)
        for token in LABEL_TOKENS.values()
    }
    for token_index, token_id in enumerate(completion_ids):
        if token_id not in label_ids:
            continue
        weights[prompt_len + token_index] = boundary_weight
        for offset in (1, 2):
            previous = token_index - offset
            if previous < 0 or completion_ids[previous] in label_ids:
                break
            weights[prompt_len + previous] = max(
                weights[prompt_len + previous], boundary_weight)
    return weights


def encode_example(rec: dict, tok, boundary_weight: float = 2.0) -> dict | None:
    """把一条原始记录编码为 A1 训练样本。

    超过 512 token 的样本返回 None。
    """
    prompt = TN_PREFIX + rec["src"] + SEP
    completion = render_label_token_output(rec["edits"])
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tok(
        completion, add_special_tokens=False).input_ids + [EOS_ID]
    if len(prompt_ids) + len(completion_ids) > MAX_LEN:
        return None

    result = {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }
    if boundary_weight > 1:
        result["loss_weights"] = _build_boundary_weights(
            completion_ids, len(prompt_ids), tok, boundary_weight)
    return result


def load_jsonl(path: str) -> list[dict]:
    """读取非空 JSONL，兼容文件 BOM。"""
    records = []
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    return records


class Collator:
    """右 padding 动态组 batch。

    input_ids 用 EOS id padding；labels 和边界权重 padding 分别为 -100、0；
    attention mask 对 padding 置 0。
    """

    def __init__(self, pad_id: int = EOS_ID):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        loss_weights = []
        has_weights = "loss_weights" in batch[0]
        for item in batch:
            length = len(item["input_ids"])
            pad = max_length - length
            input_ids.append(item["input_ids"] + [self.pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * length + [0] * pad)
            if has_weights:
                loss_weights.append(item["loss_weights"] + [0.0] * pad)

        result = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }
        if has_weights:
            result["loss_weights"] = torch.tensor(loss_weights)
        return result
