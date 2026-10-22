"""片段化训练样本: 用线上同一个切片器(train/split.py::split_chunks)把训练句切成小片。

为什么(07 第 3 轮错误分析): 服务端先按标点切片再送模型, 线上输入大量是"数字打头"的片;
训练句却几乎都是完整句(人工 95/4167 条 span 在句首), 模型在片首会丢掉首字母
(JP¥10000 → P¥10000, AUD100 → UD100)。这是训练/线上输入分布不一致, 用线上切片器补齐。

做法: 对人工训练集与合成形态池的句子跑 split_chunks, 只保留含 span 的片(span 必然完整
落在片内, 是 split_chunks 的不变式); 人工、合成各取一半。标签 = 原标签平移到片内坐标。
另做"片首对齐"(crop_left): 自带类别线索的值, 一半概率裁掉左侧上下文 —— 教位置不变性。

用法: python data_synth/make_chunk_samples.py --n 1400 --seed 21
产物: data_synth/output/synth_chunked_from_human_and_form_<N>.jsonl
"""
import argparse
import json
import os
import random
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from train.split import split_chunks          # noqa: E402

T = os.path.dirname(os.path.abspath(__file__))
SOURCES = {"human": os.path.join(T, "..", "data", "train_merged.jsonl"),
           "form": os.path.join(T, "output", "synth_form_coverage_round2_8334.jsonl")}


def crop_left(c, rng, p=0.5):
    """片首对齐: 首个 span 自带类别线索(币种/单位/%/字母/=)时, 一半概率裁掉它左边的上下文。
    纯数字+分隔符的歧义外形(3:1、2/3、13800138000)不裁 —— 裁掉左边线索后类别会变得不可判。"""
    sp = c["edits"][0][0]
    i = c["src"].find(sp)
    if i <= 0 or not re.search(r"[^\d.,:/\-\s]", sp) or rng.random() >= p:
        return c
    return dict(c, src=c["src"][i:], first_at_start=True)


def chunks_of(r):
    src, cursor, spans = r["src"], 0, []
    for span, lab in r["edits"]:
        i = src.find(span, cursor)
        if i < 0:
            return []
        spans.append((i, i + len(span), span, lab))
        cursor = i + len(span)
    out, pos = [], 0
    for ch in split_chunks(src):
        a, b = pos, pos + len(ch)
        pos = b
        inside = [(s - a, e - a, sp, lab) for s, e, sp, lab in spans if a <= s and e <= b]
        if not inside or ch.strip() == src.strip():
            continue                                   # 只要真正被切出来、且含 span 的片
        lead = len(ch) - len(ch.lstrip())
        out.append({"src": ch.strip(), "edits": [[sp, lab] for _, _, sp, lab in inside],
                    "first_at_start": inside[0][0] - lead == 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for name, fn in SOURCES.items():
        pool = []
        for line in open(os.path.join(T, fn) if not fn.startswith("..") else os.path.normpath(os.path.join(T, fn)),
                         encoding="utf-8"):
            r = json.loads(line)
            for c in chunks_of(r):
                c = crop_left(c, rng)
                pool.append({"src": c["src"], "edits": c["edits"],
                             "meta": {"source": "synth_chunked", "from": name, "span_at_start": c["first_at_start"]}})
        rng.shuffle(pool)
        seen, take = set(), []
        for c in pool:
            if c["src"] not in seen:
                seen.add(c["src"])
                take.append(c)
            if len(take) == args.n // 2:
                break
        print(f"{name}: 可切片 {len(pool)}, 取 {len(take)}, 其中 span 在片首 "
              f"{sum(c['meta']['span_at_start'] for c in take)}")
        rows += take
    rng.shuffle(rows)
    path = os.path.join(T, "output", f"synth_chunked_from_human_and_form_{len(rows)}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("写出", path)
    for r in rows[:8]:
        print("  ", r["src"], r["edits"])


if __name__ == "__main__":
    main()
