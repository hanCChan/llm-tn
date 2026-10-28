# =============================================================================
# llm-tn source header
# Path: test/eval_e2e.py
# 职责: 端到端评估: Tag 准确率 + 绝对文本(口语)正确率
# 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
# =============================================================================

"""端到端评估: Tag 准确率 + 绝对文本(口语)正确率。

    python test/eval_e2e.py                              # 默认跑两个测试集
    python test/eval_e2e.py --data data/test_data/hard_num.jsonl
    python test/eval_e2e.py --model model/v2 --dump test_output/e2e.dump.txt

两个指标(均基于真实管线 main.TN: LLM 标注 -> TNReader 两遍读音):

  1) Tag 准确率  —— 预测 edits vs gold edits(与 reader 无关)
     · span 级 micro P/R/F1 + 按类别细分
     · 句级完全匹配率(预测 edits 集合 == gold 集合)
     · 负例未误标率(gold 为空且预测也为空)

  2) 绝对文本正确率 —— 端到端(tag+reader)口语 vs 人工标准口语 gold_spoken
     · 逐字精确匹配(reader 已关全/半角转换, 标点原样保留)
     · gold_spoken 为人工标注的"绝对正确读音", 独立于 reader; 测的是整条
       管线(定类 + 读音)最终读对与否。

数据需含字段: src, edits, gold_spoken。
"""
import argparse
import collections
import json
import os
import sys

# 让脚本能 import 到 main / train.*(仓库根目录)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main import TN  # noqa: E402

DEFAULT_DATA = [
    "data/test_data/polynorm_eval.jsonl",
    "data/test_data/hard_num.jsonl",
]


def prf(tp, fp, fn):
    """由 TP/FP/FN 计算 precision、recall 和 F1；分母为零时返回 0。"""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def eval_file(tn, path, dump=None):
    """跑一个数据集, 返回统计 dict。"""
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    tp = fp = fn = 0
    per = collections.defaultdict(lambda: [0, 0, 0])   # label -> [tp, fp, fn]
    tag_exact = 0                                       # tag 句级完全正确
    neg_total = neg_ok = 0                              # 负例
    txt_ok = 0                                          # 绝对文本逐字匹配

    for r in recs:
        src = r["src"]
        gold = {(s, l) for s, l in r["edits"]}
        pred_edits = tn.tag(src)
        pred = {(s, l) for s, l in pred_edits}

        # --- Tag ---
        if pred == gold:
            tag_exact += 1
        if not gold:
            neg_total += 1
            neg_ok += (len(pred) == 0)
        for x in pred & gold:
            tp += 1; per[x[1]][0] += 1
        for x in pred - gold:
            fp += 1; per[x[1]][1] += 1
        for x in gold - pred:
            fn += 1; per[x[1]][2] += 1

        # --- 绝对文本(端到端读音 vs 人工 gold_spoken, 逐字精确) ---
        pred_spoken = tn.reader.apply(src, pred_edits)
        gold_spoken = r["gold_spoken"]
        ok = pred_spoken == gold_spoken
        txt_ok += ok

        if dump:
            dump.write(
                f"IN  : {src}\n"
                f"GOLD_TAG: {sorted(gold)}\n"
                f"PRED_TAG: {sorted(pred)}\n"
                f"GOLD_TXT: {gold_spoken}\n"
                f"PRED_TXT: {pred_spoken}\n"
                f"{'TAG_OK' if pred == gold else 'TAG_DIFF'}\n\n"
                f"{'TXT_OK' if ok else 'TXT_DIFF'}\n\n"
            )

    return {
        "path": path, "n": len(recs),
        "tp": tp, "fp": fp, "fn": fn, "per": per,
        "tag_exact": tag_exact, "neg_total": neg_total, "neg_ok": neg_ok,
        "txt_ok": txt_ok,
    }


def report(st):
    """把一个数据集的统计字典格式化为 Tag 与绝对文本指标。"""
    n = st["n"]
    P, R, F = prf(st["tp"], st["fp"], st["fn"])
    print(f"\n数据集: {st['path']}   样本: {n}")
    print("=" * 60)
    print("【Tag 准确率】")
    print(f"  micro  P={P:.3f}  R={R:.3f}  F1={F:.3f}   (TP={st['tp']} FP={st['fp']} FN={st['fn']})")
    print(f"  句级完全正确: {st['tag_exact']}/{n} = {st['tag_exact']/n:.3f}")
    if st["neg_total"]:
        print(f"  负例未误标 : {st['neg_ok']}/{st['neg_total']}")
    print("  按类别:")
    print(f"    {'label':16s} {'P':>6} {'R':>6} {'F1':>6}  (TP/FP/FN)")
    for lab in sorted(st["per"]):
        t, f, nn = st["per"][lab]
        p, rr, f1 = prf(t, f, nn)
        print(f"    {lab:16s} {p:6.3f} {rr:6.3f} {f1:6.3f}  ({t}/{f}/{nn})")
    print("-" * 60)
    print("【绝对文本正确率】(端到端读音 vs 人工 gold_spoken, 逐字精确)")
    print(f"  逐字完全正确: {st['txt_ok']}/{n} = {st['txt_ok']/n:.3f}")


def main():
    """解析评估参数，逐数据集运行真实 TN 管线并输出汇总指标。

    原理: ``TN.tag`` 单独评估类别边界，随后把预测 edits 交给 ``TNReader``
    与人工 ``gold_spoken`` 做逐字比较，从而分别暴露定类错误和读音错误。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/v2")
    ap.add_argument("--data", nargs="*", default=DEFAULT_DATA,
                    help="一个或多个 jsonl(需含 src+edits+gold_spoken); 默认跑内置两个测试集")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--dump", help="把每条 IN/GOLD/PRED(tag+文本) 写到该文件")
    args = ap.parse_args()

    tn = TN(model=args.model, max_new_tokens=args.max_new)
    dump = open(args.dump, "w", encoding="utf-8") if args.dump else None

    stats = [eval_file(tn, p, dump) for p in args.data]
    if dump:
        dump.close()

    for st in stats:
        report(st)

    if len(stats) > 1:                                  # 汇总
        n = sum(s["n"] for s in stats)
        tp = sum(s["tp"] for s in stats)
        fp = sum(s["fp"] for s in stats)
        fn = sum(s["fn"] for s in stats)
        tag_exact = sum(s["tag_exact"] for s in stats)
        txt_ok = sum(s["txt_ok"] for s in stats)
        P, R, F = prf(tp, fp, fn)
        print(f"\n{'#'*60}\n汇总(全部 {len(stats)} 个数据集, 共 {n} 条)")
        print(f"  Tag  micro F1={F:.3f} (P={P:.3f} R={R:.3f})  句级完全正确={tag_exact}/{n}={tag_exact/n:.3f}")
        print(f"  绝对文本 逐字完全正确={txt_ok}/{n}={txt_ok/n:.3f}")


if __name__ == "__main__":
    main()
