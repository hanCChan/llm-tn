"""NAR 离线评测: 与 v2 同口径的端到端管线(标注 -> TNReader 读音), 不起服务。

    python test/eval_nar.py --model runs/nar_human
    python test/eval_nar.py --model runs/nar_human --data data/test_data/hard_num.jsonl

指标与 test/eval_e2e.py 同义(v2 的对照评测), 便于直接对比:
    1) Tag:   span 级 micro P/R/F1、按类别细分、句级完全匹配、负例未误标;
    2) 文本:  端到端口语输出逐字 == 人工 gold_spoken。

评测口径(为什么逐句 batch=1): 合批的 bf16 抖动会让 ~0.3% 的标注翻转,
质量数字不应受它影响 —— 这是 NAR 评测的既定结论; 并发性能另行压测。
"""
import argparse
import collections
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train.nar import NARRunner                     # noqa: E402
from train.tn_fst import TNReader                   # noqa: E402

DEFAULT_DATA = ["data/test_data/polynorm_eval.jsonl", "data/test_data/hard_num.jsonl"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def eval_file(nar, reader, path, dump=None, idx0=0):
    """跑一个数据集, 返回统计 dict; 逐句前向(质量口径)。"""
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    tp = fp = fn = 0
    per = collections.defaultdict(lambda: [0, 0, 0])   # label -> [tp, fp, fn]
    tag_exact = neg_total = neg_ok = txt_ok = 0

    for k, r in enumerate(recs):
        src = r["src"]
        gold = {tuple(e) for e in r["edits"]}
        edits = nar.tag([src])[0]                       # 单句前向
        pred = {tuple(e) for e in edits}

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

        pred_spoken = reader.apply(src, edits)
        gs = r.get("gold_spoken")
        ok = gs is not None and pred_spoken == gs
        txt_ok += ok
        if dump:
            dump.write(json.dumps({"idx": idx0 + k, "src": src, "gold_edits": r["edits"],
                                   "pred_edits": edits, "gold_spoken": gs,
                                   "pred_spoken": pred_spoken, "tag_ok": pred == gold,
                                   "txt_ok": ok}, ensure_ascii=False) + "\n")

    n = len(recs)
    P, R, F = prf(tp, fp, fn)
    print(f"\n数据集: {path}   样本: {n}")
    print(f"  Tag micro P={P:.3f} R={R:.3f} F1={F:.3f} (TP={tp} FP={fp} FN={fn})"
          f"  句级完全正确 {tag_exact}/{n}"
          + (f"  负例未误标 {neg_ok}/{neg_total}" if neg_total else ""))
    for lab in sorted(per):
        t, f_, n_ = per[lab]
        p, rr, f1 = prf(t, f_, n_)
        print(f"    {lab:18s} P={p:.3f} R={rr:.3f} F1={f1:.3f} ({t}/{f_}/{n_})")
    if any(r.get("gold_spoken") is not None for r in recs):
        print(f"  文本逐字正确: {txt_ok}/{n}")
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tag_exact": tag_exact,
            "txt_ok": txt_ok, "path": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/nar_human")
    ap.add_argument("--data", nargs="*", default=DEFAULT_DATA)
    ap.add_argument("--dump", help="逐条 jsonl dump(供与 v2 的 dump 逐条 diff)")
    args = ap.parse_args()

    nar = NARRunner(args.model)
    reader = TNReader()
    dump = None
    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
        dump = open(args.dump, "w", encoding="utf-8")
    stats, tot = [], collections.Counter()
    for p in args.data:
        st = eval_file(nar, reader, p, dump)
        stats.append(st)
        for k in ("n", "tp", "fp", "fn", "tag_exact", "txt_ok"):
            tot[k] += st[k]
    if dump:
        dump.close()
    if len(stats) > 1:
        n = tot["n"]
        P, R, F = prf(tot["tp"], tot["fp"], tot["fn"])
        print(f"\n汇总 {n} 条: TagF1={F:.3f} (P={P:.3f} R={R:.3f}) 句级 {tot['tag_exact']}/{n}"
              f"  文本正确 {tot['txt_ok']}/{n}")


if __name__ == "__main__":
    main()
