"""切分器不变式验证: 拿 data/train_merged.jsonl 的真实 gold 标注跑。

    python test/test_split.py

两条不变式:
    1) 无损:     "".join(split_chunks(t)) == t
    2) 不断 span: 每个 gold span 在原文中的区间完整落在某一个片内
       —— 这条直接证明"没切断数字", 比人眼看例子可靠。

除单条样本外, 还把多条样本拼成几百字段落再验一遍 —— 长输入才是切分器的实际战场,
也只有拼接后才会出现单句里不存在的标点组合。
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.split import has_digit, split_chunks   # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "train_merged.jsonl")


def _spans(src, edits):
    """gold edits -> [(start, end)]; 按出现顺序从左往右定位, 与 _locate 同构。"""
    out, pos = [], 0
    for span, _label in edits:
        idx = src.find(span, pos)
        if idx < 0:                                # span 抄错/不在原文, 跳过
            continue
        out.append((idx, idx + len(span)))
        pos = idx + len(span)
    return out


def _bounds(chunks):
    """片 -> [(start, end)] 字符区间。"""
    out, pos = [], 0
    for c in chunks:
        out.append((pos, pos + len(c)))
        pos += len(c)
    return out


def check(src, edits, fails):
    chunks = split_chunks(src)

    if "".join(chunks) != src:
        fails.append(("lossy", src, None))
        return chunks

    bounds = _bounds(chunks)
    for s, e in _spans(src, edits):
        if not any(bs <= s and e <= be for bs, be in bounds):
            crossed = [c for (bs, be), c in zip(bounds, chunks) if bs < e and s < be]
            fails.append(("split-span", src[s:e], crossed))
    return chunks


def main():
    rows = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    print(f"样本 {len(rows)} 条")

    fails, n_chunks, n_digit = [], 0, 0
    for d in rows:
        chunks = check(d["src"], d["edits"], fails)
        n_chunks += len(chunks)
        n_digit += sum(1 for c in chunks if has_digit(c))

    print(f"单条: 片数 {n_chunks} (均 {n_chunks / len(rows):.1f}/条), "
          f"含数字 {n_digit} ({n_digit / n_chunks:.0%}) -> 需过 LLM")

    # 拼成长段落再验: 3~8 条随机拼接, 逼近真实的几百字请求。
    rng = random.Random(0)
    long_fails, tot_chars, tot_chunks, tot_digit = [], 0, 0, 0
    for _ in range(500):
        picked = [rows[rng.randrange(len(rows))] for _ in range(rng.randint(3, 8))]
        src = "".join(p["src"] for p in picked)
        edits, off = [], 0
        for p in picked:                           # 拼接后 span 仍在, 顺序不变
            edits.extend(p["edits"])
            off += len(p["src"])
        chunks = check(src, edits, long_fails)
        tot_chars += len(src)
        tot_chunks += len(chunks)
        tot_digit += sum(1 for c in chunks if has_digit(c))

    print(f"长段落: 500 段, 均 {tot_chars / 500:.0f} 字 -> 均 {tot_chunks / 500:.1f} 片, "
          f"含数字 {tot_digit / tot_chunks:.0%} (可跳过 {1 - tot_digit / tot_chunks:.0%})")

    fails += long_fails
    if fails:
        print(f"\n失败 {len(fails)} 例:")
        for kind, what, ctx in fails[:20]:
            print(f"  [{kind}] {what!r}" + (f" 被切成 {ctx!r}" if ctx else ""))
        return 1
    print("\n两条不变式全部通过: 无损 + 无 span 被切断")
    return 0


if __name__ == "__main__":
    sys.exit(main())
