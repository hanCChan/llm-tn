"""合成训练数据构建器: 句子框架 × 值文法 → 标签由构造决定的训练样本(无蒸馏)。

从消融仓库(ceshi)的 05-form-coverage 移植, 路径适配本仓库。消融结论(见 data_synth/NOTES.md):
最终配方只用第 2 轮形态池 + 第 3 轮对比池 + 线上切片样本, 共三份合成文件。

两种数据(产物落 data_synth/output/):
    synth_form_coverage_<round>_<N>.jsonl     形态覆盖(D1): (类,子类,书写形态) 等概率采样
    synth_context_contrast_<round>_<N>.jsonl  语境对比(D2): 同一个外形歧义的值放进两个不同类的框架, 成对出现

框架两个来源(都不碰任何评测句):
    human  人工训练集 4167 条: 把句中每个 gold 片段挖成槽位, 按原值推断子类; 回填同子类的新形态值。
           语境 = 人工原句 → 不引入新的"关键词→类别"关联, 变的只有书写形态。
    agent  隔离写作的框架(data_synth/sentence_frames/), 补人工数据里没有的子类语境(时长、am/pm、国际电话…)。

每条样本过 4 道检查(任何一道不过就丢, 丢弃数写进审计):
    glue   值与框架邻字粘连(数字/字母紧贴、值后紧跟"点年月号元"等) → 边界不唯一;
    fst    整句按标签过 TNReader, 读音里残留阿拉伯数字 = 写法读不出来;
    dedup  与评测集共享 8 字以上片段(防用评测句造训练数据);
    dup    与已生成样本重复。

用法:
    python data_synth/build_synth_data.py --kind form --n 8334 --seed 11 --round 2
    python data_synth/build_synth_data.py --kind contrast --n 4168 --seed 13 --round 3
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from value_forms import FORMS, gen_value, product_glued, product_spaced          # noqa: E402

TRAIN_HUMAN = os.path.join(_ROOT, "data", "train_merged.jsonl")
OUT_DIR = os.path.join(_HERE, "output")
EVAL_GLOBS = ["data/test_data/*.jsonl", "data/test_data/gold_v2_serial/*.jsonl"]

# ---------------------------------------------------------------------------
# 人工片段的子类推断(只看值本身的写法)
# ---------------------------------------------------------------------------
_MEASURE_UNITS = [
    ("duration", r"小时|分钟|秒|天$|周$|个月"), ("age", r"岁"), ("temperature", r"℃|°C|°F|度$"),
    ("speed", r"/h|/s|公里/小时|米/秒|节$"), ("area", r"平方|㎡|m²|亩|公顷"),
    ("data_size", r"GB|MB|TB|KB"), ("electric", r"mAh|kW|W$|V$|瓦|伏"),
    ("volume", r"ml|mL|L$|升|立方|m³|加仑"), ("weight", r"kg|KG|g$|t$|斤|克|吨|磅"),
    ("length", r"km|cm|mm|m$|米|公里|英寸|英尺|海里"),
    ("count", r"[个名位辆栋台件次场家条只头张篇所架艘人]$"),
]


def infer_sub(cls, v):
    if cls == "PHONE":
        if re.fullmatch(r"(\+?86[- ]?)?1\d{2}[- ]?\d{4}[- ]?\d{4}", v):
            return "mobile"
        if v.startswith(("400", "800")) or len(re.sub(r"\D", "", v)) <= 5:
            return "hotline"
        if "转" in v or "ext" in v:
            return "extension"
        if v.startswith("+") or v.startswith("00"):
            return "international"
        return "landline"
    if cls == "TIME":
        if re.search(r"[至到~—\-]", v):
            return "range"
        if re.search(r"[ap]\.?m|[AP]\.?M", v):
            return "ampm"
        return "duration_like_clock" if v.count(":") == 2 else "clock"
    if cls == "DATE":
        if re.search(r"[至到~—]|\d-\d+年|日-", v):
            return "range"
        if re.fullmatch(r"\d{1,2}/\d{1,2}", v):
            return "slash"
        if re.fullmatch(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}", v):
            return "numeric"
        if "年代" in v or "世纪" in v:
            return "decade"
        if re.fullmatch(r"\d{2,4}年", v):
            return "year"
        if "年" in v:
            return "full"
        return "month_day"
    if cls == "MONEY":
        if re.search(r"\d\s*[至到~—\-]\s*\D?\d", v):
            return "range"
        if re.search(r"[万亿]", v):
            return "large"
        if re.search(r"[$€£]|美元|欧元|英镑|日元|港|USD|EUR", v):
            return "foreign"
        return "cny"
    if cls == "MEASURE":
        for sub, pat in _MEASURE_UNITS:
            if re.search(pat, v):
                return sub
        return None
    if cls == "PERCENT":
        return "range" if v.count("%") > 1 or re.search(r"[至到~—]", v) else "rate"
    if cls == "SCORE":
        return "match"
    if cls == "FRACTION":
        return "ratio"
    if cls == "MATH":
        return "algebra" if re.search(r"[a-z]", v.replace("x", "x")) and re.search(r"[a-wyz]|x[^\d]|x$", v) else "arith"
    if cls == "SERIAL":
        if re.match(r"[一-鿿][A-Z]", v):
            return "plate"
        return "model"
    if cls == "VERSION":
        return "software"
    if cls == "ADDRESS":
        return "street_no" if re.fullmatch(r"\d+号|\d+弄\d+号", v) else "building"
    return None


def locate(src, edits):
    out, cursor = [], 0
    for span, lab in edits:
        cls = lab.split(":")[0]
        i = src.find(span, cursor)
        if i < 0:
            return None
        out.append((i, i + len(span), cls))
        cursor = i + len(span)
    return out


# ---------------------------------------------------------------------------
# 框架库
# ---------------------------------------------------------------------------

def load_frames():
    """返回 {(cls, sub): [frame]}, frame = {"src", "slots": [(s, e, cls, sub, orig)], "target": i, "from"}"""
    bank = collections.defaultdict(list)
    for line in open(TRAIN_HUMAN, encoding="utf-8"):
        r = json.loads(line)
        if not r["edits"]:
            continue
        loc = locate(r["src"], r["edits"])
        if not loc:
            continue
        slots = [(s, e, c, infer_sub(c, r["src"][s:e]), r["src"][s:e]) for s, e, c in loc]
        for i, sl in enumerate(slots):
            if sl[3] and (sl[2], sl[3]) in FORMS:
                bank[(sl[2], sl[3])].append({"src": r["src"], "slots": slots, "target": i, "from": "human"})
    for p in sorted(glob.glob(os.path.join(_HERE, "sentence_frames", "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            f = json.loads(line)
            fr = f["frame"].replace("{P}", "{X}")
            if fr.count("{X}") != 1 or re.search(r"[0-9]", fr):
                continue
            s = fr.index("{X}")
            src = fr.replace("{X}", "")
            key = (f["cls"], f["sub"])
            if key in FORMS or f["cls"] == "PRODUCT":
                bank[key].append({"src": src, "slots": [(s, s, f["cls"], f["sub"], "")], "target": 0,
                                  "from": "agent"})
    return bank


_GLUE_RIGHT = set("点年月日号元个分秒%万亿岁度米克吨时楼室层栋")


def fill(frame, values):
    """values: {slot_idx: 值字符串 | [(片段, 类或None), ...]}; 其它槽位保留原值。
    列表形式用于枚举("12:00、18:00和21:00" = 三个 span + 两个分隔符)。返回 (src, edits) 或 None(粘连)。"""
    src, pos = frame["src"], 0
    text, edits = "", []
    for i, (s, e, c, sub, orig) in enumerate(frame["slots"]):
        v = values.get(i, orig)
        pieces = [(v, c)] if isinstance(v, str) else v
        seg = src[pos:s]
        if edits and seg:
            last = edits[-1][0]
            if (seg[0].isascii() and seg[0].isalnum()) or (last[-1].isdigit() and seg[0] in _GLUE_RIGHT):
                return None
        text += seg
        left = text[-1:]
        first = pieces[0][0]
        if left and left.isascii() and left.isalnum() and first[0].isascii() and first[0].isalnum():
            return None
        for t, pc in pieces:
            text += t
            if pc:
                edits.append([t, f"{pc}:{pc.lower()}"])
        pos = e
    tail = src[pos:]
    if tail and edits:
        last = edits[-1][0]
        if (tail[0].isascii() and tail[0].isalnum()) or (last[-1].isdigit() and tail[0] in _GLUE_RIGHT):
            return None
    return text + tail, edits


# ---------------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------------

def eval_ngrams(n=8):
    grams = set()
    for g in EVAL_GLOBS:
        for p in glob.glob(os.path.join(_ROOT, g)):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    s = json.loads(line)["src"]
                    grams.update(s[i:i + n] for i in range(len(s) - n + 1))
    return grams


class Checker:
    def __init__(self):
        from train.tn_fst import TNReader
        self.reader = TNReader()
        self.grams = eval_ngrams()
        self.seen = set()
        self.drop = collections.Counter()

    def ok(self, src, edits):
        if src in self.seen:
            self.drop["dup"] += 1
            return False
        if any(src[i:i + 8] in self.grams for i in range(len(src) - 7)):
            self.drop["dedup_eval"] += 1
            return False
        try:
            spoken = self.reader.apply(src, edits)
        except Exception:                                          # noqa: BLE001
            self.drop["fst_error"] += 1
            return False
        if edits and re.search(r"[0-9]", spoken):                 # 负例不查读音(不标的数字由默认规则读)
            self.drop["fst_digit_left"] += 1
            return False
        self.seen.add(src)
        return True


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

# 枚举(第 2 轮): dev 上"12:00、18:00、21:00"被并成一个 span —— 训练里从没见过同类值并列。
# 列表里每个值各自一个 span, 分隔符不进 span(LABEL_SPEC: 一个连续表达一个 span, 并列是多个表达)。
LISTABLE = {"TIME", "DATE", "MONEY", "PERCENT", "MEASURE", "PHONE", "SERIAL", "VERSION", "SCORE"}
LIST_SEPS = [("、", "和"), ("、", "、"), ("，", "，"), ("和", "和"), ("/", "/"), ("、", "或"), (" / ", " / ")]


def list_value(rng, c, s):
    k = rng.choice([2, 2, 3])
    sep_mid, sep_last = rng.choice(LIST_SEPS)
    pieces = []
    for j in range(k):
        if j:
            pieces.append((sep_last if j == k - 1 else sep_mid, None))
        pieces.append((gen_value(rng, c, s)[0], c))
    return pieces


def sample_form(rng, bank, chk, n, p_human=0.5, p_other=0.5, p_list=0.0):
    """D1: 类等概率 → 子类等概率 → 形态等概率; 框架 human/agent 各半(某侧缺则用另一侧)。"""
    classes = sorted({c for c, _ in FORMS})
    subs = collections.defaultdict(list)
    for (c, s) in FORMS:
        if bank.get((c, s)):
            subs[c].append(s)
    out = []
    while len(out) < n:
        c = rng.choice(classes)
        s = rng.choice(sorted(subs[c]))
        frames = bank[(c, s)]
        hum = [f for f in frames if f["from"] == "human"]
        agt = [f for f in frames if f["from"] == "agent"]
        pool = hum if (hum and (not agt or rng.random() < p_human)) else agt
        f = rng.choice(pool)
        if c in LISTABLE and rng.random() < p_list:
            v, form = list_value(rng, c, s), "list"
        else:
            v, form = gen_value(rng, c, s)
        vals = {f["target"]: v}
        for i, sl in enumerate(f["slots"]):                    # 人工多槽句: 其余槽位一半概率也换形态
            if i != f["target"] and sl[3] and (sl[2], sl[3]) in FORMS and rng.random() < p_other:
                vals[i] = gen_value(rng, sl[2], sl[3])[0]
        res = fill(f, vals)
        if res is None:
            chk.drop["glue"] += 1
            continue
        src, edits = res
        if not chk.ok(src, edits):
            continue
        out.append({"src": src, "edits": edits,
                    "meta": {"source": "synth_form_coverage", "cls": c, "sub": s, "form": form,
                             "frame": f["from"]}})
    return out


# 外形歧义族: 生成器 + 可放进去的 (类, 子类) 框架。值必须在每个候选类里都合理。
def _hm_ambig(r):
    return f"{r.randint(1, 23)}:{r.randint(0, 59):02d}"


def _slash_ambig(r):
    return f"{r.randint(1, 12)}/{r.randint(2, 28)}"


def _digits11(r):
    return r.choice(["13", "15", "18", "17", "19"]) + "".join(r.choice("0123456789") for _ in range(9))


def _dotdate(r):
    return f"{r.randint(2015, 2026)}.{r.randint(1, 12)}.{r.randint(1, 28)}"


def _small_colon(r):
    return f"{r.randint(0, 5)}:{r.randint(0, 5)}"


FAMILIES = {
    "a:b 时刻|比分": (_hm_ambig, [("TIME", "clock"), ("SCORE", "match"), ("SCORE", "vote")]),
    "a/b 日期|分数": (_slash_ambig, [("DATE", "slash"), ("FRACTION", "ratio")]),
    "11位数 电话|单号": (_digits11, [("PHONE", "mobile"), ("SERIAL", "order"), ("SERIAL", "account")]),
    "a.b.c 日期|版本": (_dotdate, [("DATE", "numeric"), ("VERSION", "software")]),
    "小a:b 比分|比例": (_small_colon, [("SCORE", "match"), ("MATH", "ratio_calc")]),
}


# 第 3 轮: "同外形、不同语义"成对族 —— 两个值外形相同(字母码+数字 / 数字+时间单位), 由语义定类。
# 成对出现, 两类各占一半, 不让任何一侧的先验因数据量膨胀。
def _cur_amount(r):
    from value_forms import CUR_CODE_V2, _amount
    return r.choice(CUR_CODE_V2) + r.choice(["", "", " "]) + _amount(r)


def _code_like(r):
    from value_forms import _letters
    return _letters(r, "ABCDEFGHJKLMNPRSTVXYZ", 3, 3) + str(r.randint(10, 99999))


def _duration_hms(r):
    return r.choice([f"{r.randint(1, 59)}分{r.randint(1, 59)}秒", f"{r.randint(1, 12)}小时{r.randint(1, 59)}分钟",
                     f"{r.randint(1, 12)}小时{r.randint(1, 59)}分", f"{r.randint(1, 59)}分钟{r.randint(1, 59)}秒"])


def _clock_hm(r):
    return r.choice([f"{r.randint(0, 23)}点{r.randint(1, 59)}分", f"{r.randint(0, 23)}时{r.randint(0, 59)}分",
                     f"{r.randint(0, 23)}时{r.randint(0, 59)}分{r.randint(0, 59)}秒", f"{r.randint(1, 12)}点半"])


SHAPE_PAIRS = {
    "字母码+数字 币种→MONEY|编号→SERIAL": [(("MONEY", "foreign"), _cur_amount), (("SERIAL", "model"), _code_like)],
    "数字+时间单位 时长→MEASURE|时刻→TIME": [(("MEASURE", "duration"), _duration_hms), (("TIME", "clock"), _clock_hm)],
}


def sample_contrast(rng, bank, chk, n, product=False, round_=1):
    """D2: 抽一个歧义族 → 生成一个值 → 放进该族两个不同类别的框架, 两句都要通过检查才收。
    product=True(第 2 轮)再加一族: 产品框架里"品牌 空格 数字"(负例, 不标) vs "字母数字紧连"(SERIAL)。
    round_>=3: 产品族只留负例半边(紧连半边让 SERIAL 先验膨胀, test373 上 AUD100→SERIAL);
              新增两个"同外形不同语义"成对族(见 SHAPE_PAIRS)。"""
    out = []
    fams = sorted(FAMILIES) + (["产品名 空格不标|紧连SERIAL"] if product and bank.get(("PRODUCT", "name")) else [])
    if round_ >= 3:
        fams += sorted(SHAPE_PAIRS)
    while len(out) < n:
        fam = rng.choice(fams)
        if fam in SHAPE_PAIRS:
            pair = []
            for cell, gen in SHAPE_PAIRS[fam]:
                f = rng.choice(bank[cell])
                res = fill(f, {f["target"]: gen(rng)})
                if res is None:
                    chk.drop["glue"] += 1
                    break
                pair.append((cell, f, res))
            if len(pair) == 2 and all(chk.ok(*p[2]) for p in pair):
                for cell, f, (src, edits) in pair:
                    out.append({"src": src, "edits": edits,
                                "meta": {"source": "synth_context_contrast", "family": fam, "cls": cell[0],
                                         "sub": cell[1], "frame": f["from"]}})
            continue
        if fam.startswith("产品名"):
            pair = []
            members = ((product_spaced(rng), None),) if round_ >= 3 else \
                ((product_spaced(rng), None), (product_glued(rng), "SERIAL"))
            for v, lab in members:
                f = rng.choice(bank[("PRODUCT", "name")])
                res = fill(f, {0: [(v, lab)]})
                if res is None:
                    chk.drop["glue"] += 1
                    break
                pair.append((("NEG" if lab is None else "SERIAL", "product"), f, res))
            if len(pair) == len(members) and all(chk.ok(*p[2]) for p in pair):
                for cell, f, (src, edits) in pair:
                    out.append({"src": src, "edits": edits,
                                "meta": {"source": "synth_context_contrast", "family": fam, "cls": cell[0],
                                         "sub": cell[1], "frame": f["from"]}})
            continue
        gen, cells = FAMILIES[fam]
        cells = [c for c in cells if bank.get(c)]
        a, b = rng.sample(cells, 2)
        if a[0] == b[0]:
            continue
        v = gen(rng)
        pair = []
        for cell in (a, b):
            f = rng.choice(bank[cell])
            res = fill(f, {f["target"]: v})
            if res is None:
                chk.drop["glue"] += 1
                break
            pair.append((cell, f, res))
        if len(pair) < 2 or not all(chk.ok(*p[2]) for p in pair):
            continue
        for cell, f, (src, edits) in pair:
            out.append({"src": src, "edits": edits,
                        "meta": {"source": "synth_context_contrast", "family": fam, "cls": cell[0],
                                 "sub": cell[1], "value": v, "frame": f["from"]}})
    return out[:n]


def audit(rows, chk, path):
    lines = [f"样本 {len(rows)}; 丢弃 {dict(chk.drop)}"]
    by = collections.Counter((r["meta"]["cls"], r["meta"].get("sub")) for r in rows)
    cls = collections.Counter(r["meta"]["cls"] for r in rows)
    fr = collections.Counter(r["meta"]["frame"] for r in rows)
    lines.append(f"框架来源 {dict(fr)}")
    lines.append("类别 " + " ".join(f"{k}:{v}" for k, v in sorted(cls.items())))
    lines += [f"  {c}/{s}: {v}" for (c, s), v in sorted(by.items())]
    if rows and "form" in rows[0]["meta"]:
        forms = collections.Counter((r["meta"]["cls"], r["meta"]["form"]) for r in rows)
        lines.append(f"形态覆盖 {len(forms)} 种(文法共 {sum(len(v) for v in FORMS.values())} 种)")
    if rows and "family" in rows[0]["meta"]:
        lines.append("歧义族 " + str(dict(collections.Counter(r["meta"]["family"] for r in rows))))
    lines.append("样例:")
    lines += ["  " + r["src"] + "  " + json.dumps(r["edits"], ensure_ascii=False) for r in rows[:25]]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:6 + len(by)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["form", "contrast"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--round", type=int, default=1, help="2 = 第 2 轮(枚举 12% + 产品名对比族)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bank = load_frames()
    chk = Checker()
    if args.kind == "form":
        rows = sample_form(rng, bank, chk, args.n, p_list=0.12 if args.round >= 2 else 0.0)
    else:
        rows = sample_contrast(rng, bank, chk, args.n, product=args.round >= 2, round_=args.round)
    rng.shuffle(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    name = {"form": "synth_form_coverage", "contrast": "synth_context_contrast"}[args.kind]
    if args.round >= 2:
        name += f"_round{args.round}"
    path = os.path.join(OUT_DIR, f"{name}_{args.n}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.makedirs(os.path.join(_HERE, "out"), exist_ok=True)
    audit(rows, chk, os.path.join(_HERE, "out", f"audit_{name}_{args.n}.txt"))
    print(f"写出 {path}")


if __name__ == "__main__":
    main()
