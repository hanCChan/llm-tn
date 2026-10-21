"""非自回归(NAR)定类: v2 主干 + 逐字 25 类 BIO 分类头, 一次前向出全部 span。

任务重构(相对 v2 的自回归生成):
    v2 要把 span 逐 token 抄出来再接 "\tCLASS:ctx", 延迟 ≈ 每 token ~3ms × 输出长度,
    span 多的长句正是 p99 的来源。span 本来就在输入里, 模型只需判断"哪几个字、属于
    哪类" —— 改成逐字分类后只剩一次前向, 延迟与 span 数量/长度无关。

四个关键设计(消融实验已逐项验证收益, 见 feature 分支 METRICS 说明):
    1) 25 类 BIO 头: 12 个业务类各有 B/I 标签 + O, 共 25 类; 复用 v2 主干权重,
       分类头随机初始化(小方差), 训练时头 lr ×10(从零学的参数要更快收敛)。
    2) Echo 复读输入: "<tn> 原文 <sep> 原文", 只监督第二份。Qwen 是因果注意力,
       单份原文时第 i 个字看不到右边(3:1获胜 必须看到"获胜"才知道是比分);
       复读后第二份每个字都能看到第一份完整全文, 等效双向上下文。
    3) CharTokenizer 逐字分词: BPE 会把 span 边界两侧的字符粘成一个 token
       (训练集 4.4% 的 span 骑界, 如 '%，' 一半在 span 里一半在外, 标签无解);
       逐字一个 token 后字符与标签一一对应, 骑界从定义上消失。
    4) 两步 span 解码(decode_offsets) + 结构修复(repair_offsets):
       先按 P(非O)>P(O) 切边界, 再对 span 内各字符的类别概率求和整段定一次类,
       避免逐字 argmax 时同一 span 内类别跳变; 之后按配对/前缀等结构规则修正
       括号、币种代码等确定性错误 —— 二者都不改模型概率, 不重训即可生效。

训练(两卡):
    torchrun --nproc_per_node 2 -m train.nar --data data/train_merged.jsonl:2 \
        --init model/v2 --out runs/nar_human
评测(单卡):
    python test/eval_nar.py --model runs/nar_human
推理(单卡):
    r = NARRunner("runs/nar_human"); r.tag(["比分3:1获胜"])
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train.constants import SEP_ID, TN_PREFIX_ID   # noqa: E402

# 12 个业务类(顺序与 v2 的标签约定一致); 标签表: O 在前, 每类 B/I 相邻排列,
# decode_offsets 里用切片 [1::2]/[2::2] 取 B/I 列。
CLASSES = ["DATE", "TIME", "MONEY", "MEASURE", "FRACTION", "MATH",
           "SCORE", "PERCENT", "PHONE", "SERIAL", "VERSION", "ADDRESS"]
LABELS = ["O"] + [f"{p}-{c}" for c in CLASSES for p in ("B", "I")]
N_LABELS = len(LABELS)
_CLS_INDEX = {c: i for i, c in enumerate(CLASSES)}
_CLS_LABEL = {i: f"{c}:{c.lower()}" for i, c in enumerate(CLASSES)}

MAX_CHARS = 254          # 单份原文上限; 复读后总长 2n+2 <= 510


# ---------------------------------------------------------------------------
# 分词 / 样本构造
# ---------------------------------------------------------------------------

class CharTokenizer:
    """逐字分词: 每个字符恰好一个 token, 保证 BIO 标签与字符一一对应。

    字 -> id 走 Qwen 的 BPE 词表: 常用字本身就是单 token; 罕见字(龘、扩展区)会被
    BPE 拆成多个 byte token, 这里取第一个 —— 损失一点罕见字的区分度, 换来严格的
    一字一位。结果按字缓存, 线上建输入只剩查 dict。
    """

    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
        self._cache = {}

    def _id(self, ch):
        i = self._cache.get(ch)
        if i is None:
            ids = self.tok.encode(ch, add_special_tokens=False)
            i = ids[0] if ids else SEP_ID
            self._cache[ch] = i
        return i

    def encode(self, text):
        cache = self._cache
        return [cache[ch] if ch in cache else self._id(ch) for ch in text]

    def save(self, out):
        # 词表必须随权重走: 评测端只拿模型目录, 不存的话 fast->slow 回退会崩。
        self.tok.save_pretrained(out)


def echo_ids(char_ids):
    """<tn> 原文 <sep> 原文。返回 (ids, 标签起点) —— 标签起点 = 第二份原文的起始下标。

    第一份是"参考书"(提供完整全文, 不监督), 第二份是"答题卡"(被监督)。len = 2n+2。
    """
    n = len(char_ids)
    return [TN_PREFIX_ID] + char_ids + [SEP_ID] + char_ids, n + 2


def locate(src, edits):
    """edits 按出现顺序定位成不重叠的 [(start, end, CLASS)]; 与 TNReader._locate 同
    规则(训练标签与线上读音对 span 位置的理解必须一致)。找不到/重叠/未知类的丢弃。"""
    out, cursor = [], 0
    for item in edits:
        if not item or not item[0]:
            continue
        span, label = item[0], item[1]
        cls = label.split(":", 1)[0].strip().upper()
        if cls not in _CLS_INDEX:
            continue
        idx = src.find(span, cursor)
        if idx < 0:
            idx = src.find(span)
        if idx < 0:
            continue
        out.append((idx, idx + len(span), cls))
        cursor = idx + len(span)
    out.sort()
    merged, last = [], -1
    for s, e, c in out:
        if s >= last:
            merged.append((s, e, c))
            last = e
    return merged


def char_bio(src, edits):
    """原文逐字 BIO: lab[i] = 0(O)/1+2k(B-类k)/2+2k(I-类k)。"""
    lab = [0] * len(src)
    for s, e, c in locate(src, edits):
        k = _CLS_INDEX[c]
        lab[s] = 1 + 2 * k
        for i in range(s + 1, e):
            lab[i] = 2 + 2 * k
    return lab


def encode_example(src, edits, ctok, max_chars=MAX_CHARS):
    """单条样本 -> (echo 后的 ids, 对齐标签)。对齐铁律: ids 与标签逐位等长,
    只有第二份原文的 n 个位置监督 charBIO, 其余位置(前缀/第一份/分隔)全部 -100。"""
    char_ids = ctok.encode(src)
    ids, off = echo_ids(char_ids)
    lab = [-100] * len(ids)
    lab[off:off + len(src)] = char_bio(src, edits)
    assert len(ids) == len(lab)
    return ids, lab


# ---------------------------------------------------------------------------
# 模型: 复用 HF Qwen3 的子模块, 但前向自己写
# ---------------------------------------------------------------------------
# 为什么不调 Qwen3Model.forward: 后续接 GPU 推理优化(注意力实现替换 + CUDA graph)时,
# HF 的 mask 构造与 kwargs 透传在各条路径上各有坑; 逐层前向只有十几行, 训练与推理
# 共用同一份, 数值路径完全一致。

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class NARTagger(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
        self.config = config
        self.backbone = Qwen3Model(config)
        self.head = nn.Linear(config.hidden_size, N_LABELS)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.n_rep = config.num_attention_heads // config.num_key_value_heads
        self.scale = self.head_dim ** -0.5

    # attend(q, k, v): q (B, L, H, D), k/v (B, L, Hkv, D) -> (B, L, H*D)
    def forward(self, input_ids, position_ids, attend):
        """最朴素 eager 前向: 右侧 pad + 纯因果注意力, 不需要任何 mask。
        pad 的是 token id=SEP, pad 位置在真实 token 右侧, 因果注意力天然看不到。"""
        bb = self.backbone
        h = bb.embed_tokens(input_ids)
        cos, sin = bb.rotary_emb(h, position_ids)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)          # (B, L, 1, D) 广播到头维
        hd = self.head_dim
        for layer in bb.layers:
            a = layer.self_attn
            x = layer.input_layernorm(h)
            shp = (*x.shape[:-1], -1, hd)
            q = a.q_norm(a.q_proj(x).view(shp))
            k = a.k_norm(a.k_proj(x).view(shp))
            v = a.v_proj(x).view(shp)
            q = q * cos + _rotate_half(q) * sin                # RoPE 旋转
            k = k * cos + _rotate_half(k) * sin
            h = h + a.o_proj(attend(q, k, v))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.head(bb.norm(h))                            # (B, L, 25)

    def attend_causal(self, q, k, v):
        """pad 路径/训练: 右 padding + 因果, 真实 token 天然看不到右侧 pad, 不需要 mask。"""
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.n_rep > 1:                                      # GQA: KV 头少, 复制到 Q 头数
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return o.transpose(1, 2).reshape(*o.shape[:1], o.shape[2], -1)

    def save(self, out, ctok=None):
        """存权重 + config + tokenizer。bf16 存储(与推理一致), 半个 fp32 的体积。"""
        from safetensors.torch import save_file
        os.makedirs(out, exist_ok=True)
        self.config.save_pretrained(out)
        if ctok is not None:
            ctok.save(out)
        sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu()
              for k, v in self.state_dict().items()}
        save_file(sd, os.path.join(out, "model.safetensors"))
        with open(os.path.join(out, "nar.json"), "w") as f:
            json.dump({"labels": LABELS, "echo": True, "max_chars": MAX_CHARS}, f,
                      ensure_ascii=False)

    @classmethod
    def load(cls, model_dir, device="cuda", dtype=torch.bfloat16):
        """评测/推理用加载。meta 设备上建壳再 assign 权重, 避免 2G fp32 走一遍 CPU 内存。"""
        from safetensors.torch import load_file
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_dir)
        with torch.device("meta"):
            m = cls(cfg)
        sd = load_file(os.path.join(model_dir, "model.safetensors"), device=str(device))
        m.load_state_dict(sd, assign=True)
        # rotary 的 inv_freq 是非持久 buffer, meta 壳上要重算一份真的, 且保持 fp32
        # (bf16 的 inv_freq 会让长位置的角度漂移, 与训练数值不一致)。
        rot = m.backbone.rotary_emb
        inv, _ = rot.rope_init_fn(cfg, device)
        rot.inv_freq = inv.float()
        rot.original_inv_freq = inv.float()
        m = m.to(device=device, dtype=dtype).eval()
        rot.inv_freq = inv.float()
        rot.original_inv_freq = inv.float()
        return m

    @classmethod
    def from_causal_lm(cls, path):
        """从 v2(Qwen3ForCausalLM) 初始化主干, 丢掉 lm_head; 分类头小方差随机初始化。"""
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        m = cls(lm.config)
        m.backbone = lm.model
        nn.init.normal_(m.head.weight, std=0.02)
        nn.init.zeros_(m.head.bias)
        return m


# ---------------------------------------------------------------------------
# 两步 span 解码 + 结构修复(都不改模型概率, 不重训即可生效)
# ---------------------------------------------------------------------------

def decode_offsets(probs):
    """两步 span 解码, 返回 [(start, end, class_idx)]。

    1) 切边界: P(非O) > P(O) 的字在 span 内; 连续一段里 ΣP(B) > ΣP(I) 的字起新 span
       (一段非 O 的开头必然起 span, 哪怕 I 更大 —— 防止 span 起点被 I 投掉)。
    2) 定类: span 内每字 P(B-c)+P(I-c) 求和取 argmax, 整段只做一次类别决策 ——
       避免逐字 argmax 时同一 span 内类别跳变(100:98 -> 1 TIME + 00:98 SCORE)。
    全部走 numpy: torch 小算子每个都会释放/重抢 GIL, 服务里和事件循环、FST 管理
    线程争抢时解码会从 ~1ms 涨到几十上百 ms; numpy 小数组运算不释放 GIL, 一口气做完。
    """
    n = probs.shape[0]
    if n == 0:
        return []
    p_o = probs[:, 0]
    inside = p_o < 0.5
    if not inside.any():
        return []
    pb = probs[:, 1::2]
    pi = probs[:, 2::2]
    start = inside & (pb.sum(1) > pi.sum(1))
    prev_in = np.concatenate(([False], inside[:-1]))
    start |= inside & ~prev_in                    # 一段非 O 的开头必然起 span(哪怕 I 更大)
    starts = np.flatnonzero(start)
    # 每个 span 的终点: 下一个起点与下一个 O 中更早者。
    outs = np.flatnonzero(~inside)
    nxt_start = np.append(starts[1:], n)
    k = np.searchsorted(outs, starts)
    if len(outs):
        nxt_out = np.where(k < len(outs), outs[np.minimum(k, len(outs) - 1)], n)
    else:                                         # 整片都是 span(如片就是 "3:1"), 没有 O
        nxt_out = np.full(len(starts), n)
    ends = np.minimum(nxt_start, nxt_out)
    cls = pb + pi
    sums = np.add.reduceat(cls, starts, axis=0)   # 按起点分段求和(段内恰为 [start, 下个起点))
    # reduceat 的段是 [start_i, start_{i+1}), 当 O 提前结束 span 时要扣掉 O 部分 —— 直接重算这些段。
    c_idx = sums.argmax(1)
    short = np.flatnonzero(ends < nxt_start)
    for j in short.tolist():
        c_idx[j] = cls[starts[j]:ends[j]].sum(0).argmax()
    return list(zip(starts.tolist(), ends.tolist(), c_idx.tolist()))


_OPEN = {")": "(", "）": "（", "]": "[", "】": "【"}
_CLOSE_OF = {v: k for k, v in _OPEN.items()}
_LETTER = re.compile(r"[A-Za-z]")
_PURE_NUM = re.compile(r"^[0-9][0-9,，.]*$")
_MONEY_CODES = None


def _money_codes():
    """money/code.tsv 第一列(USD、HK$、A$ ...), 长的优先匹配。懒加载, 不用修复层时不读文件。"""
    global _MONEY_CODES
    if _MONEY_CODES is None:
        p = os.path.join(_ROOT, "wetext", "tn", "chinese", "data", "money", "code.tsv")
        codes = set()
        with open(p, encoding="utf-8") as f:
            for line in f:
                c = line.split("\t", 1)[0].strip()
                if c:
                    codes.add(c)
        _MONEY_CODES = sorted(codes, key=len, reverse=True)
    return _MONEY_CODES


def repair_offsets(src, spans):
    """修复层: 解码后三条**结构规则**, 不改模型概率。

    这些错误源自老师 v2 的标注噪声(蒸馏时被学走), 用数据修要重训且有随机性; 结构规则
    确定、可审计、零延迟成本。
      1) 去 span 首尾空白;
      2) 括号配对: span 里右括号比左括号多、且 span 左邻正是对应左括号 -> 把左括号
         并进来(对称地, 左括号多且右邻是对应右括号 -> 并进右括号), 否则读音会残留
         半个括号(模型标了 010)65432100, 修成 (010)65432100);
      3) 货币代码前缀: 纯数字 span 左侧是 money/code.tsv 里的代码(中间可隔一个空格)
         -> 并进来并定为 MONEY; 代码左侧不能是字母, 防 "XUSD 1" 误配。
    扩张后与前一个 span 重叠: 保留前一个(修复层不能制造重叠)。
    """
    out = []
    for s, e, c in spans:
        while s < e and src[s].isspace():
            s += 1
        while e > s and src[e - 1].isspace():
            e -= 1
        if s >= e:
            continue
        seg = src[s:e]
        for r, l in _OPEN.items():
            if seg.count(r) > seg.count(l) and s > 0 and src[s - 1] == l:
                s -= 1
            elif seg.count(l) > seg.count(r) and e < len(src) and src[e] == _CLOSE_OF[l]:
                e += 1
        if _PURE_NUM.match(src[s:e]):
            j = s - 1 if s > 0 and src[s - 1] == " " else s
            for code in _money_codes():
                b = j - len(code)
                if b >= 0 and src[b:j] == code and not (b > 0 and _LETTER.match(src[b - 1])):
                    s, c = b, _CLS_INDEX["MONEY"]
                    break
        if out and s < out[-1][1]:                # 扩张后与前一个 span 重叠: 保留前一个
            continue
        out.append((s, e, c))
    return out


# ---------------------------------------------------------------------------
# 推理引擎
# ---------------------------------------------------------------------------

class NARRunner:
    """单卡单句推理: 建输入 -> 一次前向 -> numpy 两步解码 -> 结构修复。

    消融阶段的最小实现(无合批/无 CUDA graph/无 packing) —— 那些属于后续的
    服务端优化, 本文件只交付"算法四件套"。同一 runner 只能在一个线程里用。
    """

    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.ctok = CharTokenizer(model_dir)
        self.model = NARTagger.load(model_dir, device=self.device, dtype=dtype)

    def tag(self, texts):
        """texts: list[str] -> list[edits]。质量口径逐句调用(单句前向, 无合批抖动)。"""
        out = []
        for src in texts:
            src = src[:MAX_CHARS]
            n = len(src)
            ids, off = echo_ids(self.ctok.encode(src))
            t = torch.tensor([ids], device=self.device)
            pos = torch.arange(len(ids), device=self.device)[None]
            with torch.inference_mode():
                logits = self.model(t, pos, self.model.attend_causal)
                probs = torch.softmax(logits[0, off:off + n].float(), dim=-1).cpu().numpy()
            spans = decode_offsets(probs)
            spans = repair_offsets(src, spans)
            out.append([[src[s:e], _CLS_LABEL[c]] for s, e, c in spans])
        return out


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def load_mix(specs):
    """--data a.jsonl:2 b.jsonl —— 冒号后是重复次数(人工数据 ×2 = 对齐 v2 每样本
    2 个 epoch 的监督量; NAR 跑 2 epoch 会过拟合, 用数据复制 + 1 epoch 替代)。"""
    recs = []
    for spec in specs:
        path, rep = spec, 1
        m = re.match(r"^(.*):(\d+)$", spec)
        if m and not os.path.exists(spec):
            path, rep = m.group(1), int(m.group(2))
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        recs += rows * rep
        print(f"[data] {path} ×{rep}: {len(rows)}", flush=True)
    return recs


def _sha16(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def _batches(encoded, bs, seed, epoch):
    """按长度分桶的随机批: 先打乱, 每 100 批一个窗口内按长度排序再切, 减少 pad。"""
    rng = random.Random(seed + epoch)
    idx = list(range(len(encoded)))
    rng.shuffle(idx)
    win = bs * 100
    out = []
    for w in range(0, len(idx), win):
        chunk = sorted(idx[w:w + win], key=lambda i: len(encoded[i][0]))
        out += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    rng.shuffle(out)
    return out


def train(args):
    import torch.distributed as dist

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)

    ctok = CharTokenizer(args.init)
    recs = load_mix(args.data)
    encoded, skipped = [], 0
    for r in recs:
        src = r["src"]
        if not src or len(src) > args.max_chars:
            skipped += 1
            continue
        encoded.append(encode_example(src, r.get("edits") or [], ctok, args.max_chars))
    if rank == 0:
        print(f"[data] encoded={len(encoded)} skipped={skipped}", flush=True)

    model = NARTagger.from_causal_lm(args.init).to(dev)
    model.train()
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[torch.cuda.current_device()])

    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.lr * args.head_lr_mult}],
                            weight_decay=args.wd, betas=(0.9, args.beta2))

    per = args.global_batch // world
    steps_per_epoch = math.ceil(len(encoded) / args.global_batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    warm = max(1, int(total * 0.03))

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    step, t0, ep = 0, time.time(), 0
    while step < total:
        for gb in _batches(encoded, args.global_batch, args.seed, ep):
            if step >= total:
                break
            mine = gb[rank * per:(rank + 1) * per] or gb[:1]
            L = max(len(encoded[i][0]) for i in mine)
            ids = torch.full((len(mine), L), SEP_ID, dtype=torch.long)
            lab = torch.full((len(mine), L), -100, dtype=torch.long)
            for r_, i in enumerate(mine):
                a, b = encoded[i]
                ids[r_, :len(a)] = torch.tensor(a)
                lab[r_, :len(b)] = torch.tensor(b)
            ids, lab = ids.to(dev), lab.to(dev)
            pos = torch.arange(L, device=dev).expand(len(mine), L)
            for g in opt.param_groups:
                g["lr"] = g.setdefault("base_lr", g["lr"]) * lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, pos, model.module.attend_causal if ddp else model.attend_causal)
            loss = F.cross_entropy(logits.float().view(-1, N_LABELS), lab.view(-1),
                                   ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if rank == 0 and (step % 20 == 0 or step == total):
                print(f"[train] step {step}/{total} loss {loss.item():.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
        ep += 1

    if rank == 0:
        (model.module if ddp else model).save(args.out, ctok)
        meta = dict(vars(args))
        meta.update({"world": world, "steps": total, "n_encoded": len(encoded),
                     "n_skipped": skipped,
                     "data_sha256_16": {s: _sha16(s.rsplit(":", 1)[0] if not os.path.exists(s) else s)
                                        for s in args.data}})
        with open(os.path.join(args.out, "train_args.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print(f"TRAIN_DONE {args.out}", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+", required=True, help="path[:重复次数] ...")
    ap.add_argument("--init", default="model/v2", help="主干初始化(v2 SFT 权重)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="1 epoch; 监督量靠数据 :N 复制凑齐(NAR 跑 2 epoch 会过拟合)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr-mult", type=float, default=10.0, help="随机初始化的分类头学习率倍数")
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
