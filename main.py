"""端到端中文文本规整(TN): 中文文本 -> LLM 标注 -> wetext FST 读音 -> 口语文本。

    python main.py "会议3月5日下午2点，客服电话400-820-8820"
    python main.py                 # 跑内置示例
    echo "..." | python main.py -  # 从 stdin 读

流程:
  1) LLM(SFT Qwen3-0.6B) 把句子标注成 [(span, "CLASS:ctx")...]
  2) TNReader 两遍读音: 场景化强制读 + 默认 TN 兜底 -> 口语串
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.constants import TN_PREFIX, SEP, EOS_ID, SEPARATOR
from train.latex import delatex
from train.tn_fst import TNReader

MODEL = "model/v2"


class TN:
    def __init__(self, model=MODEL, max_new_tokens=128):
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model, fix_mistral_regex=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.reader = TNReader()   # 预加载全部 wetext 场景 normalizer

    def tag(self, src):
        """LLM 标注 -> [[span, "CLASS:ctx"], ...]"""
        ids = self.tok(TN_PREFIX + src + SEP, add_special_tokens=False,
                       return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new_tokens,
                do_sample=False, eos_token_id=EOS_ID, pad_token_id=EOS_ID)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        edits = []
        for line in text.split("\n"):
            if SEPARATOR in line:
                span, _, label = line.partition(SEPARATOR)
                if span and label:
                    edits.append([span, label])
        return edits

    def normalize(self, src):
        """中文文本 -> (口语文本, LLM标注)

        先把 LaTeX 记法改写成 math 规则认得的写法(\\frac{1}{2} -> 1/2), 再走标注+读音。
        改写是确定性的正则替换, 不含 LaTeX 的文本原样返回。
        """
        src = delatex(src)
        edits = self.tag(src)
        return self.reader.apply(src, edits), edits


def main():
    args = sys.argv[1:]
    if args == ["-"]:
        srcs = [line.strip() for line in sys.stdin if line.strip()]
    elif args:
        srcs = [" ".join(args)]
    else:
        srcs = [
            "COVID-19是一种新型病毒",
            "若存在非零常数  T ，使得对任意  x in D ，均有  x+T in D  且  f(x+T)=2x+1，那么g(x)=\\frac{x^2}{x}",
            "[y][in2]卡里100块"
            "这场比分3:1，胜率75%，用的是型号N8599、版本1.2.3的设备。",
            "所以1.857（857循环）＞ 1.85（7循环）＞1.8（57循环）",
            "所以1.857 > 1.85 > 1.8（57循环）",
            "所以1.857 ＞ 1.85 > 1.8（57循环）",
            "国庆假期覆盖了01年04月~10年04月，是旅游的黄金周。",
            "独立公式示例： $$ \\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi} $$",
            "矩阵示例： $$ A = \\begin{pmatrix} 1 & 0 \\ 0 & 1 \\end{pmatrix} $$",
            "测试第2页" 
        ]
    tn = TN()
    for s in srcs:
        spoken, edits = tn.normalize(s)
        print("IN  :", s)
        print("TAG :", edits)
        print("OUT :", spoken)
        print()


if __name__ == "__main__":
    main()
