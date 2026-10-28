"""A1 中文文本规整命令行入口。

相对原始 v2
===========
v2 让 Qwen 生成 ``span<TAB>CLASS:ctx`` 文本，再用文本 parser 解析。A1 让
Qwen 生成 ``span<|TN_CLASS|>`` token 序列，直接按 token id 恢复类别边界，
不依赖 TAB、换行或 special token 的文本解码。

推理仍保持两段式：LLM 只定类，TNReader/FST 负责读音。
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.constants import EOS_ID, SEP, TN_PREFIX
from train.label_tokens import add_label_tokens, parse_label_token_ids
from train.latex import delatex
from train.tn_fst import TNReader


MODEL = "model/a1"


class TN:
    """加载 A1 模型和 FST，并复用它们处理多条文本。"""

    def __init__(self, model=MODEL, max_new_tokens=128):
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model, fix_mistral_regex=True)
        add_label_tokens(self.tok)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        if len(self.tok) != self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tok))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.reader = TNReader()

    def tag(self, src):
        """greedy 生成类别 token 序列，并按 token id 恢复 edits。"""
        input_ids = self.tok(
            TN_PREFIX + src + SEP,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=EOS_ID,
                pad_token_id=EOS_ID,
            )
        generated_ids = output[0, input_ids.shape[1]:].tolist()
        return parse_label_token_ids(generated_ids, self.tok)

    def normalize(self, src):
        """返回 ``(spoken, edits)``；先做确定性 LaTeX 改写，再定类和读音。"""
        written = delatex(src)
        edits = self.tag(written)
        return self.reader.apply(written, edits), edits


def main():
    """从命令行或 stdin 读取文本，打印 A1 标注与最终读音。"""
    args = sys.argv[1:]
    if args == ["-"]:
        source_texts = [line.strip() for line in sys.stdin if line.strip()]
    elif args:
        source_texts = [" ".join(args)]
    else:
        source_texts = [
            "COVID-19是一种新型病毒",
            "若存在非零常数 T，使得对任意 x in D，均有 x+T in D 且 f(x+T)=2x+1",
            "这场比分3:1，胜率75%，用的是型号N8599、版本1.2.3的设备。",
            "国庆假期覆盖了01年04月~10年04月，是旅游的黄金周。",
            "测试第2页",
        ]

    tn = TN()
    for source in source_texts:
        spoken, edits = tn.normalize(source)
        print("IN  :", source)
        print("TAG :", edits)
        print("OUT :", spoken)
        print()


if __name__ == "__main__":
    main()
