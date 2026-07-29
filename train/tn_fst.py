"""读音: 退回纯 wetext 默认 Normalizer(不再用场景化 normalizer)。

场景化按类 normalizer 的方案已移除。现在直接用 wetext 默认 Normalizer 读整句。
LLM 标注仍产出(供 tag 评测), 但不参与读音——默认 normalizer 不做按 span 路由。
(后续更简单的方案会在此默认基础上迭代。)

用法:
    from train.tn_fst import TNReader
    r = TNReader()
    r.apply("会议3月5日下午2点开")   # -> 默认 TN 口语串
"""
import os
import sys

_WETEXT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wetext"))
if _WETEXT not in sys.path:
    sys.path.insert(0, _WETEXT)

from importlib_resources import files            # noqa: E402
from tn.chinese.normalizer import Normalizer     # noqa: E402


class TNReader:
    def __init__(self):
        # 原 wetext 默认 Normalizer; full_to_half=False 保留全/半角标点, 便于逐字对齐 gold_spoken
        self.default = Normalizer(cache_dir=files("tn") / "fsts", full_to_half=False)

    def apply(self, src, edits=None, fallback=True):
        """默认 TN 读音。edits 仅为兼容旧签名保留, 不参与读音。"""
        return self.default.normalize(src)


if __name__ == "__main__":
    r = TNReader()
    for s in ["会议3月5日下午2点在中山路88号3栋502室开，预算¥5000。",
              "比分3:1，胜率75%，型号N8599，版本1.2.3。"]:
        print("IN :", s)
        print("OUT:", r.apply(s))
        print()
