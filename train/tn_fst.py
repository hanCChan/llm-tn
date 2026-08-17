"""读音: LLM 定类 + wetext 场景 normalizer 按 span 路由读音。

LLM 把句子标成 [(span, "CLASS:ctx")...]。TNReader 按 span 的 CLASS 路由到对应
场景 normalizer 逐段读音, 拼接后整句再过一遍默认 Normalizer 兜底。

场景 normalizer(各自 boost 相关规则):
    NormalizerTime  -> DATE / TIME              (date/time)
    NormalizerMath  -> MATH / FRACTION / MEASURE / MONEY / PERCENT
    NormalizerScore -> SCORE                    (score)
    NormalizerPhone -> PHONE / VERSION          (phone/version)
    NormalizerSerial-> SERIAL / ADDRESS         (逐位读, 1 读"一"而非"幺", 段间"-"读"杠")
    其余(未识别类) -> 默认 Normalizer

用法:
    from train.tn_fst import TNReader
    r = TNReader()
    r.apply("会议3月5日下午2点开", [["3月5日", "DATE:date"], ["下午2点", "TIME:time"]])
"""
import os
import sys

_WETEXT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wetext"))
if _WETEXT not in sys.path:
    sys.path.insert(0, _WETEXT)

from importlib_resources import files            # noqa: E402
from tn.chinese.normalizer import Normalizer     # noqa: E402

from tn.chinese.normalizer_math import NormalizerMath     # noqa: E402
from tn.chinese.normalizer_phone import NormalizerPhone   # noqa: E402
from tn.chinese.normalizer_score import NormalizerScore   # noqa: E402
from tn.chinese.normalizer_serial import NormalizerSerial  # noqa: E402
from tn.chinese.normalizer_time import NormalizerTime     # noqa: E402

# CLASS(取标签冒号前的类别名) -> 场景 normalizer 属性名; 未列出的类走默认 normalizer。
_ROUTE = {
    "DATE": "time",
    "TIME": "time",
    "MATH": "math",
    "FRACTION": "math",
    "MEASURE": "math",
    "MONEY": "math",
    "PERCENT": "math",
    "SCORE": "score",
    "PHONE": "phone",
    "VERSION": "phone",
    "SERIAL": "serial",
    "ADDRESS": "serial",
}


class TNReader:
    def __init__(self):
        # full_to_half=False 保留全/半角标点, 便于逐字对齐 gold_spoken。
        # 每个场景 normalizer 用独立 cache 子目录, 避免共享 cache_dir 时的 FST 撞车。
        fsts = files("tn") / "fsts"
        self.default = Normalizer(cache_dir=fsts, full_to_half=False)
        self.time = NormalizerTime(cache_dir=fsts / "time", full_to_half=False)
        self.math = NormalizerMath(cache_dir=fsts / "math", full_to_half=False)
        self.score = NormalizerScore(cache_dir=fsts / "score", full_to_half=False)
        self.phone = NormalizerPhone(cache_dir=fsts / "phone", full_to_half=False)
        self.serial = NormalizerSerial(cache_dir=fsts / "serial", full_to_half=False)

    def _normalizer_for(self, label):
        """标签 "CLASS:ctx" -> 场景 normalizer, 无对应场景则返回 None(走默认)。"""
        cls = label.split(":", 1)[0].strip().upper()
        attr = _ROUTE.get(cls)
        return getattr(self, attr) if attr else None

    def _read_scene(self, text, normalizer):
        """场景遍: 用场景 normalizer 强制把该 span 读成对应类别。

        场景 normalizer 只 union 了少数规则, 对未覆盖的部分原样透传(如 NormalizerTime
        对 '3月5日' 不转), 由 apply 之后的整句默认遍补齐。抛异常则返回原文。
        """
        if not text:
            return ""
        if normalizer is None:
            return text
        try:
            return normalizer.normalize(text)
        except Exception:
            return text

    def _locate(self, src, edits):
        """把 edits 在 src 中按出现顺序定位为不重叠的 [(start, end, label)] 区间。"""
        spans = []
        cursor = 0
        for item in edits:
            if not item:
                continue
            span, label = item[0], item[1]
            if not span:
                continue
            idx = src.find(span, cursor)
            if idx < 0:                       # 从头再找一次(顺序被模型打乱时的兜底)
                idx = src.find(span)
            if idx < 0:
                continue
            spans.append((idx, idx + len(span), label))
            cursor = idx + len(span)
        spans.sort()
        # 丢弃与前一区间重叠的项, 保证可线性切分。
        merged, last_end = [], -1
        for start, end, label in spans:
            if start < last_end:
                continue
            merged.append((start, end, label))
            last_end = end
        return merged

    def apply(self, src, edits=None, fallback=True):
        """按 LLM 标注逐 span 路由读音; edits 为空时整句走默认 normalizer。

        两遍读音: 先逐 span 走场景 normalizer, 拼接后**整句**再走一遍默认 normalizer
        兜底。默认遍作用于整句(而非逐段)是必要的 —— 依赖右邻上下文的规则(如罗马数字:
        裸 XX 与英文缩写无法区分, 只有紧跟"世纪"等单位才转)在 LLM 只标出部分 span
        (如 "XXI世纪" 只标 "XXI"、"IDR 100000" 只标 "100000")时, 逐段跑会切断该
        上下文导致规则失效。
        场景已转成的中文(如 '三比一' '幺三八')不含阿拉伯数字/符号, 默认遍不会再改动,
        故两遍安全。
        """
        if not src:
            return ""
        if not edits:
            return self.default.normalize(src)

        spans = self._locate(src, edits)
        if not spans:
            return self.default.normalize(src) if fallback else src

        out, pos = [], 0
        for start, end, label in spans:
            if start > pos:                                   # span 前的普通文本
                out.append(src[pos:start])
            out.append(self._read_scene(src[start:end], self._normalizer_for(label)))
            pos = end
        if pos < len(src):                                    # 末尾普通文本
            out.append(src[pos:])

        staged = "".join(out)
        try:
            return self.default.normalize(staged)
        except Exception:
            return staged


if __name__ == "__main__":
    r = TNReader()
    samples = [
        ("会议3月5日下午2点在中山路88号3栋502室召开，预算¥5000，联系电话400-820-8820。",
         [["3月5日", "DATE:date"], ["下午2点", "TIME:time"], ["88号3栋502室", "ADDRESS:address"],
          ["¥5000", "MONEY:money"], ["400-820-8820", "PHONE:phone"]]),
        ("这场比分3:1，胜率75%，用的是型号N8599、版本1.2.3的设备。",
         [["3:1", "SCORE:score"], ["75%", "PERCENT:percent"], ["N8599", "SERIAL:serial"],
          ["1.2.3", "VERSION:version"]]),
        ("仓库还有5箱货，计算3x60=180千米，活动10月1日~10月7日。",
         [["5箱", "MEASURE:measure"], ["3x60=180千米", "MATH:math"], ["10月1日~10月7日", "DATE:date"]]),
    ]
    for s, edits in samples:
        print("IN  :", s)
        print("TAG :", edits)
        print("OUT :", r.apply(s, edits))
        print()
