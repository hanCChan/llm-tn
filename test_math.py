"""math 规则读音单测: 跑 written => spoken 用例表(不跑 LLM, 不依赖 pytest)。

    python test_math.py        # 只打印失败的用例
    python test_math.py -v     # 通过的也打印

用例表在 wetext/tn/chinese/test/data/math.txt, 与上游 math_test.py 共用同一份
数据 —— 加用例只改那个 txt。
"""
import os
import sys

_WETEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wetext")
if _WETEXT not in sys.path:
    sys.path.insert(0, _WETEXT)

from tn.chinese.rules.math import Math                # noqa: E402
from tn.chinese.test.utils import parse_test_case     # noqa: E402


def main():
    verbose = "-v" in sys.argv[1:]
    math = Math()

    cases = list(parse_test_case("data/math.txt"))
    fails = []
    for written, want in cases:
        try:
            got = math.normalize(written)
        except Exception:
            got = "<NO PATH>"            # 规则匹配不上整个输入
        if got == want:
            if verbose:
                print(f"ok   {written!r} -> {got!r}")
        else:
            fails.append((written, got, want))

    for written, got, want in fails:
        print(f"FAIL {written!r}")
        print(f"       got  {got!r}")
        print(f"       want {want!r}")

    print(f"\n{len(cases) - len(fails)}/{len(cases)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
