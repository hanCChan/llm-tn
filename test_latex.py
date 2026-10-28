# =============================================================================
# llm-tn source header
# Path: test_latex.py
# 职责: latex 改写单测: 跑 latex => 改写后 用例表(不跑 LLM, 不跑 FST, 不依赖 pytest)
# 核心原理: 用固定样例或真实数据做回归验证，隔离并暴露规则、边界或协议回归。
# =============================================================================

"""latex 改写单测: 跑 latex => 改写后 用例表(不跑 LLM, 不跑 FST, 不依赖 pytest)。

    python test_latex.py        # 只打印失败的用例
    python test_latex.py -v     # 通过的也打印

测的是 delatex 的记法改写(LaTeX -> math 规则认得的 unicode 写法), 读音不在这测
—— 读音由 wetext 规则负责, 见 test_math.py。

用例表在 wetext/tn/chinese/test/data/latex.txt, 格式与 math.txt 一致(written => spoken),
加用例只改那个 txt。
"""
import os
import sys

_WETEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wetext")
if _WETEXT not in sys.path:
    sys.path.insert(0, _WETEXT)

from tn.chinese.test.utils import parse_test_case     # noqa: E402

from train.latex import delatex                       # noqa: E402


def main():
    """逐条执行 LaTeX 改写用例并返回进程退出码。

    原理: 只比较 ``delatex`` 的确定性文本结果，不加载模型或读音 FST，因此
    该测试可快速隔离公式记法替换错误。
    """
    verbose = "-v" in sys.argv[1:]

    cases = list(parse_test_case("data/latex.txt"))
    fails = []
    for written, want in cases:
        got = delatex(written)
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
