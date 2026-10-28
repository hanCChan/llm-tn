# =============================================================================
# llm-tn source header
# Path: train/latex.py
# 职责: LaTeX -> math 规则认得的写法: 正则把 LaTeX 记法改写成等价的 unicode 算式
# 核心原理: 围绕类别 token 协议完成数据构造、训练、FST 读音或线上调度，保持训练与推理口径一致。
# =============================================================================

"""LaTeX -> math 规则认得的写法: 正则把 LaTeX 记法改写成等价的 unicode 算式。

只做"记法改写", 不管读音 —— 改写后的文本走原有管线(LLM 定类 -> TNReader 读音),
读得对不对是 TN 规则的事。

    \\frac{1}{2}+\\sqrt{9}=3.5   -->  1/2+√(9)=3.5  -->  一除以二加根号九等于三点五
    x^{2}+y_{1}\\geq 0           -->  x^(2)+y1≥0    -->  x的二次方加y一大于等于零

嵌套靠反复套正则到定点: 每轮只匹配参数里已不含花括号的最内层命令, 外层等下一轮。
表里没有的 \\command 当排版命令丢掉(连 {…} 参数), 不猜读法。
"""
import os
import re

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "latex")
_A = r"\{([^{}]*)\}"                                    # 已不含花括号的最内层参数
# 命令前缀: 一或两个反斜杠。调用方多转义一层时会送来 '\\frac', 直接在各条规则里认掉,
# 不必先做一遍全局归一 —— 归一得靠白名单区分命令与 'C:\\newfolder' 这类路径, 而规则
# 本身已经限定了命令名, 天然就是白名单。裸 '\\'(LaTeX 换行)不匹配任何命令, 不受影响。
_BS = r"\\{1,2}"
# LaTeX 公式的四类特征, 命中任一即认为整句含公式:
#   1. 反斜杠命令与定界符: '\frac' '\alpha' '\begin{...}' '\(' '\['
#      命令限定 2+ 字母, 减少 '\n' '\t' 这类代码转义符的误命中; 词表里最短的
#      真命令也是 2 字母('\pi' '\le'), 不会漏。
#   2. '$$' 跨行显示公式: '$$\nx^2+y^2=z^2\n$$' —— markdown 最常见的写法,
#      不能并进第 3 条: 那条排除了换行, 会把独占一行的 '$$' 漏掉。
#   3. 一对 '$' 包裹的行内公式: '$x^n$'
#   4. 花括号上下标: 'x^{2}' 'a_{i}' —— 无定界符也无 \command 的裸公式全靠这条,
#      否则 'f^{-1}(y)' 进不了改写, 花括号漏到读音层会念成 'f^{负一}'。
# 这里判宽无妨: 真要动手剥定界符还得过 _FORMULA + _MATHY 两道严格校验,
# 所以货币('价格$100')、Windows 路径('C:\newfolder')放行进来也不会被改坏。
_HAS_LATEX = re.compile(r"\\(?:[a-zA-Z]{2,}|[(\[])|\$\$|\$[^$\n]{1,200}\$|[\^_]\{")


def _load(name):
    """读取制表符分隔的 LaTeX 映射表，返回两列组成的字典。

    原理: 只接受包含 TAB 的行，注释或损坏行会被自然跳过；映射表在模块
    导入时一次加载，后续匹配不再访问磁盘。
    """
    with open(os.path.join(_DATA, name), encoding="utf-8") as f:
        return dict(line.rstrip("\n").split("\t", 1) for line in f if "\t" in line)


_TO_UNICODE = _load("to_unicode.tsv")                   # \times -> ×, 由 math 规则读
_TO_CHINESE = _load("to_chinese.tsv")                   # \in -> 属于, 无 unicode 才给中文
# 长命令优先, 否则 \le 会先吃掉 \leq 的前缀。命令名去掉表里的反斜杠, 由 _BS 统一带上。
_CMD = re.compile("%s(?:%s)(?![A-Za-z])" % (_BS, "|".join(
    re.escape(c.lstrip("\\")) for c in sorted(set(_TO_UNICODE) | set(_TO_CHINESE),
                                              key=len, reverse=True))))

_RULES = [
    # 矩阵环境: \begin{pmatrix} -> 矩阵, 只标出"这是个矩阵", 元素照原样留给后续规则。
    # \end{pmatrix} 由 _FINAL 的未知命令那条丢掉。
    (re.compile(_BS + r"begin\s*\{[a-zA-Z]*matrix\*?\}"), "矩阵"),
    # 排版噪声: \quad x -> ' x'    a\,b -> ab
    (re.compile(_BS + r"(?:displaystyle|limits|nolimits|quad|qquad)|" + _BS + r"[,;!: ]"), ""),
    # 只标高度的定界符: \left(a\right) -> a    \left| x \right| -> ' x '
    (re.compile(_BS + r"(?:left|right)\s*(?:" + _BS + r"[A-Za-z]+|.)"), ""),
    # 取内容: \text{速度} -> 速度  \mathrm{d} -> d  \mathbb{R} -> R  \textbf{重要} -> 重要
    # 只改字形的命令, 花括号里是正文该照读 —— 不列进来会被 _FINAL 当未知命令连内容一起删掉。
    (re.compile(_BS + r"(?:text(?:bf|it|rm|sf|tt|up|sl|md|normal)?"
                r"|math(?:rm|bf|bb|it|cal|sf|tt|frak)"
                r"|emph|underline|operatorname)\s*" + _A), r"\1"),
    # 分式: \frac{1}{2} -> 1/2    \dfrac{a+b}{c} -> a+b/c
    (re.compile(_BS + r"[dt]?frac\s*%s\s*%s" % (_A, _A)), r"\1/\2"),
    # 开方: \sqrt[3]{8} -> 3次√(8)
    (re.compile(_BS + r"sqrt\s*\[([^\[\]]*)\]\s*" + _A), r"\1次√(\2)"),
    # 根号: \sqrt{16} -> √(16)
    (re.compile(_BS + r"sqrt\s*" + _A), r"√(\1)"),
    # 大算符必须在下标/上标规则之前 —— 否则 \sum_{i=1}^{n} 的 _{i=1} 先被下标规则吃掉,
    # 剩下 \sumi 就不再是表里的 \sum, 最后残留成 '\sumi=1^(n)'。
    # 求和/求积: \sum_{i=1}^{n} -> 从i=1到n求和    \prod_{i=1}^{n} -> 从i=1到n求积
    (re.compile(_BS + r"(sum|prod)\s*_\s*%s\s*\^\s*%s" % (_A, _A)),
     lambda m: "从%s到%s%s" % (m.group(2), m.group(3), "求和" if m.group(1) == "sum" else "求积")),
    # 积分: \int_{a}^{b} -> 从a到b求积分
    (re.compile(_BS + r"(?:int|iint|oint)\s*_\s*%s\s*\^\s*%s" % (_A, _A)), r"从\1到\2求积分"),
    # 极限: \lim_{x \to 0} -> 当x趋于0时(\to 已在上一轮被换成"趋于")
    (re.compile(_BS + r"lim\s*_\s*" + _A), r"当\1时"),
    # 反函数: f^{-1}(y) -> f逆y。f^{-1} 不是 1/f, 走下面的上标规则会读成"f的负一次方"。
    # 只认 f 且紧跟自变量括号这一种写法, 其余(g^{-1} A^{-1} x^{-1})不动。
    (re.compile(r"(?<![A-Za-z])f\s*\^\s*\{-1\}\s*\(([^()]*)\)"), r"f逆\1"),
    # 上标: x^{2} -> x^(2)    2^{10} -> 2^(10)
    (re.compile(r"\^\s*" + _A), r"^(\1)"),
    # 下标: y_{1} -> y1, math 规则读作"y一"。裸 _ 不处理 —— snake_case 标识符里太常见。
    (re.compile(r"_\s*" + _A), r"\1"),
    # 符号命令: \times -> ×  \geq -> ≥  \alpha -> α  \in -> 属于  \forall -> 任意
    # 查表前把前缀归一成单反斜杠 —— 匹配到的可能是 '\\times'(调用方多转义), 而表里
    # 的 key 只带一个反斜杠, 直接拿 group(0) 查会 KeyError。
    (_CMD, lambda m: (lambda k: _TO_UNICODE.get(k) or _TO_CHINESE[k])(
        "\\" + m.group(0).lstrip("\\"))),
]

# 收尾规则: 等上面全部套到定点后再跑一次。不能混在循环里 —— 丢弃未知命令那条会把
# 上一轮刚替换出的中间结果吃掉(\frac{\frac{1}{2}}{3} 第一轮得到 \frac{1/2}{3},
# 同一轮里 \frac{1/2} 就被当未知命令删了, 只剩 '3')。
_FINAL = [
    # 表里没有的命令连它的 {…} 参数一起丢掉 —— 走到这里的都是排版命令, 不读:
    # \begin{align} -> ''    \label{eq:1} -> ''    \hspace{1cm} -> ''
    # 也不能留着: 下面剥花括号那条会把它搓成 \beginalign, 比丢掉更糟。
    (re.compile(r"\\[A-Za-z]+\s*" + _A + "?"), ""),
    # 剩下的花括号只是分组, 不读: {3} -> 3    {a+b} -> a+b
    (re.compile(r"\{([^{}]*)\}"), r"\1"),
]


# 定界符候选: $…$ / $$…$$ / \(…\) / \[…\]
_DELIM = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
# 定界符里必须是"公式的样子"才认: 只含 ASCII 算式字符与 \command, 不含汉字。
# 否则 '价格$100，优惠$20' 里两个 $ 会被当成一对定界符, 把 '100，优惠' 剥掉;
# '收费$5和$8' 同理。判断可以宽(has_latex), 但真要动手替换必须严。
_FORMULA = re.compile(r"^[A-Za-z0-9\s+\-*/^_=<>(){}\[\]|.,;:'!\\&×÷≥≤≠≈≡±√∞°α-ωΑ-Ω]+$")
# 光"全是 ASCII"不够, 还得真有算式特征(\command 或运算符/上下标/希腊字母) ——
# 否则 'The cost is $5 or $10 total' 里两个 $ 会被当定界符, 剥成 '5 or 10'。
# 汉字版靠 _FORMULA 就挡住了, 英文版只能靠这条。
_MATHY = re.compile(r"\\[A-Za-z]+|[\^_=<>+*/×÷≥≤≠≈≡±√∞°]|[α-ωΑ-Ω]")


def has_latex(text):
    """是否含 LaTeX 标记。靠 \\command / $ / \\( / \\[ 判定, 确定性识别不靠模型。"""
    return bool(text) and bool(_HAS_LATEX.search(text))


def _unwrap(src):
    """反复应用 LaTeX 改写规则，直到文本达到稳定状态。

    原理: 嵌套命令必须由内向外展开，所以每轮只处理不含内层花括号的规则；
    定点迭代让 ``\\frac{\\frac{1}{2}}{3}`` 逐层化简。上限 16 轮用于阻止
    异常输入导致无限改写，最后再执行未知命令清理和花括号剥离。
    """
    text = src
    for _ in range(16):                                 # 定点迭代; 上限防病态输入
        before = text
        for pattern, repl in _RULES:
            text = pattern.sub(repl, text)
        if text == before:
            break
    for pattern, repl in _FINAL:
        text = pattern.sub(repl, text)
    return text


def delatex(text):
    """把 text 里的 LaTeX 公式改写成 math 规则认得的写法; 定界符一并剥掉。"""
    if not has_latex(text):
        return text

    def one(m):
        """校验一个数学定界符候选，并在确属公式时执行内部改写。

        ``$`` 也常用于货币，不能仅凭成对出现就剥掉。这里要求内容同时满足
        字符白名单 ``_FORMULA`` 和数学特征 ``_MATHY``，否则原样保留。
        """
        body = next(g for g in m.groups() if g is not None)
        # 不像公式就整段原样留着(连定界符) —— 大概率是货币或路径, 不是数学。
        return (_unwrap(body) if _FORMULA.match(body) and _MATHY.search(body)
                else m.group(0))

    return _unwrap(_DELIM.sub(one, text))               # 收尾: 处理无定界符的裸命令
