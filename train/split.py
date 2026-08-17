"""按标点把长文本切成小片, 供 LLM 逐片标注。

为什么要切:
    训练集(data/train_merged.jsonl)是单句/短句, 一句一到两个 span。几百字的
    整段直接喂进去有三个问题: 输出行数撞 max_new_tokens 上限被静默截断、长上下文
    下漏标串标、span 只有字符串没有 offset 导致 _locate 在重复 span 上定位错位。
    切成小片后每片都回到训练分布内, 三个问题一起消失。

    切碎不吃亏: AsyncLLM 的 continuous batching 会把同一请求的 N 片合成一个 batch,
    端到端延迟约等于单片延迟, 不是 N 倍。

切分规则(见 split_chunks):
    在标点处切, 但标点前后都是数字时不切 —— 会被切断的数字, 其分隔符两侧必然都是
    数字(1,234,567 的 ',' / 1.2.3 的 '.' / 2:30 的 ':' / 第3、4名 的 '、'),
    这条规则全挡住。只有一侧是数字则可以切(如 "400-870-0508，工作时间是..."),
    这类切点不构成歧义。而 400-820-8820、3x60=180、01年04月~10年04月 用的分隔符
    不在 _PUNCT 里, 天然不受影响。

为什么不切得比标点更碎:
    标签判据词就在同一个小句里 —— 比分3:1 是 SCORE 而 2:30 是 TIME,
    版本1.2.3 是 VERSION 而 型号N8599 是 SERIAL, ¥5000 是 MONEY 而 5箱 是 MEASURE。
    切到小句为止上下文还在, 再往下切成裸 "3:1" 模型只能猜。

不变式(test/test_split.py 拿 4167 条 gold 验证):
    1) 无损:     "".join(split_chunks(t)) == t
    2) 不断 span: 每个 gold span 在原文中的区间完整落在某一个片内
"""
import re

# 句末 + 句内标点都切。
_PUNCT = set("。！？；，、：,;:!?…\n")
# 紧跟标点的收尾符(引号/括号)归前一片, 别让它成为下一片的开头。
_CLOSE = set("”’」』）》】〕)\"'")
# 全角数字同样算数字: 训练数据里 01年04月 这类全角写法存在。
_DIGIT = re.compile(r"[0-9０-９]")
# 无数字但要读的算式符号: delatex 改写后的公式可能整片不含数字(f(x+T) = f(x)),
# 只靠 has_digit 会被当普通文本原样透传, 该读的 '+' '=' 就漏了。
_MATH = re.compile(r"[+\-*/^=<>×÷≥≤≠≈≡±√∞]|[α-ωΑ-Ω]")


def has_digit(text):
    """片内是否含数字或算式符号。都不含则无需 LLM 标注 —— 也就无需读音, 原样透传。"""
    return _DIGIT.search(text) is not None or _MATH.search(text) is not None


def _isdigit(ch):
    return bool(ch) and _DIGIT.match(ch) is not None


def split_chunks(text):
    """按标点切片, 标点前后都是数字时不切。

    返回的片保留原标点与空白, 拼接后严格等于入参(无损), 所以逐片读音的结果
    拼回去不会掉字。空串返回空列表。
    """
    if not text:
        return []
    out, start, i, n = [], 0, 0, len(text)
    while i < n:
        if text[i] not in _PUNCT:
            i += 1
            continue
        # 连续标点("?!" "……")和其后的收尾符一并吞掉, 作为一个切点。
        j = i + 1
        while j < n and (text[j] in _PUNCT or text[j] in _CLOSE):
            j += 1
        left = text[i - 1] if i else ""             # 片首标点: 左侧视作非数字
        right = text[j] if j < n else ""            # 片尾标点: 右侧视作非数字
        if not (_isdigit(left) and _isdigit(right)):
            out.append(text[start:j])
            start = j
        i = j
    if start < n:
        out.append(text[start:])
    return out
