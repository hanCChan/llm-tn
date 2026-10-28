# =============================================================================
# llm-tn source header
# Path: train/labels.py
# 职责: 训练和推理共用的标签协议
# 核心原理: 围绕类别 token 协议完成数据构造、训练、FST 读音或线上调度，保持训练与推理口径一致。
# =============================================================================

"""训练和推理共用的标签协议。

最终模型只生成 ``span<TAB>CLASS``，不再生成 ``CLASS:ctx``。为了兼容
``model/v2`` 基线，本模块仍会把 ``CLASS:ctx`` 归一化为 ``CLASS``；
所有入口都从这里取标签规则，避免训练和服务各自维护一份解析逻辑。
"""

from train.constants import SEPARATOR


# 最终模型支持的 12 个语义类别。集合只用于校验类别数量和文档说明，
# 解析时保留未知大写标签，让 TNReader 可回退到默认 normalizer。
LABELS = frozenset({
    "MEASURE", "DATE", "TIME", "MONEY", "PHONE", "MATH",
    "SERIAL", "PERCENT", "SCORE", "FRACTION", "VERSION", "ADDRESS",
})


def to_class(label):
    """把新格式 ``CLASS`` 或旧格式 ``CLASS:ctx`` 归一化为大写类别。

    旧模型可能输出 ``DATE:date``，而最终模型只输出 ``DATE``。这里只截取
    冒号前的类别名，既保持 v2 基线可运行，也不会让上下文后缀进入路由逻辑。
    """
    token = str(label).strip()
    return token.split(":", 1)[0].strip().upper()


def parse_edits(text):
    """把模型多行输出解析为 ``[[span, uppercase_class], ...]``。

    每行只接受一个 TAB 分隔的 span 和标签。空行、没有 TAB 的行以及缺少
    span/标签的行直接跳过，避免模型偶发输出污染后续定位和读音流程。
    """
    edits = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or SEPARATOR not in line:
            continue
        span, _, label = line.partition(SEPARATOR)
        span = span.strip()
        cls = to_class(label)
        if span and cls:
            edits.append([span, cls])
    return edits
