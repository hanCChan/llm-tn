"""A1 专用类别 token 协议。

每个类别注册为一个不可拆分的 special token，并同时承担类别字段和记录边界。
例如 ``110<|TN_PHONE|>110人<|TN_MEASURE|>`` 对应两条 edits，不需要 TAB 或
换行。该模块只提供协议工具，不改变默认的 ``span<TAB>CLASS`` 路径。

相对原始 v2:

* v2 类别写在普通文本 completion 中，例如 ``PHONE:phone``；
* A1 把类别和记录边界融合为一个 token, 例如 ``<|TN_PHONE|>``；
* A1 tokenizer 多 12 个 token，Qwen tied embedding/LM-head 多 12 行；
* 推理必须从 ``output.token_ids`` 解析，不能依赖会过滤 special token 的文本。
"""

from train.labels import LABELS, to_class


LABEL_TOKENS = {label: f"<|TN_{label}|>" for label in sorted(LABELS)}
TOKEN_TO_LABEL = {token: label for label, token in LABEL_TOKENS.items()}


def add_label_tokens(tokenizer) -> dict[str, int]:
    """把 12 个类别 token 注册到 tokenizer，并返回 label 到 token id 的映射。

    重复调用是安全的；Hugging Face tokenizer 会复用已经存在的 special token。
    """
    tokenizer.add_special_tokens({
        "additional_special_tokens": list(LABEL_TOKENS.values()),
    })
    ids = {
        label: tokenizer.convert_tokens_to_ids(token)
        for label, token in LABEL_TOKENS.items()
    }
    invalid = {
        label: token_id for label, token_id in ids.items()
        if token_id is None or token_id == tokenizer.unk_token_id
    }
    if invalid:
        raise ValueError(f"failed to register label tokens: {invalid}")
    return ids


def render_label_token_output(edits: list[list[str]] | list[tuple[str, str]]) -> str:
    """把 edits 渲染为 ``span<|TN_CLASS|>`` 拼接格式，不添加 EOS。"""
    return "".join(
        f"{span}{LABEL_TOKENS[to_class(label)]}"
        for span, label in edits
    )


def parse_label_token_ids(token_ids: list[int], tokenizer) -> list[list[str]]:
    """从生成 token ids 恢复 ``[[span, CLASS], ...]``。

    解析必须基于 token ids，而不是 ``output.text``。vLLM 默认会过滤 special
    token，直接解码文本会丢失类别边界。EOS 和最后一个类别 token 之后的内容
    不属于任何 span，会被忽略。
    """
    ids = add_label_tokens(tokenizer)
    id_to_label = {token_id: label for label, token_id in ids.items()}
    edits = []
    buffer = []
    for token_id in token_ids:
        label = id_to_label.get(token_id)
        if label is None:
            buffer.append(token_id)
            continue
        span = tokenizer.decode(buffer, skip_special_tokens=False).strip()
        if span:
            edits.append([span, label])
        buffer.clear()
    return edits
