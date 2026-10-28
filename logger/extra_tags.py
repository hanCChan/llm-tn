# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: logger/extra_tags.py
# 职责: 日志额外标签。JsonFormatter 会把 record 上命中 `extra_tags` 的字段
# 核心原理: 输出可检索的结构化 JSON 日志，并通过白名单字段控制记录内容。
# =============================================================================

"""日志额外标签。JsonFormatter 会把 record 上命中 `extra_tags` 的字段
挂到结构化日志里。需要新字段时，加到 `extra_tags` 并扩展 `make_extra_tags`。

本服务(llm-tn)需要两个结构化字段:
    requestid —— 把同一路请求的多条日志串起来。由客户端在请求 body 里带入，
                 缺省时服务端生成，并在响应里回传，便于跨服务对账。
    sdkappid  —— 调用方应用标识, 由客户端在请求 body 里带入(body 里是 int,
                 调用方转成 str 再传进来)。用它按业务方聚合日志(分应用看量、
                 看错误率)。缺省为空串。
耗时、入参长度等直接写在 content 里，不单独成字段。
"""


extra_tags: list[str] = ["requestid", "sdkappid"]


def make_extra_tags(requestid: str = "", sdkappid: str = "") -> dict:
    """构造日志白名单字段。

    原理: JsonFormatter 只复制 ``extra_tags`` 中声明的字段，因此这里返回的
    字典可以安全传给 ``logging`` 的 ``extra``，同时避免保留 LogRecord 自带
    属性而产生格式冲突。
    """
    return {"requestid": requestid, "sdkappid": sdkappid}
