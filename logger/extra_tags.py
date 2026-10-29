# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: logger/extra_tags.py
# 职责: 日志额外标签。JsonFormatter 会把 record 上命中 `extra_tags` 的字段
# 核心原理: 输出可检索的结构化 JSON 日志，并通过白名单字段控制记录内容。
# =============================================================================

"""定义允许进入结构化日志的动态字段白名单。

总体怎么做
==========
    ``make_extra_tags`` 只返回白名单字段，调用方把它传给 logging 的 ``extra``；
    ``JsonFormatter`` 遍历 LogRecord 时只复制白名单字段，避免任意对象或
    LogRecord 内建属性污染日志格式。

核心函数
========
    * ``make_extra_tags``: 构造 requestid/sdkappid 字段。

函数如何协作
============
    业务代码调用 ``make_extra_tags`` -> 字典传给 ``logging`` 的 ``extra``
    -> ``JsonFormatter.format`` 按 ``extra_tags`` 白名单筛选并编码。

关键参数
========
    * ``requestid``: 串联同一请求的多条日志，缺省为空。
    * ``sdkappid``: 业务方标识，缺省为空。

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

    参数:
        requestid: 请求链路 ID；空串表示没有调用方 ID。
        sdkappid: 业务方标识；空串表示未提供。

    返回:
        只包含 ``requestid`` 和 ``sdkappid`` 的字典。
    """
    return {"requestid": requestid, "sdkappid": sdkappid}
