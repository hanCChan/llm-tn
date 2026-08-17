# -*- coding: utf-8 -*-
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
    return {"requestid": requestid, "sdkappid": sdkappid}
