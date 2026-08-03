# -*- coding: utf-8 -*-
"""日志额外标签。JsonFormatter 会把 record 上命中 `extra_tags` 的字段
挂到结构化日志里。需要新字段时，加到 `extra_tags` 并扩展 `make_extra_tags`。

本服务(llm-tn)只需要一个结构化字段 `requestid` —— 用它把同一路请求的多条日志
串起来。它由客户端在请求 body 里带入，缺省时服务端生成，并在响应里回传，
便于跨服务对账。耗时、入参长度等直接写在 content 里，不单独成字段。
"""


extra_tags: list[str] = ["requestid"]


def make_extra_tags(requestid: str = "") -> dict:
    return {"requestid": requestid}
