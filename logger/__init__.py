# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: logger/__init__.py
# 职责: 日志包 - 提供统一的日志接口
# 核心原理: 输出可检索的结构化 JSON 日志，并通过白名单字段控制记录内容。
# =============================================================================

"""日志包统一入口。

总体怎么做
==========
    导入时初始化 ``app`` logger；模块级 debug/info/warning/error/exception
    函数把调用位置和结构化 tags 透传给标准 logging。

核心函数
========
    * ``debug/info/warning/error/critical/exception``: 业务代码使用的日志门面。
    * ``get_logger``: 获取根 logger 或命名 logger。
    * ``make_extra_tags``: 构造 requestid/sdkappid 白名单字段。

函数如何协作
============
    业务模块调用本包门面 -> ``logging.Logger`` -> QueueHandler/文件 handler
    -> ``JsonFormatter`` -> ``log/app.log``。

关键参数
========
    * ``msg``: 日志正文。
    * ``extra``: 由 ``make_extra_tags`` 生成的白名单字段。
    * ``stacklevel=2``: 把日志调用位置指回业务代码，而不是本门面。
"""

from .core import get_logger
from .extra_tags import make_extra_tags


# 获取根logger实例
_root_logger = get_logger()

# 提供便捷的日志方法
def debug(msg, extra=None):
    """记录 DEBUG 日志。

    原理: 把调用方位置通过 ``stacklevel=2`` 指回业务代码，并透传结构化
    ``extra`` 字段，方便日志直接按 requestid/sdkappid 检索。

    参数:
        msg: 日志正文。
        extra: 可选的 ``make_extra_tags`` 结果。
    """
    _root_logger.debug(msg, stacklevel=2, extra=extra)

def info(msg, extra=None):
    """记录 INFO 日志；用于正常业务链路。

    参数:
        msg: 日志正文。
        extra: 可选的 ``make_extra_tags`` 结果。
    """
    _root_logger.info(msg, stacklevel=2, extra=extra)

def warning(msg, extra=None):
    """记录 WARNING 日志；用于可恢复异常、截断或接近配额的情况。

    参数:
        msg: 日志正文。
        extra: 可选的 ``make_extra_tags`` 结果。
    """
    _root_logger.warning(msg, stacklevel=2, extra=extra)

def warn(msg, extra=None):
    """``warning`` 的兼容别名，行为与 :func:`warning` 完全一致。"""
    _root_logger.warning(msg, stacklevel=2, extra=extra)

def error(msg, extra=None):
    """记录 ERROR 日志；用于请求失败但服务仍可继续运行的错误。

    参数:
        msg: 日志正文。
        extra: 可选的 ``make_extra_tags`` 结果。
    """
    _root_logger.error(msg, stacklevel=2, extra=extra)

def critical(msg, extra=None):
    """记录 CRITICAL 日志；用于不可继续提供服务的严重状态。

    参数:
        msg: 日志正文。
        extra: 可选的 ``make_extra_tags`` 结果。
    """
    _root_logger.critical(msg, stacklevel=2, extra=extra)

def exception(msg, extra=None):
    """记录 ERROR 日志并附带当前异常堆栈。

    原理: 依赖 ``logging.exception`` 在 except 上下文中读取当前异常信息；
    调用点必须位于异常处理分支内，否则不会产生 traceback。

    参数:
        msg: 日志正文。
        extra: 可选的 ``make_extra_tags`` 结果。
    """
    _root_logger.exception(msg, stacklevel=2, extra=extra)


__all__ = ['get_logger', 'debug', 'info', 'warning', 'error', 'critical', 'exception', 'make_extra_tags']
