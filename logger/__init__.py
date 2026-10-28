# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: logger/__init__.py
# 职责: 日志包 - 提供统一的日志接口
# 核心原理: 输出可检索的结构化 JSON 日志，并通过白名单字段控制记录内容。
# =============================================================================

"""日志包 - 提供统一的日志接口"""

from .core import get_logger
from .extra_tags import make_extra_tags


# 获取根logger实例
_root_logger = get_logger()

# 提供便捷的日志方法
def debug(msg, extra=None):
    """记录 DEBUG 日志。

    原理: 把调用方位置通过 ``stacklevel=2`` 指回业务代码，并透传结构化
    ``extra`` 字段，方便日志直接按 requestid/sdkappid 检索。
    """
    _root_logger.debug(msg, stacklevel=2, extra=extra)

def info(msg, extra=None):
    """记录 INFO 日志；原理同 :func:`debug`，用于正常业务链路。"""
    _root_logger.info(msg, stacklevel=2, extra=extra)

def warning(msg, extra=None):
    """记录 WARNING 日志；用于可恢复异常、截断或接近配额的情况。"""
    _root_logger.warning(msg, stacklevel=2, extra=extra)

def warn(msg, extra=None):
    """``warning`` 的兼容别名，行为与 :func:`warning` 完全一致。"""
    _root_logger.warning(msg, stacklevel=2, extra=extra)

def error(msg, extra=None):
    """记录 ERROR 日志；用于请求失败但服务仍可继续运行的错误。"""
    _root_logger.error(msg, stacklevel=2, extra=extra)

def critical(msg, extra=None):
    """记录 CRITICAL 日志；用于不可继续提供服务的严重状态。"""
    _root_logger.critical(msg, stacklevel=2, extra=extra)

def exception(msg, extra=None):
    """记录 ERROR 日志并附带当前异常堆栈。

    原理: 依赖 ``logging.exception`` 在 except 上下文中读取当前异常信息；
    调用点必须位于异常处理分支内，否则不会产生 traceback。
    """
    _root_logger.exception(msg, stacklevel=2, extra=extra)


__all__ = ['get_logger', 'debug', 'info', 'warning', 'error', 'critical', 'exception', 'make_extra_tags']
