# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: logger/core.py
# 职责: 日志核心模块
# 核心原理: 输出可检索的结构化 JSON 日志，并通过白名单字段控制记录内容。
# =============================================================================

"""按白名单输出结构化 JSON 的服务日志核心。

总体怎么做
==========
1. ``setup_logger`` 创建 ``app`` logger 和 100MB 滚动文件 handler。
2. ``JsonFormatter`` 把固定字段和 requestid/sdkappid 白名单编码成单行 JSON。
3. 默认同步模式直接写文件；``enable_async`` 可把 handler 移到 QueueListener。
4. 异步模式下业务线程只执行 ``QueueHandler.put``，JSON 格式化和磁盘 I/O
   在后台线程完成。
5. ``stop_async`` 先 drain 队列，再恢复同步 handler，保证退出日志落盘。

核心函数
========
* ``setup_logger``: 幂等创建 logger 和滚动文件 handler。
* ``get_logger``: 返回根 logger 或 ``app.<name>`` 子 logger。
* ``enable_async``: 切换到 QueueHandler + QueueListener。
* ``stop_async``: 排空异步队列并恢复同步 handler。

函数如何协作
============
``get_logger`` -> ``setup_logger`` 创建 handler；``enable_async`` 把 handler
移到 listener 线程；业务日志经 ``QueueHandler`` 入队；``stop_async`` 排空队列
并恢复原 handler。格式化和字段白名单由 ``JsonFormatter`` 完成。

关键参数
========
* ``_LOG_FILE``: 相对工作目录的日志路径，默认 ``log/app.log``。
* ``_LOG_LEVEL``: 固定 INFO 级。
* ``_LOG_MAX_MB/_LOG_BACKUP``: 单文件 100MB，保留 5 份。
* ``get_logger(name)``: 非空时返回 ``app.<name>``，仍向 app logger 传播。

注意:
    异步队列是无界的；极端磁盘阻塞时可能增加内存。进程被 SIGKILL 时，
    尚未 drain 的队列记录无法保证落盘。
"""

import logging
import os
from typing import Optional

from concurrent_log_handler import ConcurrentRotatingFileHandler

from .formatter import JsonFormatter


_LOG_FILE = "log/app.log"
_LOG_LEVEL = logging.INFO
_LOG_MAX_MB = 100
_LOG_BACKUP = 5

_logger: Optional[logging.Logger] = None
_configured = False
_listener = None
_async_handlers = None


def setup_logger() -> logging.Logger:
    """初始化并返回 ``app`` logger。

    返回:
        配置了 JSON 滚动文件 handler 的 ``app`` logger；重复调用直接复用。

    副作用:
        创建日志目录并清空 ``app`` logger 上已有的 handler。
    """
    global _logger, _configured
    if _configured and _logger is not None:
        return _logger

    _logger = logging.getLogger("app")
    _logger.setLevel(_LOG_LEVEL)
    _logger.handlers.clear()

    log_file = os.path.abspath(_LOG_FILE)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = ConcurrentRotatingFileHandler(
        log_file,
        maxBytes=_LOG_MAX_MB * 1024 * 1024,
        backupCount=_LOG_BACKUP,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    _logger.addHandler(file_handler)

    _logger.propagate = False
    _configured = True
    return _logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """返回根 logger 或命名 logger。

    参数:
        name: 为空返回 ``app``；否则返回 ``app.<name>``，并向上传播到根 logger。
    """
    root = setup_logger()
    if name is None:
        return root
    return logging.getLogger(f"app.{name}")


def enable_async() -> None:
    """启用异步文件日志；重复调用是幂等的。

    原始文件 handler 从 logger 移到 QueueListener，业务线程只负责把 LogRecord
    放入无锁队列。原 handler 暂存在模块级变量，供 :func:`stop_async` 恢复。

    该切换只改变日志 I/O 执行位置，不改变日志级别、格式或白名单字段。
    """
    import queue
    from logging.handlers import QueueHandler, QueueListener

    global _listener, _async_handlers
    root = setup_logger()
    if _listener is not None:
        return
    q = queue.SimpleQueue()
    handlers = list(root.handlers)
    _async_handlers = handlers
    root.handlers.clear()
    root.addHandler(QueueHandler(q))
    _listener = QueueListener(q, *handlers, respect_handler_level=True)
    _listener.start()


def stop_async() -> None:
    """停止异步 listener、排空日志队列，并恢复同步文件 handler。

    必须在服务退出阶段调用；队列 drain 完成后，生命周期末尾的“已退出”
    日志仍会通过恢复后的文件 handler 落盘。
    """
    global _listener, _async_handlers
    if _listener is None:
        return

    # QueueListener.stop() 会等待队列中的记录处理完；随后恢复文件 handler，
    # 让 lifespan 里最后一条“已退出”日志也能正常写入。
    _listener.stop()
    root = setup_logger()
    root.handlers.clear()
    if _async_handlers:
        root.handlers.extend(_async_handlers)
    _async_handlers = None
    _listener = None
