# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: logger/core.py
# 职责: 日志核心模块
# 核心原理: 输出可检索的结构化 JSON 日志，并通过白名单字段控制记录内容。
# =============================================================================

"""日志核心模块。

写死默认值：INFO 级别，JSON 格式，滚动写入 `log/app.log`，仅文件输出。
目录名与 Dockerfile / start.sh 里创建的 `$WORKSPACE/log` 保持一致。

线上服务可调用 :func:`enable_async` 把 JSON 序列化和文件写盘移到后台线程，
避免每次请求日志阻塞 asyncio 事件循环。停止时 QueueListener 会先 drain 队列，
随后恢复原始文件 handler，保证退出日志仍能落盘。
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
    """初始化并返回 ``app`` logger；重复调用不会重复添加 handler。"""
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
    """返回根 logger 或带 ``app.`` 前缀的命名 logger。"""
    root = setup_logger()
    if name is None:
        return root
    return logging.getLogger(f"app.{name}")


def enable_async() -> None:
    """启用异步文件日志；重复调用是幂等的。

    原始文件 handler 从 logger 移到 QueueListener，业务线程只负责把 LogRecord
    放入无锁队列。原 handler 暂存在模块级变量，供 :func:`stop_async` 恢复。
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
    """停止异步 listener、排空日志队列，并恢复同步文件 handler。"""
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
