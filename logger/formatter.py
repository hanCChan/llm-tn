# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: logger/formatter.py
# 职责: 日志格式化器模块
# 核心原理: 输出可检索的结构化 JSON 日志，并通过白名单字段控制记录内容。
# =============================================================================

"""把 LogRecord 编码为单行、可检索的 JSON 日志。

总体怎么做
==========
    ``logging`` -> ``JsonFormatter.format`` -> 固定字段 + 白名单 extra
    -> ``json.dumps`` -> 文件 handler。

核心函数
========
    * ``JsonFormatter.__init__``: 设置时间格式。
    * ``JsonFormatter.format``: 组装字段、附加 traceback，并保证总能返回 JSON。

函数如何协作
============
    ``logging`` 创建 LogRecord 后调用 ``format``；本类读取
    ``extra_tags`` 白名单筛出动态字段，再合并固定字段和异常堆栈，最后交给
    文件 handler 写入。

关键参数
========
    * ``time_fmt``: 日志时间格式，默认精确到秒，毫秒单独追加。
    * ``record``: Python LogRecord；只复制 ``extra_tags`` 白名单中的动态字段。
"""

import json
import logging
from datetime import datetime

from .extra_tags import extra_tags

class JsonFormatter(logging.Formatter):
    """将 Python LogRecord 转成一行 JSON 的 formatter。"""
    
    def __init__(self, time_fmt='%Y-%m-%d %H:%M:%S'):
        """初始化标准 Formatter。

        参数:
            time_fmt: ``datetime.strftime`` 使用的时间格式。
        """
        super().__init__()
        self.time_fmt = time_fmt
    
    def format(self, record):
        """把一条 LogRecord 编码成单行 JSON。

        原理: 固定字段承载时间、级别、位置、进程和正文；仅白名单
        ``extra_tags`` 中的动态字段会进入输出，避免任意对象导致序列化失败。
        序列化异常时仍返回合法 JSON，并附带错误原因，保证写日志链路不中断。

        参数:
            record: logging 产出的 LogRecord。

        返回:
            不含换行的 JSON 字符串，可直接写入日志文件。
        """
        log_data = {
            'timestamp': f"{datetime.fromtimestamp(record.created).strftime(self.time_fmt)}.{int(record.msecs):03d}",
            'level': record.levelname,
            'file': f"{record.filename}:{record.lineno}",
            'process': record.process,
            'content': record.getMessage(),
        }
        
        for key, value in record.__dict__.items():
            if key in extra_tags:
                log_data[key] = value
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        try:
            return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            error_data = {
                'timestamp': f"{datetime.fromtimestamp(record.created).strftime(self.time_fmt)}.{int(record.msecs):03d}",
                'level': 'ERROR',
                'file': f"{record.filename}:{record.lineno}",
                'process': record.process,
                'content': log_data.get('message', ''),
                'exception': f"JsonFormatter format error: {str(e)}" 
            }
            return json.dumps(error_data, ensure_ascii=False, separators=(',', ':'))
