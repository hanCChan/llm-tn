# -*- coding: utf-8 -*-

# =============================================================================
# llm-tn source header
# Path: logger/formatter.py
# 职责: 日志格式化器模块
# 核心原理: 输出可检索的结构化 JSON 日志，并通过白名单字段控制记录内容。
# =============================================================================

"""日志格式化器模块"""

import json
import logging
from datetime import datetime

from .extra_tags import extra_tags

class JsonFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def __init__(self, time_fmt='%Y-%m-%d %H:%M:%S'):
        """保存时间格式，并初始化标准 ``logging.Formatter`` 基类。"""
        super().__init__()
        self.time_fmt = time_fmt
    
    def format(self, record):
        """把一条 LogRecord 编码成单行 JSON。

        原理: 固定字段承载时间、级别、位置、进程和正文；仅白名单
        ``extra_tags`` 中的动态字段会进入输出，避免任意对象导致序列化失败。
        序列化异常时仍返回合法 JSON，并附带错误原因，保证写日志链路不中断。
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
