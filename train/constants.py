# =============================================================================
# llm-tn source header
# Path: train/constants.py
# 职责: A1 训练与推理共用的 token 协议常量
# 核心原理: 复用 Qwen 已有的 FIM/EOS token 作为提示边界和停止符，避免额外扩张底座词表。
# =============================================================================

"""定义 A1 全链路共用的提示模板、停止符和兼容分隔符。

训练数据构造、命令行推理和 vLLM 服务都从这里取 token id，避免某一端
硬编码不同的 special token。``SEPARATOR`` 仅保留给 v2 文本标签协议兼容。
"""

MODEL_ID = "Qwen/Qwen3-0.6B-Base"

TN_PREFIX = "<|fim_prefix|>"   # 充当 <|tn|>,id 151659
SEP = "<|fim_suffix|>"         # 充当 <|sep|>,id 151661
EOS = "<|endoftext|>"          # id 151643
TN_PREFIX_ID = 151659
SEP_ID = 151661
EOS_ID = 151643

SEPARATOR = "\t"
