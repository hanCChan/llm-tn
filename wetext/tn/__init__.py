# =============================================================================
# llm-tn source header
# Path: wetext/tn/__init__.py
# 职责: 包入口：wetext/tn
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from tn.alignment import AlignmentError, NormalizationMapping, NormalizationResult

__all__ = ["AlignmentError", "NormalizationMapping", "NormalizationResult"]
