# =============================================================================
# llm-tn source header
# Path: wetext/tn/__main__.py
# 职责: Python 包的命令行启动入口
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from tn.main import main

if __name__ == "__main__":
    main()
