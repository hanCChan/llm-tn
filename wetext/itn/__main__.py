# =============================================================================
# llm-tn source header
# Path: wetext/itn/__main__.py
# 职责: Python 包的命令行启动入口
# 核心原理: 逆文本规整（ITN）：把口语形式通过 tagger/verbalizer FST 转回书面或结构化形式。
# =============================================================================

from itn.main import main

if __name__ == "__main__":
    main()
