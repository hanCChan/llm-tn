# =============================================================================
# llm-tn source header
# Path: wetext/runtime/processor/CMakeLists.txt
# 职责: CMake 构建入口
# 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
# =============================================================================

add_library(wetext_processor STATIC
  wetext_processor.cc
  wetext_token_parser.cc
)

target_link_libraries(wetext_processor PUBLIC fst wetext_utils)

# ----------------------------------------------------------------------------
# C API shared library (wetext_processor_c)
# ----------------------------------------------------------------------------
add_library(wetext_processor_c SHARED
  wetext_processor_c_api.cc
)

target_link_libraries(wetext_processor_c PUBLIC wetext_processor)
