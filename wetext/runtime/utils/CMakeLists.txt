# =============================================================================
# llm-tn source header
# Path: wetext/runtime/utils/CMakeLists.txt
# 职责: CMake 构建入口
# 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
# =============================================================================

add_library(wetext_utils STATIC wetext_string.cc)

target_link_libraries(wetext_utils PUBLIC glog)
