# =============================================================================
# llm-tn source header
# Path: wetext/runtime/bin/CMakeLists.txt
# 职责: CMake 构建入口
# 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
# =============================================================================

add_executable(processor_main processor_main.cc)
target_link_libraries(processor_main PRIVATE wetext_processor)
target_link_libraries(processor_main PRIVATE gflags glog)
