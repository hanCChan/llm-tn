# =============================================================================
# llm-tn source header
# Path: wetext/runtime/bindings/android/CMakeLists.txt
# 职责: CMake 构建入口
# 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
# =============================================================================

# Android JNI binding. Included by the top-level runtime/CMakeLists.txt only when
# configuring with the NDK toolchain (if(ANDROID)). It reuses the parent's
# openfst/glog/gflags and the wetext_processor/wetext_utils targets, so it just
# defines the JNI shared library and how to install it.
add_library(wetextprocessing SHARED ${CMAKE_CURRENT_SOURCE_DIR}/wetextprocessing.cc)

target_link_libraries(wetextprocessing PRIVATE wetext_processor)

# Install ONLY this target under its own component, so
#   cmake --install <dir> --component jni
# drops the single libwetextprocessing.so straight into CMAKE_INSTALL_PREFIX
# (set per-ABI to .../jniLibs/<abi> by the presets). Using a dedicated component
# avoids pulling in the install rules that glog/gflags register via FetchContent,
# which would otherwise litter jniLibs with headers and static libs.
install(TARGETS wetextprocessing
  LIBRARY DESTINATION "." COMPONENT jni
  RUNTIME DESTINATION "." COMPONENT jni)
