// =============================================================================
// llm-tn source header
// Path: wetext/runtime/android/build.gradle.kts
// 职责: Android/JNI 客户端或 Gradle 构建源码
// 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
// =============================================================================

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.compose.compiler) apply false
}
