// =============================================================================
// llm-tn source header
// Path: wetext/runtime/android/app/src/main/java/com/wenet/WeTextProcessing/WeTextProcessing.kt
// 职责: Android/JNI 客户端或 Gradle 构建源码
// 核心原理: 遵循仓库现有的 A1 类别 token 与 pynini FST 管线，注释仅解释实现，不改变运行行为。
// =============================================================================

package com.wenet.WeTextProcessing

object WeTextProcessing {

    init {
        System.loadLibrary("wetextprocessing")
    }

    @JvmStatic
    external fun init(modelDir: String)

    @JvmStatic
    external fun normalize(input: String): String

    @JvmStatic
    external fun inverse_normalize(input: String): String
}
