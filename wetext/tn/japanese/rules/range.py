# Copyright (c) 2026, WENET COMMUNITY.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# llm-tn source header
# Path: wetext/tn/japanese/rules/range.py
# 职责: 定义 Range
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from pynini import accep, cross
from pynini.lib.pynutil import delete

from tn.processor import Processor


class Range(Processor):
    """Tags Japanese range separators and verbalizes their spoken form."""

    def __init__(self, input_normalizer=None):
        super().__init__(name="range")
        self.range = self.apply_input_processor(
            cross("-", "から") | cross("~", "から") | accep("から"),
            input_normalizer,
        )
        self.tagger = self.add_tokens(self.tag_field("value", self.range))
        verbalizer = delete('value: "') + self.range + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
