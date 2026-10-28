# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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
# Path: wetext/tn/chinese/rules_custom/score.py
# 职责: 定义 Score
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from pynini import cross, union
from pynini.lib.pynutil import delete

from tn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor


class Score(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="score")
        self.cardinal = cardinal or Cardinal()
        self.score = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        # 比分: "3:1"/"2-1"/"3/1" -> 三比一。分隔符统一读作 "比"。
        # span 已由 LLM 判为 SCORE, 两侧直接复用 cardinal.number 以正确读多位数 (30 -> 三十)。
        symbol = union(":", "-", "/")
        number = self.cardinal.number
        rmspace = delete(" ").ques
        self.score = (number + rmspace + cross(symbol, "比") + rmspace + number).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.score))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.score + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
