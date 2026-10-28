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
# Path: wetext/tn/chinese/rules_custom/version.py
# 职责: 定义 Version
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from pynini import cross
from pynini.lib.pynutil import delete

from tn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor


class Version(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="version")
        self.cardinal = cardinal or Cardinal()
        self.version = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        # 版本号逐位读, 分隔符 "." 读作 "点": 1.10.5 -> 一点一零点五。
        digits = self.cardinal.digits
        segment = digits.plus
        self.version = (segment + (cross(".", "点") + segment).star).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.version))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.version + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
