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
# Path: wetext/tn/chinese/rules_custom/phone.py
# 职责: 定义 Phone
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from pynini import cross
from pynini.lib.pynutil import delete

from tn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor


class Phone(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="phone")
        self.cardinal = cardinal or Cardinal()
        self.phone = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        # 逐位读; 电话号里的分隔符 "-"、括号 "()"、国际前缀 "+"、空格一律删除
        # (它们标记区号/国家码/分段, 不发音), 只留连续数字。整串拼好后再把 一->幺
        # (电话习惯)。注意: rewrite 必须作用于整串, 否则首段的 "1" 不会变 "幺"。
        digits = self.cardinal.digits
        sep = (delete("-") | delete("(") | delete(")") | delete("+") | delete(" ")).star
        number = sep + digits.plus + (sep + digits.plus).star + sep
        self.phone = (number @ self.build_rule(cross("一", "幺"))).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.phone))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.phone + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
