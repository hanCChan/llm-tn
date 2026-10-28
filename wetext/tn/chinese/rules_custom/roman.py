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
# Path: wetext/tn/chinese/rules_custom/roman.py
# 职责: 定义 Roman
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from pynini import string_file
from pynini.lib.pynutil import delete

from tn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Roman(Processor):
    """罗马数字 + 序数单位, 数字读整数(与阿拉伯数字的 21世纪 读法一致)。

        XX世纪   -> 二十世纪
        XXI世纪  -> 二十一世纪
        第VII章  -> 第七章
        第IX届   -> 第九届
        Ⅹ世纪    -> 十世纪
    后缀(初/末/中叶/20年代)不在本规则内, 由 normalizer 的其余规则兜底:
        XX世纪初     -> 二十世纪初
        XX世纪30年代 -> 二十世纪三十年代

    **右邻单位是本规则唯一的消歧手段**: 裸罗马数字与英文缩写无法区分(CD/MV/LV/MIX/
    XL 都是合法罗马数字), 故强制要求紧跟 units.tsv 中的单位才匹配, 其余位置的罗马
    数字一律不转(CD唱片/MV/XL码 保持原样)。单位表只收与序数强绑定的词, 刻意不收
    型/级/类/度 —— 中文里 C级/D类/CD型/LV级 极常见, 收了会大面积误读。
    """

    def __init__(self, cardinal=None):
        super().__init__(name="roman")
        self.cardinal = cardinal or Cardinal()
        self.roman = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        # 罗马数字 -> 阿拉伯数字 -> 中文整数, 复用 cardinal 以正确读多位数。
        arabic = string_file(get_abs_path("chinese/data/roman/roman_to_arabic.tsv"))
        units = string_file(get_abs_path("chinese/data/roman/units.tsv"))
        self.roman = ((arabic @ self.cardinal.number) + units).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.roman))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.roman + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
