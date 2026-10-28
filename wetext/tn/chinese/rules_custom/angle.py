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
# Path: wetext/tn/chinese/rules_custom/angle.py
# 职责: 定义 Angle
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from pynini import cross
from pynini.lib.pynutil import delete

from tn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor


class Angle(Processor):
    """经纬度 / 角度: 度分秒结构 度[分[秒]]。

    °->度, ′->分, ″->秒, 数字读整数(非逐位):
        31°14′      -> 三十一度十四分
        22°15′0″    -> 二十二度十五分零秒
        90°         -> 九十度
    前缀"东经/西经/北纬/南纬"由 normalizer 的 Char 规则兜底, 不在本规则内。
    单独成规则以免 ′/″ 污染通用 measure 单位表(它们也可能是撇号/英尺英寸)。
    """

    def __init__(self, cardinal=None):
        super().__init__(name="angle")
        self.cardinal = cardinal or Cardinal()
        self.angle = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        number = self.cardinal.number
        degree = number + cross("°", "度")
        minute = number + cross("′", "分")
        second = number + cross("″", "秒")
        self.angle = (degree + minute.ques + second.ques).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.angle))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.angle + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
