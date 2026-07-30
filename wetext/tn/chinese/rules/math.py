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

from pynini import cross, string_file
from pynini.lib.pynutil import add_weight, delete

from tn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Math(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="math")
        self.cardinal = cardinal or Cardinal()
        self.math = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        operator = string_file(get_abs_path("chinese/data/math/operator.tsv"))
        unambiguous_operator = string_file(get_abs_path("chinese/data/math/operator_unambiguous.tsv"))
        # "/" 在算式里当除号读作 "除以"(a/b -> a除以b, 顺序读无需换序)。"/" 高度歧义
        # (分数线/月日/每/比分), 但定类由上游(LLM)决定; 一旦走到 math 即按算式除号读。
        # 纯分数 '1/2' 单独出现时仍被 fraction 以更优权重抢走(读 '二分之一'); 只有当 "/"
        # 处于算式(带 =、其它运算符等 fraction 匹配不全的上下文)时才归 math, 从而修掉
        # '1/2=0.5' 被切成 '一/二等于零点五' 的破碎输出。
        symbols = (cross("~", "到") | cross(":", "比") | cross("<", "小于")
                   | cross(">", "大于") | cross("/", "除以"))

        # 操作数可携带单位, 复用 measure 的单位表(units_en 按表转写, units_zh 原样)。
        # 否则 '3×60=180千米' 里 math 只吃到 '=1', '80千米' 归 measure, 180 被从中间
        # 切成 '一'+'八十千米'。让单位吸附进操作数后, '180千米' 整体读作 '一百八十千米'。
        units_en = string_file(get_abs_path("chinese/data/measure/units_en.tsv"))
        units_zh = string_file(get_abs_path("chinese/data/measure/units_zh.tsv"))
        units = add_weight((cross("k", "千") | cross("w", "万")), 0.1).ques + (units_en | units_zh)

        number = self.cardinal.number
        operand = number + (delete(" ").ques + units).ques
        # 注: 用 .star 而非 .plus。运算符出现 0 次时 math 退化成"单个 operand", 因此裸数字
        # (如 '23')也由 math 匹配读出——全系统裸数字都依赖此(cardinal.tagger 只认 IP/电话
        # 等特殊格式, 不认裸数字)。改 .plus 会导致裸数字失读。'数字+单位'该归 measure 的,
        # 靠 measure 的更优权重争取(见 normalizer 里 measure 权重), 而非在此限制 math。
        self.math = (operand + (delete(" ").ques + (operator | symbols) + delete(" ").ques + operand).star
                     | unambiguous_operator).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.math))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.math + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
