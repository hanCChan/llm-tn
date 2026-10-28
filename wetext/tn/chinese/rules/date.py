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
# Path: wetext/tn/chinese/rules/date.py
# 职责: 定义 Date
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from pynini import string_file
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Date(Processor):

    def __init__(self, range_tagger=None):
        super().__init__(name="date")
        self.range_tagger = range_tagger
        self.year = None
        self.month = None
        self.day = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        digit = string_file(get_abs_path("chinese/data/number/digit.tsv"))
        zero = string_file(get_abs_path("chinese/data/number/zero.tsv"))

        yyyy = digit + (digit | zero)**3
        m = string_file(get_abs_path("chinese/data/date/m.tsv"))
        mm = string_file(get_abs_path("chinese/data/date/mm.tsv"))
        d = string_file(get_abs_path("chinese/data/date/d.tsv"))
        dd = string_file(get_abs_path("chinese/data/date/dd.tsv"))
        rmsign = (delete("/") | delete("-") | delete(".")) + insert(" ")

        self.year = (yyyy + insert("年")).optimize()
        self.month = (m | mm).optimize()
        self.day = (d | dd).optimize()

        year = self.tag_field("year", self.year)
        month = self.tag_field("month", self.month)
        month_two_digit = self.tag_field("month", mm)
        day = self.tag_field("day", self.day)

        # yyyy/m/d | dd/mm/yyyy | mm/dd/yyyy(美式 M/D/Y)
        # yyyy/0m | 0m/yyyy | 0m/dd
        # 歧义主要靠"月段必须匹配 mm(01-12)"自动消解: 10/20/2018 里 20 不能当月, 只能
        # 走 M/D/Y(20日); 20/10/2018 里 20 不能当月, 只能 D/M/Y。仅 xx/xx/yyyy 两段都
        # ≤12 时(如 01/02/2018)才真歧义, 由 union 顺序定(此处 D/M/Y 在前), 样本罕见。
        date = ((year + rmsign + month + rmsign + day)
                | (day + rmsign + month + rmsign + year)
                | (month + rmsign + day + rmsign + year)
                | (year + rmsign + month_two_digit)
                | (month_two_digit + rmsign + year)
                | (month_two_digit + rmsign + day))
        tagger = self.add_tokens(date)

        if self.range_tagger is None:
            self.tagger = tagger
        else:
            self.tagger = tagger + (self.range_tagger + tagger).ques

    def build_verbalizer(self):
        year = delete('year: "') + self.year + delete('" ')
        month = delete('month: "') + self.month + delete('"')
        day = delete(' day: "') + self.day + delete('"')
        verbalizer = year.ques + month + day.ques
        self.verbalizer = self.delete_tokens(verbalizer)
