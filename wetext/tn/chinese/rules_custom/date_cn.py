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

from pynini import accep, string_file
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class DateCN(Processor):
    """中文"年月日/号"格式日期(公共 Date 只认符号分隔 2004/09/01, 不认中文字分隔)。

    - `2004年09月01日` -> 二零零四年九月一日: 年份逐位(含中间零)、月/日去前导零。
    - `08月07号` -> 八月七号: "号"是"日"的口语说法, 原样保留(不转"日")。
    - 配合 range_tagger, `年月日-年月日` / `年月~年月` 的 -/~ 读"到":
      `2004年09月01日-2008年06月08日` -> `...九月一日到二零零八年六月八日`。

    实现: month 复用 m/mm 的 tsv(值已含"月"、无前导零, '01'->'一月'); day 先从 d/dd
    的 tsv 值剥掉固定的"日"得纯数字读法, 再拼回输入原样的"日"/"号"单位字, 使单位跟随
    输入。仅供 NormalizerTime(DATE 场景)使用, 不改公共 rules/date.py。
    """

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

        self.year = (yyyy + insert("年")).optimize()
        self.month = (m | mm).optimize()
        # 纯数字读法(剥掉 tsv 值固定的"日") + 原样保留输入的"日"/"号"。
        day_num = ((d | dd) @ self.build_rule(delete("日"), r="[EOS]")).optimize()
        self.day = (day_num + (accep("日") | accep("号"))).optimize()

        year = self.tag_field("year", self.year) + delete("年")
        month = self.tag_field("month", self.month) + delete("月")
        day = self.tag_field("day", self.day)
        date = ((year + insert(" ") + month + (insert(" ") + day).ques)
                | (month + (insert(" ") + day).ques))
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
