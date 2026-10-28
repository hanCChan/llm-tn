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
# Path: wetext/tn/chinese/rules_custom/address.py
# 职责: 定义 Address
# 核心原理: 文本规整（TN）：用 pynini 构造 tagger 与 verbalizer FST，将书面形式转成可读文本。
# =============================================================================

from pynini import cross, string_file
from pynini.lib.pynutil import add_weight, delete

from tn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Address(Processor):
    """门牌地址: 按现实口语读法, 单位决定其前数字读整数还是逐位。

    - 整数单位(号/楼/栋/幢/座/单元/层/弄/段…): 21号楼 -> 二十一号楼, 88号 -> 八十八号
    - 逐位单位(室): 502室 -> 五零二室, 2106室 -> 二一零六室
    - 连字符分段号(3-1-3805 / 103-3 / 1137-1128号): 整段逐位, "-" 读"杠",
      即便后面跟单位也整段逐位(1128号 -> 一一二八号, 不读一百二十八)
    - 无单位的裸数字尾段: 逐位(房间号 506 -> 五零六)
    字母及未登记中文(A座/区/之等)由 normalizer 里的 Char 规则逐字兜底。

    权重: 数字+单位/连字符段(0.1) 优于裸数字(0.5), 避免 '15层' 被拆成
    逐位的 '一五' + char '层'; 让它整段走 chunk_card 读 '十五层'。
    """

    def __init__(self, cardinal=None):
        super().__init__(name="address")
        self.cardinal = cardinal or Cardinal()
        self.address = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        number = self.cardinal.number      # 88 -> 八十八 (整数)
        digits = self.cardinal.digits      # 8 -> 八 (逐位)
        card_unit = string_file(get_abs_path("chinese/data/address/units_cardinal.tsv"))
        digit_unit = string_file(get_abs_path("chinese/data/address/units_digit.tsv"))

        # 连字符分段号: 整体逐位 + "-"读"杠"; 后可带单位(原样)。优先级最高。
        seg = digits.plus + (cross("-", "杠") + digits.plus).plus + (card_unit | digit_unit).ques
        chunk_card = number + card_unit    # 88号 -> 八十八号
        chunk_digit = digits.plus + digit_unit  # 502室 -> 五零二室
        bare = digits.plus                 # 裸数字尾段逐位

        piece = (add_weight(seg, 0.1) | add_weight(chunk_card, 0.1)
                 | add_weight(chunk_digit, 0.1) | add_weight(bare, 0.5))
        self.address = (piece + piece.star).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.address))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.address + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
