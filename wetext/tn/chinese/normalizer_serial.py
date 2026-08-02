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

from pynini.lib.pynutil import delete

from tn.chinese.rules.char import Char
from tn.chinese.rules.postprocessor import PostProcessor
from tn.chinese.rules.range import Range
from tn.processor import Processor, RuleSpec

from tn.chinese.rules_custom.address import Address

TOKEN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["denominator", "numerator"],
    "measure": ["denominator", "numerator", "value"],
    "money": ["value", "currency"],
    "time": ["noon", "hour", "minute", "second"],
}


class NormalizerSerial(Processor):

    def __init__(
        self,
        cache_dir=None,
        overwrite_cache=False,
        remove_interjections=True,
        remove_erhua=True,
        traditional_to_simple=True,
        remove_puncts=False,
        full_to_half=True,
        tag_oov=False,
    ):
        super().__init__(name="zh_normalizer_serial", token_orders=TOKEN_ORDERS)
        self.remove_interjections = remove_interjections
        self.remove_erhua = remove_erhua
        self.traditional_to_simple = traditional_to_simple
        self.remove_puncts = remove_puncts
        self.full_to_half = full_to_half
        self.tag_oov = tag_oov
        self.build_fst(
            "zh_tn_serial",
            cache_dir,
            overwrite_cache,
            {
                "full_to_half": self.full_to_half,
                "remove_erhua": self.remove_erhua,
                "remove_interjections": self.remove_interjections,
                "remove_puncts": self.remove_puncts,
                "tag_oov": self.tag_oov,
                "traditional_to_simple": self.traditional_to_simple,
            },
        )

    def build_tagger_and_verbalizer(self):
        range_rule = Range()
        # SERIAL 与 ADDRESS 共用 Address 规则(它是 Serial 逐位能力的超集: 逐位段、
        # 连字符段读"杠"、数字+门牌单位整数读)。SERIAL 的 span 不含门牌单位, 故不受
        # "数字+单位整数读"分支影响, 仍全逐位。
        serial = Address()
        char = Char()

        rules = (
            RuleSpec(serial, 1.05),
            RuleSpec(char, 100),
            RuleSpec(range_rule),
        )
        tagger = self.tagger_union(rules).star
        self.tagger = tagger @ self.build_rule(delete(" "), r="[EOS]")

        verbalizer = self.verbalizer_union(rules)

        postprocessor = PostProcessor(
            remove_interjections=self.remove_interjections,
            remove_puncts=self.remove_puncts,
            full_to_half=self.full_to_half,
            tag_oov=self.tag_oov,
            traditional_to_simple=self.traditional_to_simple,
        ).processor
        self.verbalizer = (verbalizer @ postprocessor).star
