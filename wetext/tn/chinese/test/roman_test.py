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

import pytest

from tn.chinese.normalizer import Normalizer
from tn.chinese.rules_custom.roman import Roman
from tn.chinese.test.utils import parse_test_case


@pytest.fixture(scope="module")
def normalizer(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("zh_tn_roman")
    return Normalizer(cache_dir=cache_dir, overwrite_cache=True, full_to_half=False)


class TestRoman:

    roman = Roman()
    roman_cases = parse_test_case("data/roman.txt")

    @pytest.mark.parametrize("written, spoken", roman_cases)
    def test_roman(self, written, spoken):
        assert self.roman.normalize(written) == spoken

    @pytest.mark.parametrize(
        "written, spoken",
        [
            ("XX世纪的作品。", "二十世纪的作品。"),
            ("第XX世纪", "第二十世纪"),
            # 后缀由 normalizer 的其余规则兜底, 不属于 roman 规则本身
            ("XX世纪初", "二十世纪初"),
            ("XIX世纪末的欧洲", "十九世纪末的欧洲"),
            ("XX世纪30年代", "二十世纪三十年代"),
            # 序数单位(原 polynorm-bench 的 ROMAN 用例)
            ("第VII章很有趣。", "第七章很有趣。"),
            ("第IX届运动会。", "第九届运动会。"),
            ("第XVIII修正案", "第十八修正案"),
            ("第XII军团", "第十二军团"),
            ("第IV部分内容。", "第四部分内容。"),
            ("第VI代产品。", "第六代产品。"),
            ("第III次尝试。", "第三次尝试。"),
            # 阿拉伯数字世纪的既有读法不受影响
            ("21世纪20年代", "二十一世纪二十年代"),
        ],
    )
    def test_roman_in_sentence(self, normalizer, written, spoken):
        assert normalizer.normalize(written) == spoken

    @pytest.mark.parametrize(
        "written",
        [
            # 均为合法罗马数字, 但未紧跟单位表中的单位, 一律不得转换
            "VIP会员",
            "这是CD唱片",
            "USB接口",
            "MI手机",
            "MMXXIII年。",
            "看了这部MV",
            "买了个LV包",
            # 单位表刻意不收 型/级/类/度: 这些在中文里大量接英文缩写
            "CD型播放器",
            "他是C级选手",
            "属于D类物品",
            "穿XL码",
        ],
    )
    def test_roman_requires_unit_suffix(self, normalizer, written):
        assert normalizer.normalize(written) == written
