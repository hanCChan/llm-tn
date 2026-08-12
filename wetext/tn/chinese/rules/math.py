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

from pynini import accep, cross, string_file, union
from pynini.lib.pynutil import add_weight, delete, insert

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
        # "/" 在算式里当除号读作 "除以"(1/2 -> 一除以二, 顺序读无需换序)。"/" 高度歧义
        # (分数线/月日/每/比分), 但定类由上游(LLM)决定; 一旦走到 math 即按算式除号读。
        # 纯分数 '1/2' 单独出现时仍被 fraction 以更优权重抢走(读 '二分之一'); 只有当 "/"
        # 处于算式(带 =、其它运算符等 fraction 匹配不全的上下文)时才归 math, 从而修掉
        # '1/2=0.5' 被切成 '一/二等于零点五' 的破碎输出。
        slash = cross("/", "除以")
        symbols = (cross("~", "到") | cross(":", "比") | cross("<", "小于")
                   | cross(">", "大于"))

        # 操作数可携带单位, 复用 measure 的单位表(units_en 按表转写, units_zh 原样)。
        # 否则 '3×60=180千米' 里 math 只吃到 '=1', '80千米' 归 measure, 180 被从中间
        # 切成 '一'+'八十千米'。让单位吸附进操作数后, '180千米' 整体读作 '一百八十千米'。
        units_en = string_file(get_abs_path("chinese/data/measure/units_en.tsv"))
        units_zh = string_file(get_abs_path("chinese/data/measure/units_zh.tsv"))
        units = add_weight((cross("k", "千") | cross("w", "万")), 0.1).ques + (units_en | units_zh)

        number = self.cardinal.number
        # 操作数 = 前扩展符* + 核 + 后扩展符*
        #   前扩展符  √(根号) sin cos tan   —— 可叠加(√sin)
        #   核        数字(可带单位) 或 字母/常数簇(mc kx iπ, 并列表隐式相乘)
        #   后扩展符  ²³(上标) ^n(幂) °(度) —— 可叠加
        # 原本核只能是数字, 非数字会打断运算符链, 使 'a²+b²=c²' 逐字符原样输出、
        # '11²=121' 读成 '十一²=一百二十一'。放宽成代数项后整条式子才能连起来读。
        prefix_op = string_file(get_abs_path("chinese/data/math/prefix.tsv"))
        superscript = string_file(get_abs_path("chinese/data/math/superscript.tsv"))
        constant = string_file(get_abs_path("chinese/data/math/constant.tsv"))
        greek = string_file(get_abs_path("chinese/data/math/greek.tsv"))
        letter = union(*[accep(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"])

        # 字母/常数可并列(隐式相乘), 但数字与数字绝不能相邻 —— 否则 '121' 会被切成
        # '12'+'1' 读出"十二一"。故系数只允许 number 单向前缀(2x)。
        # 簇长限 3: 真实代数变量簇不超过 3 个字母(mc kx iπ abc), 更长的字母串是英文单词
        # 而非变量, 不该被当代数式吞掉('iPhone=x' 应原样保留而非读成变量簇)。
        # 注: sin/cos 等函数名走 prefix_op, 不占这里的额度。
        atom = letter | constant | greek
        cluster = atom | atom**2 | atom**3
        core = (number + (delete(" ").ques + units).ques) | (number + cluster) | cluster

        # 运算符分两档:
        #   num_op  纯数字链用, 含 "/" —— '1/2=0.5' 依赖它
        #   op      有字母参与时用, 不含 "/" —— "/" 在中文文本里绝大多数不是算式除号, 而是
        #           单位(km/h g/cm³)、分数、月日、比分或路径。字母能当操作数后 'g/cm³' 就成
        #           了合法算式被读成"g除以cm的立方"; 代数式真要表除法写 ÷ 或分数线。
        num_op = delete(" ").ques + (operator | symbols | slash) + delete(" ").ques
        op = delete(" ").ques + (operator | symbols) + delete(" ").ques

        # 指数可以是括号算式: 2^(3+4) -> 二的三加四次方, e^(iπ) -> e的i派次方。
        # 指数层用 core 而非完整操作数 —— 否则 caret -> 操作数 -> suffix_op -> caret
        # 自引用无法构造。代价是 2^(3^4) 指数里再套幂不支持, 真实文本可忽略。
        exp_expr = core + (op + core).star
        caret = (insert("的") + delete("^")
                 + (core | (delete("(") + exp_expr + delete(")")))
                 + insert("次方"))
        # 上标不叠加(a²³ 无意义), 故用 .ques 而非 .star
        suffix_op = (superscript | caret | cross("°", "度")).ques

        # f(x) -> fx: 函数名紧跟括号参数, 括号不读。函数名限 2 个字母 —— f/g/h 这类匿名
        # 函数是单字母, ln/lg/Re 这类双字母也逐字母读出; 而 3 字母以上的是具名函数
        # (sin cos tan), 该走 prefix_op 读成"正弦/余弦"。限长后 'cos(60)' 结构上就进不了
        # 这条路径, 不必靠权重去争。
        call = (atom | atom**2) + delete("(") + core + delete(")")
        # 分组符 ( ) | 一律不读, 只标示作用范围, 因此平铺匹配即可 —— 不必为它们展开一层
        # 嵌套表达式。'(a+b)²' 由操作数 '(a' 和 'b)²' 拼出: 虽然 ² 结构上只挂在 b 上而非
        # 整个分组, 但括号不读, 两种解析读出的字串完全相同, 听者无从区分。
        # 右括号在后扩展符之前 —— '(a+b)²' 的写法是 ')' 在 '²' 前。
        open_grp = (delete("(") | delete("|")).ques
        close_grp = (delete(")") | delete("|")).ques
        operand = (open_grp + prefix_op.star + open_grp + (call | core)
                   + close_grp + suffix_op + close_grp)
        # 纯数字链: 操作数只能是"数字(可带单位)", 运算符含 "/"。这是本规则的原有行为,
        # 裸数字也靠它读出 —— 注: 用 .star 而非 .plus, 运算符出现 0 次时退化成单个操作数,
        # 因此裸数字(如 '23')也由 math 匹配读出(cardinal.tagger 只认 IP/电话等特殊格式,
        # 不认裸数字)。改 .plus 会导致裸数字失读。'数字+单位'该归 measure 的, 靠 measure 的
        # 更优权重争取(见 normalizer 里 measure 权重), 而非在此限制 math。
        num_operand = number + (delete(" ").ques + units).ques
        numeric = num_operand + (num_op + num_operand).star
        # 代数链: 操作数放宽到字母/常数/根号/函数/分组, 运算符不含 "/"。
        # 用 .plus —— 代数式一定含运算符(a²+b²=c², x+1=2, E=mc²); 而纯字母簇单独出现时
        # 不是代数而是单位(cm³ m²)、缩写或型号, 若允许运算符 0 次, 'g/cm³' 会被切成 'g'
        # 和 'cm³' 两个单操作数, 把 ³ 读成"的立方"。裸数字的 0 运算符退化路径由上面的
        # numeric 链(.star)保留, 两者互不影响。
        algebraic = operand + (op + operand).plus

        self.math = (numeric | algebraic | unambiguous_operator).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.math))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.math + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
