"""值文法: 按 (类, 子类) 生成**书写形态多样**的数字表达, 标签由构造决定。

设计原则(泛化红线 R2):
    形态按"这个类在真实中文文本里有哪些书写规范"全量枚举, 不按测试集错例挑。每个子类
    下的形态**等概率**抽样 —— 训练分布不偏向任何评测集的形态比例。数值全随机, 不复用评测值。

用法:
    from forms import FORMS, gen_value
    v, form = gen_value(rng, "PHONE", "landline")   # ('(021)6355 7591', 'paren_area_space')
"""
import random

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

AREA3 = ["010", "020", "021", "022", "023", "024", "025", "027", "028", "029"]
AREA4 = ["0755", "0571", "0512", "0731", "0371", "0532", "0591", "0851", "0898", "0431", "0451", "0991"]
MOBILE_PREFIX = ["13", "14", "15", "16", "17", "18", "19"]
PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
CUR_SYMBOL = [("¥", ""), ("￥", ""), ("$", ""), ("€", ""), ("£", ""), ("HK$", ""), ("US$", "")]
CUR_CODE = ["USD", "EUR", "GBP", "JPY", "HKD", "CNY", "RMB", "SGD", "AUD", "CAD", "CHF", "KRW", "THB"]
# 第 2 轮(07 错误分析: 币种代码+数字被当成 SERIAL 编码): 币种代码按 ISO 4217 常用币种全量列出,
# 并让 SERIAL 字母前缀避开这些代码 —— 否则"KRW10000"在合成数据里既可能是金额又可能是编号, 标签自相矛盾。
CUR_CODE_V2 = CUR_CODE + ["RUB", "INR", "IDR", "MYR", "PHP", "VND", "TWD", "MOP", "NZD", "SEK", "NOK", "DKK",
                          "PLN", "TRY", "BRL", "MXN", "ARS", "ZAR", "AED", "SAR", "ILS", "EGP", "PKR", "BDT",
                          "NGN", "KZT", "CZK", "HUF", "CLP", "COP", "PEN", "QAR", "KWD"]
CUR_NAME = ["美元", "欧元", "英镑", "日元", "港元", "港币", "韩元", "澳元", "加元", "卢布", "泰铢", "新台币",
            "瑞士法郎", "新加坡元", "阿根廷比索", "越南盾"]
CUR_NAME_V2 = CUR_NAME + ["俄罗斯卢布", "印度卢比", "印尼盾", "印度尼西亚盾", "马来西亚林吉特", "林吉特", "菲律宾比索",
                          "新西兰元", "瑞典克朗", "挪威克朗", "丹麦克朗", "波兰兹罗提", "土耳其里拉", "巴西雷亚尔",
                          "墨西哥比索", "南非兰特", "迪拉姆", "沙特里亚尔", "埃及镑", "澳门元", "人民币"]
_CUR_SET = set(CUR_CODE_V2)


def d(rng, n, lead_zero=True):
    s = "".join(rng.choice("0123456789") for _ in range(n))
    if not lead_zero and s[0] == "0":
        s = rng.choice("123456789") + s[1:]
    return s


def num(rng, lo, hi, dec_p=0.0, dec_n=(1, 2)):
    v = str(rng.randint(lo, hi))
    if rng.random() < dec_p:
        v += "." + d(rng, rng.randint(*dec_n))
    return v


def thousands(v):
    head, _, tail = v.partition(".")
    out = f"{int(head):,}"
    return out + ("." + tail if tail else "")


def range_join(rng, a, b, seps=("-", "~", "至", "到", "—")):
    return a + rng.choice(seps) + b


def mobile(rng):
    return rng.choice(MOBILE_PREFIX) + d(rng, 9)


def area(rng):
    return rng.choice(AREA3 + AREA4)


def local(rng, a):
    return d(rng, 8 if len(a) == 3 else rng.choice([7, 8]), lead_zero=False)


# ---------------------------------------------------------------------------
# 各子类的形态表: form_name -> 生成函数(rng) -> 值
# ---------------------------------------------------------------------------

def _phone_mobile():
    def plain(r): return mobile(r)
    def dash344(r): m = mobile(r); return f"{m[:3]}-{m[3:7]}-{m[7:]}"
    def space344(r): m = mobile(r); return f"{m[:3]} {m[3:7]} {m[7:]}"
    def cc_space(r): return "+86 " + mobile(r)
    def cc_dash(r): return "+86-" + mobile(r)
    def cc_paren(r): return "(+86)" + mobile(r)
    def cc_space344(r): m = mobile(r); return f"+86 {m[:3]} {m[3:7]} {m[7:]}"
    return locals()


def _phone_landline():
    def dash(r): a = area(r); return f"{a}-{local(r, a)}"
    def paren(r): a = area(r); return f"({a}){local(r, a)}"
    def paren_full(r): a = area(r); return f"（{a}）{local(r, a)}"
    def paren_space(r): a = area(r); return f"({a}) {local(r, a)}"
    def paren_grouped(r):
        a = area(r); l = local(r, a); return f"({a}){l[:-4]} {l[-4:]}"
    def paren_full_grouped(r):
        a = area(r); l = local(r, a); return f"（{a}）{l[:-4]} {l[-4:]}"
    def space(r): a = area(r); return f"{a} {local(r, a)}"
    def space_grouped(r):
        a = area(r); l = local(r, a); return f"{a} {l[:-4]} {l[-4:]}"
    def dash_grouped(r):
        a = area(r); l = local(r, a); return f"{a}-{l[:-4]}-{l[-4:]}"
    def plain(r): a = area(r); return a + local(r, a)
    return locals()


def _phone_hotline():
    def dash(r): return f"{r.choice(['400', '800'])}-{d(r, 3)}-{d(r, 4)}"
    def space(r): return f"{r.choice(['400', '800'])} {d(r, 3)} {d(r, 4)}"
    def plain(r): return r.choice(["400", "800"]) + d(r, 7)
    def short5(r): return r.choice(["95", "96", "12"]) + d(r, 3)
    def carrier(r): return r.choice(["10086", "10010", "10000", "10085", "10011"])
    def short3(r): return r.choice(["110", "119", "120", "122", "114", "12345", "12315", "12306"])
    return locals()


def _phone_international():
    def us(r): return f"+1 {d(r, 3, False)}-{d(r, 3)}-{d(r, 4)}"
    def us_paren(r): return f"+1 ({d(r, 3, False)}) {d(r, 3)}-{d(r, 4)}"
    def uk(r): return f"+44 20 {d(r, 4)} {d(r, 4)}"
    def hk(r): return f"+852 {d(r, 4, False)} {d(r, 4)}"
    def sg(r): return f"+65 {d(r, 4, False)} {d(r, 4)}"
    def jp(r): return f"+81-3-{d(r, 4)}-{d(r, 4)}"
    def dash_cc(r): return f"+{r.choice(['852', '853', '886', '61', '49', '33'])}-{d(r, 8, False)}"
    def idd(r): return f"00{r.choice(['852', '1', '44', '81', '65'])}-{d(r, 8, False)}"
    return locals()


def _phone_extension():
    def zhuan(r): a = area(r); return f"{a}-{local(r, a)}转{d(r, r.choice([3, 4]), False)}"
    def dash_ext(r): a = area(r); return f"{a}-{local(r, a)}-{d(r, 3, False)}"
    def ext(r): a = area(r); return f"{a}-{local(r, a)} ext.{d(r, 3, False)}"
    def hotline_zhuan(r): return f"400-{d(r, 3)}-{d(r, 4)}转{d(r, 1, False)}"
    def paren_zhuan(r): a = area(r); return f"({a}){local(r, a)}转{d(r, 3, False)}"
    return locals()


def _letters(r, alphabet, lo, hi):
    """SERIAL 字母前缀: 避开 ISO 币种代码(见 CUR_CODE_V2 注释)。"""
    while True:
        s = "".join(r.choice(alphabet) for _ in range(r.randint(lo, hi)))
        if s.upper() not in _CUR_SET:
            return s


def _serial_model():
    def lettnum(r): return _letters(r, "ABCDEFGHJKLMNPRSTVXYZ", 1, 3) + d(r, r.randint(2, 5), False)
    def lower(r): return _letters(r, "abcdefghkmnprstxyz", 1, 2) + d(r, r.randint(2, 4))
    def mixed(r): return f"{r.choice('ABCDHMRSX')}{d(r, 1, False)}{r.choice('ABCDNX')}{d(r, 1, False)}"
    def dash(r): return f"{r.choice(['RTX', 'GT', 'XR', 'KX', 'MX', 'TP'])}-{d(r, 4, False)}"
    def suffix(r): return f"{r.choice('ACGHMPSTX')}{d(r, r.randint(2, 4), False)}{r.choice(['Pro', 'X', 'S', 'Max', 'U', 'e'])}"
    return locals()


def _serial_plate():
    def dot(r): return f"{r.choice(PROVINCES)}{r.choice('ABCDEFGHJK')}·{d(r, 5)}"
    def plain(r): return f"{r.choice(PROVINCES)}{r.choice('ABCDEFGHJK')}{d(r, 5)}"
    def letters(r): return f"{r.choice(PROVINCES)}{r.choice('ABCDEFGHJK')}{r.choice('ABCDEFGHJKLMN')}{d(r, 4)}"
    def ev(r): return f"{r.choice(PROVINCES)}{r.choice('ABCDEFGHJK')}D{d(r, 5)}"
    def space(r): return f"{r.choice(PROVINCES)}{r.choice('ABCDEFGHJK')} {d(r, 5)}"
    return locals()


def _serial_id_doc():
    def idcard(r): return d(r, 17, False) + r.choice("0123456789X")
    def passport(r): return r.choice("EGPD") + d(r, 8)
    def hk_permit(r): return r.choice("CHM") + d(r, 8)
    def social(r): return d(r, 15, False)
    def license(r): return d(r, 12, False)
    return locals()


def _serial_order():
    def sf(r): return "SF" + d(r, 12, False)
    def jd(r): return "JD" + d(r, 13)
    def yt(r): return r.choice(["YT", "ZT", "EMS", "JT"]) + d(r, 12, False)
    def dashed(r): return f"{r.choice(['ORD', 'NO', 'TK', 'WO'])}-{d(r, 4, False)}-{d(r, 6)}"
    def hash_(r): return f"#{d(r, 4, False)}-{d(r, 4)}"
    def digits(r): return d(r, r.randint(10, 16), False)
    return locals()


def _serial_account():
    def name_num(r): return r.choice(["user", "tom", "lily", "abc", "dell", "admin", "xiao", "test"]) + d(r, r.randint(2, 6))
    def underscore(r): return r.choice(["tencent", "wang", "game", "cool", "star"]) + "_" + d(r, 4)
    def mixed(r): return f"{r.choice('xyzqk')}{d(r, 2)}{r.choice('abcdef')}{d(r, 4)}_{r.choice('uvw')}"
    def qq(r): return d(r, r.randint(8, 10), False)
    return locals()


def _time_clock():
    def hm(r): return f"{r.randint(0, 23)}:{r.randint(0, 59):02d}"
    def hm0(r): return f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}"
    def fullcolon(r): return f"{r.randint(0, 23)}：{r.randint(0, 59):02d}"
    def dian(r): return f"{r.randint(1, 12)}点"
    def dian_ban(r): return f"{r.randint(1, 12)}点半"
    def dian_fen(r): return f"{r.randint(0, 23)}点{r.randint(1, 59)}分"
    def dian_zheng(r): return f"{r.randint(1, 12)}点整"
    def shi_fen(r): return f"{r.randint(0, 23)}时{r.randint(0, 59)}分"
    def shi(r): return f"{r.randint(0, 23)}时"
    return locals()


def _time_ampm():
    def am(r): return f"{r.randint(1, 12)}:{r.randint(0, 59):02d}{r.choice(['am', 'pm', 'AM', 'PM'])}"
    def am_space(r): return f"{r.randint(1, 12)}:{r.randint(0, 59):02d} {r.choice(['am', 'pm', 'AM', 'PM'])}"
    def am_dots(r): return f"{r.randint(1, 12)}:{r.randint(0, 59):02d}{r.choice(['a.m.', 'p.m.'])}"
    def am_dots_space(r): return f"{r.randint(1, 12)}:{r.randint(0, 59):02d} {r.choice(['a.m.', 'p.m.', 'A.M.', 'P.M.'])}"
    def hour_only(r): return f"{r.randint(1, 12)}{r.choice(['am', 'pm', 'AM', 'PM'])}"
    return locals()


def _time_range():
    def colon(r):
        a = r.randint(6, 12); b = r.randint(a + 1, 23)
        return range_join(r, f"{a}:{r.choice(['00', '30'])}", f"{b}:{r.choice(['00', '30'])}")
    def dian(r):
        a = r.randint(6, 11); b = r.randint(a + 1, 12)
        return f"{a}点{r.choice(['到', '至', '-'])}{b}点"
    def ampm(r):
        return f"{r.randint(8, 11)}:{r.choice(['00', '30'])}{r.choice(['am', 'AM', 'a.m.'])}-{r.randint(1, 9)}:{r.choice(['00', '30'])}{r.choice(['pm', 'PM', 'p.m.'])}"
    return locals()


def _time_hms():
    def hms(r): return f"{r.randint(0, 23)}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}"
    def hms0(r): return f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}"
    def ms(r): return f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}.{d(r, 3)}"
    def shifenmiao(r): return f"{r.randint(0, 23)}时{r.randint(0, 59)}分{r.randint(0, 59)}秒"
    return locals()


def _y(r): return str(r.randint(1950, 2035))
def _m(r): return r.randint(1, 12)
def _dd(r): return r.randint(1, 28)


def _date_full():
    def ymd(r): return f"{_y(r)}年{_m(r)}月{_dd(r)}日"
    def ymd_hao(r): return f"{_y(r)}年{_m(r)}月{_dd(r)}号"
    def ym(r): return f"{_y(r)}年{_m(r)}月"
    def md(r): return f"{_m(r)}月{_dd(r)}日"
    return locals()


def _date_year():
    def y(r): return f"{_y(r)}年"
    def y_short(r): return f"{r.randint(0, 99):02d}年"
    def fy(r): return f"{_y(r)}财年"
    return locals()


def _date_month_day():
    def md(r): return f"{_m(r)}月{_dd(r)}日"
    def md_hao(r): return f"{_m(r)}月{_dd(r)}号"
    def hao(r): return f"{_dd(r)}号"
    def ri(r): return f"{_dd(r)}日"
    def m(r): return f"{_m(r)}月"
    return locals()


def _date_numeric():
    def iso(r): return f"{_y(r)}-{_m(r):02d}-{_dd(r):02d}"
    def slash(r): return f"{_y(r)}/{_m(r)}/{_dd(r)}"
    def dot(r): return f"{_y(r)}.{_m(r)}.{_dd(r)}"
    def dot0(r): return f"{_y(r)}.{_m(r):02d}.{_dd(r):02d}"
    def slash0(r): return f"{_y(r)}/{_m(r):02d}/{_dd(r):02d}"
    return locals()


def _date_decade():
    def decade4(r): return f"{r.choice(range(1900, 2030, 10))}年代"
    def decade2(r): return f"{r.choice(['50', '60', '70', '80', '90'])}年代"
    def century(r): return f"{r.randint(15, 21)}世纪"
    def century_decade(r): return f"{r.randint(19, 21)}世纪{r.choice(['20', '30', '50', '80', '90'])}年代"
    return locals()


def _date_range():
    def md_md(r): m = _m(r); a = _dd(r); return f"{m}月{a}日{r.choice(['-', '至', '到', '~'])}{m}月{min(28, a + r.randint(1, 9))}日"
    def ymd_d(r): m = _m(r); a = _dd(r); return f"{_y(r)}年{m}月{a}日至{min(28, a + r.randint(1, 9))}日"
    def years(r): a = r.randint(1950, 2020); return f"{a}{r.choice(['-', '—', '~', '至'])}{a + r.randint(1, 15)}年"
    def iso_iso(r): y = _y(r); return f"{y}-{_m(r):02d}-{_dd(r):02d}~{y}-{_m(r):02d}-{_dd(r):02d}"
    return locals()


def _date_slash():
    def md(r): return f"{_m(r)}/{_dd(r)}"
    def md0(r): return f"{_m(r):02d}/{_dd(r):02d}"
    return locals()


def _amount(r):
    return r.choice([num(r, 1, 999, 0.4, (1, 2)), num(r, 1000, 99999, 0.3, (2, 2)), num(r, 100000, 9999999)])


def _money_cny():
    def yuan(r): return _amount(r) + "元"
    def yen_sym(r): return r.choice(["¥", "￥"]) + _amount(r)
    def comma(r): return r.choice(["¥", "￥", ""]) + thousands(num(r, 1000, 9999999, 0.5, (2, 2))) + ("元" if r.random() < 0.5 else "")
    def kuai(r): return num(r, 1, 999) + r.choice(["块", "块钱", "元钱"])
    def rmb(r): return r.choice(["RMB", "CNY"]) + r.choice(["", " "]) + _amount(r)
    def renminbi(r): return _amount(r) + "元人民币"
    return locals()


def _money_foreign():
    def sym(r): return r.choice(["$", "€", "£", "HK$", "US$", "S$", "A$", "C$", "NT$"]) + _amount(r)
    def sym_comma(r): return r.choice(["$", "€", "£", "HK$", "US$"]) + thousands(num(r, 1000, 9999999, 0.4, (2, 2)))
    def code_pre(r): return r.choice(CUR_CODE_V2) + r.choice(["", " "]) + _amount(r)
    def code_post(r): return _amount(r) + r.choice(["", " "]) + r.choice(CUR_CODE_V2)
    def name(r): return _amount(r) + r.choice(CUR_NAME_V2)
    return locals()


def _money_large():
    def wan(r): return num(r, 1, 9999, 0.4, (1, 2)) + r.choice(["万元", "万", "亿元", "亿", "万美元", "亿美元", "亿港元", "万欧元"])
    def sym_wan(r): return r.choice(["¥", "$", "€", "HK$"]) + num(r, 1, 999, 0.5, (1, 1)) + r.choice(["万", "亿", "万亿"])
    def wanyi(r): return f"{r.randint(1, 99)}亿{r.randint(1, 9999)}万元"
    return locals()


def _money_range():
    def yuan(r): a = r.randint(10, 5000); return range_join(r, str(a), str(a + r.randint(10, 5000))) + "元"
    def yuan_both(r): a = r.randint(10, 5000); return range_join(r, f"{a}元", f"{a + r.randint(10, 5000)}元")
    def sym(r): s = r.choice(["¥", "$", "€"]); a = r.randint(10, 5000); return range_join(r, f"{s}{a}", f"{s}{a + r.randint(10, 5000)}")
    def wan(r): a = r.randint(1, 50); return range_join(r, str(a), str(a + r.randint(1, 50))) + "万元"
    return locals()


def _pct(r): return r.choice(["", "", "", "-", "+"]) + num(r, 0, 199, 0.4, (1, 2))


def _percent_rate():
    def pct(r): return _pct(r) + "%"
    def fullwidth(r): return num(r, 0, 100, 0.3) + "％"
    def permil(r): return num(r, 1, 99, 0.3) + "‰"
    # "X个百分点"不生成: LABEL_SPEC 未定义它属 PERCENT 还是 MEASURE、span 含不含"百分点"(复核时作歧义删除)
    return locals()


def _percent_range():
    def both(r): a = r.randint(1, 60); return range_join(r, f"{a}%", f"{a + r.randint(1, 40)}%", ("-", "~", "至", "到", "—"))
    def one(r): a = r.randint(1, 60); return f"{a}{r.choice(['-', '~', '至'])}{a + r.randint(1, 40)}%"
    return locals()


def _unit_value(units, lo=1, hi=999, dec=0.4):
    def gen(r):
        u = r.choice(units)
        return num(r, lo, hi, dec) + (r.choice(["", " "]) if u[0].isascii() and u[0].isalpha() else "") + u
    return gen


def _measure(sub):
    U = {
        "weight": ["kg", "g", "t", "mg", "KG", "斤", "公斤", "千克", "克", "吨", "磅", "lb"],
        "length": ["m", "cm", "mm", "km", "米", "厘米", "毫米", "公里", "千米", "英寸", "英尺", "海里", "ft", "in"],
        "volume": ["ml", "L", "mL", "毫升", "升", "立方米", "m³", "加仑", "cc"],
        "temperature": ["℃", "°C", "°F", "度", "摄氏度"],
        "speed": ["km/h", "m/s", "公里/小时", "米/秒", "节", "mph"],
        "area": ["平方米", "㎡", "m²", "平方公里", "亩", "公顷", "km²"],
        "data_size": ["GB", "MB", "TB", "KB", "G", "M", "Mbps", "GB/s"],
        "electric": ["mAh", "W", "kW", "V", "kWh", "A", "瓦", "千瓦", "伏", "度电"],
        "count": ["个", "名", "位", "辆", "栋", "台", "件", "次", "场", "家", "条", "只", "头", "张", "篇", "所", "架", "艘",
                  # 第 2 轮: 量词全量补齐(dev 上"12期/2针/280套"边界与类别错, 原表量词太少)
                  "期", "针", "套", "户", "盒", "瓶", "杯", "本", "份", "部", "门", "节", "箱", "包", "粒", "片", "支",
                  "把", "根", "双", "对", "届", "款", "项", "起", "轮", "批", "层", "间", "座", "株", "棵"],
        "age": ["岁", "周岁", "岁半"],
    }
    U["count_noun"] = U["count"]
    RANGE = {"age": (1, 99, 0.0), "temperature": (0, 45, 0.3), "speed": (1, 400, 0.2),
             "count": (1, 999, 0.0), "electric": (1, 9999, 0.2)}
    lo, hi, dec = RANGE.get(sub, (1, 999, 0.4))
    forms = {"plain": _unit_value(U[sub], lo, hi, dec)} if sub in U else {}
    if sub == "age":
        forms["months"] = lambda r: f"{r.randint(1, 23)}个月"
    if sub == "temperature":
        forms["neg"] = lambda r: "-" + num(r, 1, 40, 0.3) + r.choice(["℃", "°C", "度"])
        forms["range"] = lambda r: (lambda a: f"{a}{r.choice(['-', '~', '至', '到'])}{a + r.randint(2, 12)}{r.choice(['℃', '°C', '度'])}")(r.randint(-10, 30))
    if sub == "count":
        forms["big"] = lambda r: num(r, 1, 999, 0.3) + r.choice(["万", "万人", "万辆", "亿人", "万名"])
    if sub in ("weight", "length", "volume"):
        forms["range"] = lambda r: (lambda a, u: range_join(r, str(a), str(a + r.randint(1, 100))) + u)(r.randint(1, 500), r.choice(U[sub]))
    if sub == "duration":
        forms = {
            "hour": lambda r: num(r, 1, 72, 0.2, (1, 1)) + r.choice(["小时", "个小时", "h", "小时"]),
            "minute": lambda r: str(r.randint(1, 180)) + r.choice(["分钟", "min", "分钟"]),
            "second": lambda r: num(r, 1, 59, 0.3, (1, 2)) + r.choice(["秒", "s", "秒钟"]),
            "h_m": lambda r: f"{r.randint(1, 12)}小时{r.randint(1, 59)}分{r.choice(['钟', ''])}",
            "m_s": lambda r: f"{r.randint(1, 59)}分{r.randint(1, 59)}秒",
            "day": lambda r: str(r.randint(1, 365)) + r.choice(["天", "日", "个月", "周", "年"]),
            "period": lambda r: (lambda a: r.choice([str(a), f"{a}{r.choice(['-', '~', '到', '至'])}{a + r.randint(1, 5)}"]))(r.randint(1, 30))
                                + r.choice(["个工作日", "个月", "年", "周", "天", "个季度", "个星期"]),
        }
    if sub == "points":
        forms = {
            "exam": lambda r: num(r, 0, 150, 0.2, (1, 1)) + "分",
            "rating": lambda r: f"{r.randint(1, 9)}.{r.randint(0, 9)}分",
            "index": lambda r: thousands(num(r, 1000, 30000, 0.9, (2, 2))) if r.random() < 0.3 else num(r, 1000, 30000, 0.9, (2, 2)),
            "bp": lambda r: f"{r.randint(5, 200)}个基点",
        }
        forms["index"] = (lambda f: (lambda r: f(r) + "点"))(forms["index"])
    return forms


def _fraction_ratio():
    def slash(r): b = r.randint(2, 20); return f"{r.randint(1, b - 1)}/{b}"
    def slash_big(r): b = r.randint(20, 1000); return f"{r.randint(1, b - 1)}/{b}"
    def unicode(r): return r.choice("½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞")
    def mixed(r): b = r.randint(2, 9); return f"{r.randint(1, 9)}又{r.randint(1, b - 1)}/{b}"
    return locals()


def _score_match():
    def colon(r): return f"{r.randint(0, 9)}:{r.randint(0, 9)}"
    def colon_big(r): return f"{r.randint(10, 130)}:{r.randint(10, 130)}"
    def dash(r): return f"{r.randint(0, 30)}-{r.randint(0, 30)}"
    def endash(r): return f"{r.randint(0, 130)}{r.choice(['–', '—'])}{r.randint(0, 130)}"
    def bi(r): return f"{r.randint(0, 9)}比{r.randint(0, 9)}"
    def fullcolon(r): return f"{r.randint(0, 9)}：{r.randint(0, 9)}"
    return locals()


def _score_vote():
    def colon(r): return f"{r.randint(50, 400)}:{r.randint(0, 200)}"
    def bi(r): return f"{r.randint(5, 400)}比{r.randint(0, 200)}"
    def dash(r): return f"{r.randint(5, 99)}-{r.randint(0, 50)}"
    return locals()


def _math_arith():
    def add(r): a, b = r.randint(1, 999), r.randint(1, 999); return f"{a}+{b}={a + b}"
    def sub(r): a, b = r.randint(1, 999), r.randint(1, 999); return f"{max(a, b)}-{min(a, b)}={abs(a - b)}"
    def mul(r): a, b = r.randint(2, 99), r.randint(2, 99); return f"{a}{r.choice(['×', '*', 'x'])}{b}={a * b}"
    def div(r): b, q = r.randint(2, 30), r.randint(2, 99); return f"{b * q}{r.choice(['÷', '/'])}{b}={q}"
    def power(r): a = r.randint(2, 20); k = r.choice([2, 3]); return f"{a}{'²' if k == 2 else '³'}={a ** k}"
    def caret(r): a, k = r.randint(2, 9), r.randint(2, 10); return f"{a}^{k}={a ** k}"
    def sqrt(r): a = r.randint(2, 30); return f"√{a * a}={a}"
    def approx(r): return r.choice(["π≈3.14", "π≈3.1416", "e≈2.718", "√2≈1.414", "√3≈1.732"])
    def mixed(r): a, b, c = r.randint(1, 20), r.randint(1, 20), r.randint(1, 20); return f"({a}+{b})×{c}={(a + b) * c}"
    def paren_first(r): a, b, c = r.randint(1, 50), r.randint(1, 50), r.randint(2, 9); return f"({a}-{b})÷{c}"
    def trig(r): return r.choice(["sin30°=0.5", "cos60°=0.5", "tan45°=1", "sin90°=1", "cos0°=1", "sin60°≈0.866"])
    return locals()


def _math_algebra():
    def linear(r): return f"{r.choice('yz')}={r.randint(2, 9)}x{r.choice('+-')}{r.randint(1, 20)}"
    def func(r): return f"f(x)={r.randint(2, 9)}x^2{r.choice('+-')}{r.randint(1, 9)}"
    def quad(r): return f"x²{r.choice('+-')}{r.randint(1, 9)}x{r.choice('+-')}{r.randint(1, 20)}=0"
    def eq(r): a = r.randint(2, 9); x = r.randint(1, 20); b = r.randint(1, 30); return f"{a}x+{b}={a * x + b}"
    def ineq(r): return f"{r.randint(2, 9)}x{r.choice(['>', '<', '≥', '≤'])}{r.randint(1, 50)}"
    def vars_(r): return f"a{r.choice('+-')}b={r.randint(1, 30)}"
    def identity(r): return r.choice(["(a+b)²=a²+2ab+b²", "(a-b)²=a²-2ab+b²", "(a+b)(a-b)=a²-b²", "(x+1)²=x²+2x+1",
                                      "a²+b²=c²", "(x-2)(x+3)=0", f"(x+{r.randint(1, 9)})²={r.randint(4, 81)}"])
    return locals()


def _math_ratio_calc():
    def ratio(r): k = r.randint(2, 6); a, b = r.randint(1, 9), r.randint(1, 9); return f"{a * k}:{b * k}={a}:{b}"
    def frac_add(r): return f"1/{r.randint(2, 9)}+1/{r.randint(2, 9)}"
    return locals()


def _version_software():
    def V(r): return f"{r.choice(['V', 'v'])}{r.randint(1, 20)}.{r.randint(0, 20)}" + (f".{r.randint(0, 30)}" if r.random() < 0.6 else "")
    def V_space(r): return f"{r.choice(['V', 'v'])} {r.randint(1, 20)}.{r.randint(0, 20)}.{r.randint(0, 30)}"
    def bare3(r): return f"{r.randint(1, 20)}.{r.randint(0, 20)}.{r.randint(0, 30)}"
    def bare2(r): return f"{r.randint(1, 20)}.{r.randint(0, 20)}"
    def date_like(r): return f"{r.randint(2019, 2026)}.{r.randint(1, 12)}.{r.randint(1, 28)}"
    def four(r): return f"{r.randint(1, 20)}.{r.randint(0, 9)}.{r.randint(0, 9)}.{r.randint(100, 9999)}"
    return locals()


def _address_street_no():
    def hao(r): return f"{r.randint(1, 3999)}号"
    def nong(r): return f"{r.randint(1, 1999)}弄{r.randint(1, 99)}号"
    def duan(r): return f"{r.randint(1, 9)}段{r.randint(1, 999)}号"
    def hao_lou(r): return f"{r.randint(1, 999)}号{r.randint(1, 40)}楼"
    def hao_dash(r): return f"{r.randint(1, 3999)}号{r.randint(1, 30)}-{r.randint(1, 9)}"
    return locals()


def _address_building():
    def full(r): return f"{r.randint(1, 30)}号楼{r.randint(1, 6)}单元{r.randint(1, 33)}{r.randint(1, 8):02d}室"
    def dashed3(r): return f"{r.randint(1, 30)}-{r.randint(1, 6)}-{r.randint(1, 33)}{r.randint(1, 8):02d}"
    def dong(r): return f"{r.randint(1, 30)}{r.choice(['栋', '幢'])}{r.randint(1, 33)}{r.randint(1, 8):02d}{r.choice(['', '室'])}"
    def zuo(r): return f"{r.choice('ABCDE')}座{r.randint(1, 33)}{r.randint(1, 8):02d}"
    def qu(r): return f"{r.choice('ABCDE')}区{r.randint(1, 30)}号楼"
    def ceng(r): n = r.randint(2, 40); return f"{n}层{n}{r.randint(1, 20):02d}{r.choice(['', '室'])}"
    return locals()


FORMS = {
    ("PHONE", "mobile"): _phone_mobile(), ("PHONE", "landline"): _phone_landline(),
    ("PHONE", "hotline"): _phone_hotline(), ("PHONE", "international"): _phone_international(),
    ("PHONE", "extension"): _phone_extension(),
    ("SERIAL", "model"): _serial_model(), ("SERIAL", "plate"): _serial_plate(),
    ("SERIAL", "id_doc"): _serial_id_doc(), ("SERIAL", "order"): _serial_order(),
    ("SERIAL", "account"): _serial_account(),
    ("TIME", "clock"): _time_clock(), ("TIME", "ampm"): _time_ampm(), ("TIME", "range"): _time_range(),
    ("TIME", "duration_like_clock"): _time_hms(),
    ("DATE", "full"): _date_full(), ("DATE", "year"): _date_year(), ("DATE", "month_day"): _date_month_day(),
    ("DATE", "numeric"): _date_numeric(), ("DATE", "decade"): _date_decade(), ("DATE", "range"): _date_range(),
    ("DATE", "slash"): _date_slash(),
    ("MONEY", "cny"): _money_cny(), ("MONEY", "foreign"): _money_foreign(),
    ("MONEY", "large"): _money_large(), ("MONEY", "range"): _money_range(),
    ("PERCENT", "rate"): _percent_rate(), ("PERCENT", "range"): _percent_range(),
    ("FRACTION", "ratio"): _fraction_ratio(),
    ("SCORE", "match"): _score_match(), ("SCORE", "vote"): _score_vote(),
    ("MATH", "arith"): _math_arith(), ("MATH", "algebra"): _math_algebra(),
    ("MATH", "ratio_calc"): _math_ratio_calc(),
    ("VERSION", "software"): _version_software(),
    ("ADDRESS", "street_no"): _address_street_no(), ("ADDRESS", "building"): _address_building(),
}
for _s in ("weight", "length", "volume", "temperature", "speed", "area", "duration", "age", "count",
           "data_size", "electric", "count_noun", "points"):
    FORMS[("MEASURE", _s)] = _measure(_s)


# ---------------------------------------------------------------------------
# 产品名(LABEL_SPEC 总原则 4): "品牌/产品线 + 空格 + 数字" 不标; 字母与数字紧连的编码是 SERIAL。
# 第 2 轮加入: 第 1 轮合成数据大量出现"字母 空格 数字"的正例(RMB 5000、V 13.1.0、+86 138…),
# 而反例只有人工负例, dev 上 RTX 4060 / Ubuntu 22.04 / Excel 2019 被误标。品牌表按品类全量列,
# 不取自任何评测句。
# ---------------------------------------------------------------------------
PRODUCT_SPACED = {
    "phone": ["iPhone", "Galaxy S", "Galaxy Note", "小米", "Redmi Note", "华为 Mate", "华为 P", "荣耀", "OPPO Reno",
              "OPPO Find X", "vivo X", "一加", "Pixel", "魅族", "realme GT"],
    "chip": ["RTX", "GTX", "RX", "酷睿 i", "Ryzen", "骁龙", "天玑", "麒麟", "M"],
    "os_soft": ["Windows", "iOS", "Android", "macOS", "Ubuntu", "CentOS", "Debian", "Python", "Java", "CUDA",
                "Excel", "Office", "Photoshop", "Node", "Vue", "React", "PHP", "MySQL", "Redis", "Kubernetes"],
    "game_dev": ["PlayStation", "Xbox", "FIFA", "NBA 2K", "Switch", "Apple Watch Series", "iPad Pro", "AirPods",
                 "Surface Pro", "ThinkPad X", "MacBook Pro"],
    "vehicle": ["Model", "宝马 X", "奥迪 A", "奔驰 E", "波音", "空客 A", "理想 L", "蔚来 ES", "小鹏 G"],
}
SUFFIX = ["", "", "", " Pro", " Ultra", " Max", " Plus", " Ti", " LTS", " mini", " SE"]


def product_spaced(r):
    """负例值: 品牌 + 空格 + 数字(+后缀)。"""
    cat = r.choice(sorted(PRODUCT_SPACED))
    b = r.choice(PRODUCT_SPACED[cat])
    n = r.choice([str(r.randint(2, 20)), str(r.randint(2, 20)), f"{r.randint(2, 30)}.{r.randint(0, 9)}",
                  str(r.choice([360, 737, 787, 2019, 2021, 2024, 3060, 4060, 4090, 7900, 8, 9]))])
    return f"{b} {n}{r.choice(SUFFIX)}".replace("  ", " ")


def product_glued(r):
    """对照正例(SERIAL): 字母与数字紧连的型号编码。"""
    return f"{_letters(r, 'ABCDEGHKMNPRSTVXZ', 1, 3)}{r.randint(1, 9999)}{r.choice(['', '', 'S', 'X', 'Pro'])}"


def gen_value(rng, cls, sub):
    """(类, 子类) 下等概率抽一种形态, 返回 (值, 形态名)。"""
    forms = FORMS[(cls, sub)]
    name = rng.choice(sorted(forms))
    return forms[name](rng), name


if __name__ == "__main__":
    r = random.Random(0)
    n = sum(len(v) for v in FORMS.values())
    print(f"{len(FORMS)} 个子类, {n} 种形态")
    for (c, s), forms in FORMS.items():
        print(c, s, " | ".join(f"{k}:{forms[k](r)}" for k in sorted(forms)))
