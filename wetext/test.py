from pynini.lib import rewrite
from tn.chinese.rules.range import Range
from tn.chinese.rules_custom.date_cn import DateCN
from tn.chinese.normalizer_time import NormalizerTime
from importlib_resources import files            # noqa: E402


fsts = files("tn") / "fsts"
# 1. 初始化 (传入 Range 规则)
# range_processor = Range()
# processor = DateCN(range_tagger=range_processor.tagger)
fsts = files("tn") / "fsts"
processor = NormalizerTime(cache_dir=fsts / "time", full_to_half=False)

# 2. 测试用例
test_cases = [
    "国庆假期01年04月~10年04月，是旅游的黄金周。",
    "国庆假期01月04日~10月04日，是旅游的黄金周。",
    "国庆假期04号~05号，是旅游的黄金周。",
    "国庆假期2016年10月04号~05号，是旅游的黄金周。"
]

# 3. 运行测试
for text in test_cases:
    print(f"\n👉 测试文本: '{text}'")
    try:
        tagged = rewrite.top_rewrite(text, processor.tagger)
        print(f"  [Tag]        : {tagged}")

        verbalized = rewrite.top_rewrite(tagged, processor.verbalizer)
        print(f"  [Verbalized] : {verbalized}")
    except Exception as e:
        print(f"  ❌ 匹配失败: {e}")
        # 逐字探测，找出具体卡在哪个字符
        for i in range(1, len(text) + 1):
            try:
                rewrite.top_rewrite(text[:i], processor.tagger)
            except Exception:
                print(f"  🔍 定位卡点 : 在第 {i} 个字符 '{text[i-1]}' 处无法匹配 (前缀: '{text[:i]}')")
                break