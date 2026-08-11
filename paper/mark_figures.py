"""Insert figure placement markers into the manuscript before pandoc.

Usage: python mark_figures.py [--zh]
"""
import sys
ZH = "--zh" in sys.argv
MARKS_EN = {
 "## 3. Methods": "@@FIG1@@",
 "### 4.2 The rate-controlling dislocation does not": "@@FIG7@@\n\n@@FIG8@@\n\n@@FIG2@@",
 "### 4.4 What a chemical mechanism would have to supply": "@@FIG5@@",
 "## 5. Numerical validation": "@@FIG6@@",
 "## 6. Preregistered test": "@@FIG3@@\n\n@@FIG4@@",
 "## 7. Limitations": "@@FIGS1@@",
}
MARKS_ZH = {
 "## 3. 方法": "@@FIG1@@",
 "### 4.2 速率控制型位错并不遵循这一图像": "@@FIG7@@\n\n@@FIG8@@\n\n@@FIG2@@",
 "### 4.4 化学机制必须提供多少": "@@FIG5@@",
 "## 5. 数值验证": "@@FIG6@@",
 "## 6. 预注册的检验": "@@FIG3@@\n\n@@FIG4@@",
 "## 7. 局限性": "@@FIGS1@@",
}
MARKS = MARKS_ZH if ZH else MARKS_EN
SRC = "manuscript_zh.md" if ZH else "manuscript.md"
DST = "manuscript_zh_marked.md" if ZH else "manuscript_marked.md"
s = open(SRC).read()
for k, v in MARKS.items():
    assert k in s, f"anchor missing: {k}"
    s = s.replace(k, v + "\n\n" + k, 1)
open(DST, "w").write(s)
