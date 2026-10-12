"""WCAG 对比度验证：候选暗色调色板 vs 实际背景。"""
import re, sys

def lum(hexc):
    hexc = hexc.lstrip('#')
    r, g, b = (int(hexc[i:i+2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def blend(fg, bg, a):
    """fg hex + alpha 叠在 bg hex 上"""
    fg = fg.lstrip('#'); bg = bg.lstrip('#')
    ch = lambda h, i: int(h[i:i+2], 16)
    out = []
    for i in (0, 2, 4):
        v = round(ch(fg, i) * a + ch(bg, i) * (1 - a))
        out.append(f"{v:02x}")
    return ''.join(out)

def ratio(fg, bg):
    l1, l2 = lum(fg), lum(bg)
    if l1 < l2: l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)

# ---- 背景 ----
PAGE   = "09090b"
SURF   = "131316"
SURF_A = "1a1a20"

# ---- 候选文字/语义色 ----
CAND = {
    "text":      "fafafa",
    "text2":     "a1a1aa",
    "text3":     "8e8e99",
    "accent":    "7aa2f7",
    "accentDk":  "a9c1fa",
    "tier-top":  "e8604c",
    "tier-high": "e5a04b",
    "tier-mid":  "9aa3b2",
    "tier-low":  "4eb598",
}
TINT = {"tier-top": "2a1715", "tier-high": "261c10", "tier-mid": "1d1d24", "tier-low": "0f211c"}
ACCENT_WEAK_A = 0.13

def line(tag, fg, bg, need=4.5):
    r = ratio(fg, bg)
    ok = "OK " if r >= need else "FAIL"
    print(f"  [{ok}] {tag:44s} {r:5.2f}:1  (需≥{need})")

print("== 正文/次要/弱化文字 × 各表面 ==")
for name in ("text", "text2", "text3"):
    for sname, s in (("page", PAGE), ("surface", SURF), ("surface-alt", SURF_A)):
        line(f"{name} #{CAND[name]} on {sname} #{s}", CAND[name], s)
print("  --text3 是大号/弱化用，也按 4.5 从严 --")

print("\n== 品牌色（链接/按钮文字/激活态）==")
aw_surf = blend(CAND["accent"], SURF, ACCENT_WEAK_A)
aw_surfa = blend(CAND["accent"], SURF_A, ACCENT_WEAK_A)
line(f"accent on page", CAND["accent"], PAGE)
line(f"accent on surface", CAND["accent"], SURF)
line(f"accent on accent-weak(surface)≈#{aw_surf}", CAND["accent"], aw_surf)
line(f"accent-dark on accent-weak(surface-alt)≈#{aw_surfa}", CAND["accentDk"], aw_surfa)
# 主按钮：渐变上的白字
line("white on grad #5b8cff→#8f7bf0 (取中点)", "ffffff", "778cf4", need=4.5)
# 深色底上的主按钮文字用 --c-text-invert = page 色
line(f"text-invert(#09090b) on accent #7aa2f7", PAGE, CAND["accent"])

print("\n== 分档色（verdict 文字用色 × 卡片底 和 × tint 底）==")
for name, c in CAND.items():
    if not name.startswith("tier"): continue
    line(f"{name} on surface", c, SURF)
    line(f"{name} on tint #{TINT[name]}", c, TINT[name])

print("\n== 提示框 tooltip 文字 ==")
line("tooltip text #fafafa on tooltip bg #1a1a20", "fafafa", SURF_A)
line("tooltip text #fafafa on tooltip bg #1f1f26", "fafafa", "1f1f26")
