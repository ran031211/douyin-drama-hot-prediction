"""
绘图统一样式与中文字体配置
-------------------------------------------------------------------
设计取向：低饱和、克制、留白充足。
  · 三主色：靛蓝（AI 内容）/ 青绿（真人内容）/ 陶土橙（强调）
  · 中性色偏冷灰，网格线极浅，去掉上右边框
  · 标题左对齐，刻度弱化，数值标签用次级灰而非纯黑
目标：让图表看起来像专业研究报告，而不是默认 matplotlib 配色。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings

warnings.filterwarnings("ignore")

# ---- 中文候选字体（按优先级） ----
CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Microsoft JhengHei",
    "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "SimSun",
]

_installed = {f.name for f in font_manager.fontManager.ttflist}
CJK_FONT = next((f for f in CJK_CANDIDATES if f in _installed), None)
if CJK_FONT is None:
    # 兜底：直接按文件路径加载 Windows 自带字体
    for p in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\msyhbd.ttc"]:
        try:
            font_manager.fontManager.addfont(p)
            CJK_FONT = font_manager.FontProperties(fname=p).get_name()
            break
        except Exception:
            continue

# ---- 全局 rcParams ----
plt.rcParams.update({
    "font.sans-serif": ([CJK_FONT] if CJK_FONT else []) + ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    # 轴线：只留左、下，颜色极浅
    "axes.edgecolor": "#DDE1E8",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # 网格：横向浅灰细线，置于数据之下
    "axes.grid": True,
    "grid.color": "#EDF0F5",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,

    # 标题左对齐（报告体例），深墨色
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#161A23",
    "axes.titlelocation": "left",
    "axes.titlepad": 11,

    "axes.labelsize": 10.5,
    "axes.labelcolor": "#5A6376",
    "xtick.labelsize": 9.8,
    "ytick.labelsize": 9.8,
    "xtick.color": "#8A94A6",
    "ytick.color": "#8A94A6",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "text.color": "#161A23",

    "legend.fontsize": 9.8,
    "legend.frameon": False,
    "figure.autolayout": False,
})

# ---- 调色板 ----
PALETTE = {
    # 中性
    "ink": "#161A23",        # 主文字
    "ink_soft": "#5A6376",   # 次级文字 / 数值标签
    "gray": "#8A94A6",       # 弱化元素
    "grid": "#EDF0F5",
    "surface": "#F7F9FC",    # 浅底块
    # 三主色
    "ai": "#4F5BD5",         # 靛蓝 —— AI 生成类内容
    "ai_light": "#A8AEF0",
    "human": "#0E9384",      # 青绿 —— 真人内容
    "human_light": "#7FD6C8",
    "accent": "#E2603F",     # 陶土橙 —— 关键强调
    # 辅助
    "blue": "#3B7DD8",
    "violet": "#7C5CD6",
    "amber": "#C08A2E",
    "teal": "#0E9384",
}

# 多系列顺序（先主色后辅助，保证相邻色可区分）
SERIES = ["#4F5BD5", "#0E9384", "#E2603F", "#C08A2E", "#7C5CD6", "#3B7DD8", "#8A94A6"]
# 连续型色阶（浅 → 深，用于热度梯度）
SEQ_BLUE = ["#EEF0FC", "#D3D8F7", "#AEB7EF", "#8894E7", "#6273DC", "#4F5BD5", "#3B45A8"]
# 双色渐变端点（AI → 真人）
GRAD_AI_HUMAN = ("#4F5BD5", "#0E9384")

GROUP_COLOR = {
    "内容属性": "#3B7DD8",
    "观众口碑": "#0E9384",
    "评论互动": "#4F5BD5",
    "文本态度": "#C08A2E",
}


def apply():
    """空调用：import 本模块即完成配置，此函数用于显式触发"""
    return CJK_FONT


def note(fig, text, y=0.005):
    """左下角脚注（数据口径说明）"""
    fig.text(0.005, y, text, fontsize=8, color=PALETTE["gray"], ha="left", va="bottom")


def value_labels(ax, bars, fmt="{:.0f}", color=None, offset=0.012, horizontal=True):
    """给条形图加数值标签，颜色用次级灰而非纯黑"""
    c = color or PALETTE["ink_soft"]
    for b in bars:
        if horizontal:
            ax.text(b.get_width() * (1 + offset), b.get_y() + b.get_height() / 2,
                    fmt.format(b.get_width()), va="center", fontsize=9, color=c)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * (1 + offset),
                    fmt.format(b.get_height()), ha="center", fontsize=9, color=c)


def finish(fig, path, footnote=None):
    """统一保存：加脚注、关闭"""
    if footnote:
        note(fig, footnote)
    fig.savefig(path)
    plt.close(fig)
    return path
