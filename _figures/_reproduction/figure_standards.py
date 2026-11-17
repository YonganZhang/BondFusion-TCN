"""
Standardized figure sizes for scientific publications.

Based on Nature/Science journal specifications. All figures at 300 DPI.
Usage:
    from figure_standards import create_figure, FIGURE_MODES, apply_sci_style
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Figure modes (Nature/Science specs)
# ---------------------------------------------------------------------------

FIGURE_MODES = {
    'A_square':   {'figsize': (3.5, 3.5), 'dpi': 300},   # 89x89mm   single panel, single column
    'B_wide':     {'figsize': (7.2, 3.5), 'dpi': 300},   # 183x89mm  two panels side-by-side
    'C_tall':     {'figsize': (3.5, 5.5), 'dpi': 300},   # 89x140mm  vertical multi-panel
    'D_full':     {'figsize': (7.2, 7.2), 'dpi': 300},   # 183x183mm large multi-panel
    'E_panorama': {'figsize': (7.2, 2.5), 'dpi': 300},   # 183x64mm  wide single-row / timeline
}

# ---------------------------------------------------------------------------
# Typography standards
# ---------------------------------------------------------------------------

SCI_RCPARAMS = {
    # Font
    # SKILL.md hard invariant 5: Times New Roman by default; switch to sans-serif/Arial
    # only when the user names a target journal that requires it.
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'font.sans-serif': ['Arial', 'Helvetica', 'SimHei'],
    'font.size': 7,

    # Axis labels
    'axes.labelsize': 8,
    'axes.titlesize': 8,          # discouraged; use caption instead

    # Tick labels
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,

    # Legend
    'legend.fontsize': 7,
    'legend.framealpha': 0.8,
    'legend.edgecolor': '0.8',

    # Lines
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,

    # Grid (off by default)
    'axes.grid': False,

    # Export
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.dpi': 150,

    # CJK minus sign
    'axes.unicode_minus': False,
}

# Colorblind-safe palettes
OKABE_ITO = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442',
    '#0072B2', '#D55E00', '#CC79A7', '#000000',
]

PAUL_TOL_BRIGHT = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44',
    '#66CCEE', '#AA3377', '#BBBBBB',
]


# ---------------------------------------------------------------------------
# Default palette — read from the registry, never copied into code
# ---------------------------------------------------------------------------
# color-palettes.md is the single source of truth. The section whose bullet list
# carries the bold marker "**★★ SCI 绘图当前默认**" is the current default; its
# "matplotlib 直接用" python block holds the full list and a compact (<=4) list.

from pathlib import Path as _Path
import re as _re

PALETTE_REGISTRY = _Path(__file__).resolve().with_name('color-palettes.md')
_DEFAULT_MARKER = '**★★ SCI 绘图当前默认**'
_HEX_LIST_RE = _re.compile(r'^\s*\w+\s*=\s*\[([^\]]*)\]', _re.M)


def _parse_default_palettes(text, source=PALETTE_REGISTRY):
    """Return (section_name, [lists of hex]) for the current SCI default section."""
    sections = _re.split(r'^## ', text, flags=_re.M)[1:]
    hits = [sec for sec in sections
            if any(_DEFAULT_MARKER in line for line in sec.splitlines()
                   if line.lstrip().startswith('-'))]
    if len(hits) != 1:
        raise ValueError(f'{source}: expected exactly one section marked '
                         f'"{_DEFAULT_MARKER}", found {len(hits)}')
    sec = hits[0]
    blocks = _re.findall(r'```python\n(.*?)```', sec, flags=_re.S)
    lists = []
    for block in blocks:
        for m in _HEX_LIST_RE.finditer(block):
            colors = _re.findall(r'#[0-9A-Fa-f]{6}\b', m.group(1))
            if colors:
                lists.append(colors)
    if not lists:
        raise ValueError(f'{source}: default section has no python hex list')
    return sec.splitlines()[0].strip(), lists


def default_palette(n=None, registry=None):
    """Current default SCI palette, parsed from color-palettes.md.

    Parameters
    ----------
    n : int, optional
        Number of categories. ``None`` returns the full list. ``n`` up to the
        compact list length returns the compact list's first ``n`` colours
        (the registry's "<=4 categories" rule); larger ``n`` samples the full
        list evenly while keeping its order.
    registry : path, optional
        Registry file to read (defaults to ``color-palettes.md`` next to this file).

    Raises
    ------
    ValueError
        Registry has no single marked default section, or ``n`` exceeds the full list.
    """
    path = _Path(registry) if registry else PALETTE_REGISTRY
    _, lists = _parse_default_palettes(path.read_text(encoding='utf-8'), path)
    full = max(lists, key=len)
    compact = min(lists, key=len)
    if n is None:
        return list(full)
    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')
    if n <= len(compact) and len(compact) < len(full):
        return list(compact[:n])
    if n > len(full):
        raise ValueError(f'default palette has {len(full)} colours, {n} categories requested; '
                         'merge categories or pick another palette from color-palettes.md')
    idx = np.linspace(0, len(full) - 1, n).round().astype(int)
    return [full[i] for i in idx]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def apply_sci_style():
    """Apply standardized scientific publication rcParams."""
    plt.rcParams.update(SCI_RCPARAMS)


def create_figure(mode: str, nrows: int = 1, ncols: int = 1,
                  panel_labels: bool = True, **gridspec_kw):
    """
    Create a figure using a standardized mode.

    Parameters
    ----------
    mode : str
        One of 'A_square', 'B_wide', 'C_tall', 'D_full', 'E_panorama'.
    nrows, ncols : int
        Subplot grid dimensions.
    panel_labels : bool
        If True and nrows*ncols > 1, add bold lowercase a, b, c labels
        (SKILL.md hard invariant 3: no parentheses).
    **gridspec_kw
        Extra kwargs forwarded to GridSpec (e.g. width_ratios, hspace).

    Returns
    -------
    fig : Figure
    axes : Axes or list[Axes]
        Single Axes if nrows==ncols==1, else flat list.
    """
    if mode not in FIGURE_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: {list(FIGURE_MODES.keys())}"
        )

    apply_sci_style()

    cfg = FIGURE_MODES[mode]
    fig = plt.figure(figsize=cfg['figsize'], dpi=cfg['dpi'])

    gs_defaults = {'hspace': 0.35, 'wspace': 0.35}
    gs_defaults.update(gridspec_kw)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, **gs_defaults)

    axes = []
    labels = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            if panel_labels and nrows * ncols > 1:
                idx = i * ncols + j
                ax.text(-0.12, 1.08, labels[idx],
                        transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='bottom')
            axes.append(ax)

    if len(axes) == 1:
        return fig, axes[0]
    return fig, axes


def savefig(fig, path, **kwargs):
    """Save with enforced 300 DPI and tight bbox."""
    defaults = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
    defaults.update(kwargs)
    fig.savefig(path, **defaults)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Font normalization ("字体核弹")
# ---------------------------------------------------------------------------
# 解决的问题：rcParams 只管新创建的文字，管不住代码里写死 fontsize= 的。
# normalize_fonts() 遍历图上所有已存在的文字对象，按类型强制覆盖。
# 不管 Gemini/Claude/手写代码怎么设字体，最后都被统一。
#
# 用法：在 savefig 之前调用 normalize_fonts(fig)
#   或直接用 save_normalized(fig, path) 一步到位

import re as _re

# 字体大小配置 — 与 figure-modes.md 字号表一致（A–E 档位都是最终印刷尺寸，印刷 pt 各档通用）
# 改这里即可全局生效；非论文派生展示物（PPT/网页）用 sizes= 覆盖。
FONT_SIZES = {
    'panel_label':  10,     # a b c 面板标签（粗体）
    'title':         8,     # 子图标题（论文图禁止使用，仅为兜底）
    'axis_label':    8,     # xlabel / ylabel
    'tick':          7,     # 刻度数字
    'legend':        7,     # 图例文字
    'annotation':    7,     # 普通标注 / 热图格内数值默认字号
    'small':         6,     # 国家代码等短标签
    'colorbar':      7,     # colorbar 标签
}

# annotate_heatmap 写入的格内数值带此 gid；normalize_fonts 保留其字号（热图格子小时调用方会主动调小）
HEATMAP_CELL_GID = 'heatmap_cell'

def _is_panel_label(s):
    """判断是否是 (a)(b)(c) 等面板标签。"""
    return bool(_re.match(r'^\(?[a-zA-Z]\)?$', s.strip()))

def _is_country_code(s):
    """判断是否是国家代码（2-3字母大写，无换行）。"""
    s = s.strip()
    return bool(_re.match(r'^[A-Z]{2,3}$', s)) and '\n' not in s

def _scale_for_ax(ax, base_size):
    """按子图宽度比例缩放字体——小子图稍微缩小，避免文字过挤。"""
    rel_width = ax.get_position().width  # 0-1 fraction of figure
    scale = 0.8 + 0.2 * rel_width       # full=1.0x, half=0.9x, quarter=0.85x
    return base_size * scale

def normalize_fonts(fig, sizes=None, scale_by_subplot=False):
    """
    遍历 fig 上所有文字对象，按类型强制设置字体大小。
    在 savefig 之前调用一次即可。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    sizes : dict, optional
        覆盖默认 FONT_SIZES，如 {'tick': 11, 'title': 13}
    scale_by_subplot : bool
        是否按子图大小自动缩放字体（小子图字体稍小）

    Examples
    --------
    >>> normalize_fonts(fig)                         # 用默认大小
    >>> normalize_fonts(fig, {'tick': 11})            # 刻度改为11pt
    >>> normalize_fonts(fig, scale_by_subplot=True)   # 自动适配子图大小
    >>> save_normalized(fig, 'out.png')               # 一步到位
    """
    fs = {**FONT_SIZES, **(sizes or {})}

    # 0. Suptitle（Codex发现的遗漏）
    if hasattr(fig, '_suptitle') and fig._suptitle and fig._suptitle.get_text():
        fig._suptitle.set_fontsize(fs['title'])

    for ax in fig.get_axes():
        # 按子图大小缩放的辅助函数
        def _sz(key):
            base = fs[key]
            return _scale_for_ax(ax, base) if scale_by_subplot else base

        # 1. Title
        if ax.title.get_text():
            ax.title.set_fontsize(_sz('title'))

        # 2. Axis labels (含twinx的secondary axis)
        ax.xaxis.label.set_fontsize(_sz('axis_label'))
        ax.yaxis.label.set_fontsize(_sz('axis_label'))

        # 3. Tick labels
        ax.tick_params(axis='both', which='major', labelsize=_sz('tick'))

        # 4. Legend
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_fontsize(_sz('legend'))
            lt = leg.get_title()
            if lt and lt.get_text():
                lt.set_fontsize(_sz('legend'))

        # 5. Bar labels（Codex建议）
        for container in ax.containers:
            for label in container.get_children():
                if hasattr(label, 'set_fontsize'):
                    label.set_fontsize(_sz('annotation'))

        # 6. Text annotations
        for t in ax.texts:
            s = t.get_text()
            if not s or not s.strip():
                continue
            if t.get_gid() == HEATMAP_CELL_GID:
                continue  # 热图格内数值：保留 annotate_heatmap 设定的字号
            s_clean = s.strip()
            if _is_panel_label(s_clean):
                t.set_fontsize(fs['panel_label'])  # 面板标签不缩放
                t.set_fontweight('bold')
            elif _is_country_code(s_clean):
                t.set_fontsize(_sz('small'))
            else:
                t.set_fontsize(_sz('annotation'))

        # 7. Colorbar detection (robust approach)
        if hasattr(ax, '_colorbar_info') or ax.get_label() == '<colorbar>':
            ax.tick_params(labelsize=fs['tick'])
            if ax.yaxis.label.get_text():
                ax.yaxis.label.set_fontsize(fs['colorbar'])
            if ax.xaxis.label.get_text():
                ax.xaxis.label.set_fontsize(fs['colorbar'])

    # 8. Figure-level texts
    for t in fig.texts:
        s = t.get_text().strip()
        if not s:
            continue
        if _is_panel_label(s) or (len(s) <= 2 and s.isalpha()):
            t.set_fontsize(fs['panel_label'])
            t.set_fontweight('bold')
        else:
            t.set_fontsize(fs['annotation'])


# ---------------------------------------------------------------------------
# Heatmap cell annotation（热力图默认每格标数值，字色按底色自动取黑/白）
# ---------------------------------------------------------------------------

def _relative_luminance(rgb):
    """WCAG 2.x 相对亮度。"""
    def _lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def heatmap_text_color(facecolor):
    """返回与底色对比度更高的 'black' 或 'white'（WCAG 对比度）。"""
    lum = _relative_luminance(mcolors.to_rgb(facecolor))
    contrast_black = (lum + 0.05) / 0.05
    contrast_white = 1.05 / (lum + 0.05)
    return 'black' if contrast_black >= contrast_white else 'white'


def annotate_heatmap(ax, im, data=None, fmt='{:.2f}', fontsize=None, **text_kw):
    """在 imshow 热图的每个格子标数值（含 0 值），字色按该格底色自动取黑/白。

    Parameters
    ----------
    ax : Axes
    im : AxesImage
        ``ax.imshow(...)`` 的返回值（默认 extent，格子中心在整数坐标）。
    data : 2D array, optional
        要标注的数值；默认取 ``im.get_array()``。被 mask 或为 NaN 的格子不标。
    fmt : str
        ``str.format`` 模板，如 ``'{:.2f}'``、``'{:d}'``。
    fontsize : float, optional
        默认 ``FONT_SIZES['annotation']``。格子放不下时先加大格子/图幅，再调小字号。
    **text_kw
        透传给 ``ax.text``（如 ``fontweight``）；传入 ``color`` 会覆盖自动取色。

    Returns
    -------
    list[Text]
    """
    values = np.ma.masked_invalid(np.ma.asarray(im.get_array() if data is None else data))
    if values.ndim != 2:
        raise ValueError(f'annotate_heatmap expects 2D data, got shape {values.shape}')
    fontsize = FONT_SIZES['annotation'] if fontsize is None else fontsize
    mask = np.ma.getmaskarray(values)
    texts = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if mask[i, j]:
                continue
            val = values[i, j]
            kw = {'ha': 'center', 'va': 'center', 'fontsize': fontsize,
                  'color': heatmap_text_color(im.cmap(im.norm(val)))}
            kw.update(text_kw)
            t = ax.text(j, i, fmt.format(val), **kw)
            t.set_gid(HEATMAP_CELL_GID)
            texts.append(t)
    return texts


def install_auto_normalize():
    """
    猴子补丁：让 plt.savefig 自动调用 normalize_fonts。
    调用一次后，所有 plt.savefig 都会自动规范字体，无需手动调用。

    用法:
        from figure_standards import install_auto_normalize
        install_auto_normalize()
        # 之后所有 plt.savefig 自动规范字体
    """
    _orig = plt.savefig
    def _patched(*args, **kwargs):
        normalize_fonts(plt.gcf())
        return _orig(*args, **kwargs)
    plt.savefig = _patched


def save_normalized(fig, path, sizes=None, also_pdf=True, scale_by_subplot=False, **kwargs):
    """
    normalize_fonts + savefig 一步到位。

    Parameters
    ----------
    fig : Figure
    path : str
        输出路径
    sizes : dict, optional
        字体大小覆盖
    also_pdf : bool
        是否同时保存 PDF
    scale_by_subplot : bool
        透传给 normalize_fonts：小子图字体稍小

    Examples
    --------
    >>> save_normalized(fig, 'figures/main/fig2.png')
    >>> save_normalized(fig, 'fig.png', sizes={'tick': 11})
    """
    from pathlib import Path
    normalize_fonts(fig, sizes, scale_by_subplot=scale_by_subplot)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    defaults = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.08}
    defaults.update(kwargs)
    fig.savefig(p, **defaults)
    if also_pdf:
        fig.savefig(p.with_suffix('.pdf'), **defaults)
    plt.close(fig)
    try:
        from PIL import Image
    except ImportError:
        print(f"  -> {p.name}: saved (Pillow not installed, pixel size not checked)")
        return
    with Image.open(p) as img:
        print(f"  -> {p.name}: {img.size[0]}x{img.size[1]} px")


def preview(path, width=800):
    """生成 800px 宽预览图到 Temp，返回预览路径。供 Read 展示给用户。"""
    from PIL import Image
    from pathlib import Path
    img = Image.open(path)
    if img.width <= width:
        return str(path)
    h = int(img.height * width / img.width)
    import tempfile
    out = Path(tempfile.gettempdir()) / "_preview.png"   # cross-platform temp dir
    img.resize((width, h), Image.LANCZOS).save(out, optimize=True)
    return str(out)
