---
name: color-palettes
description: PPT(pptx-gen) 与 SCI 绘图(share-sci-plot)共用的配色登记处。用户自维护——换配色直接改本文件，pptx-gen 出 PPT、matplotlib 出图都来这里取当前默认色板。触发：换配色、这个配色不错、记录配色、PPT配色、SCI配色、调色板。
type: reference
---

# 配色登记处（PPT + SCI 绘图共用 · 用户自维护）

> 🔴 用户说"换配色 / 这个配色不错 / 记录配色"→ 改**本文件**（单一真源）。pptx-gen 生成 PPT、share-sci-plot 出 matplotlib 图，都从这里取**当前默认色板**。
> 🔴 后续用户自己换色：只动本文件一处，下游自动跟随。

## 色板：阿昆 浮世绘配色（暖红→冷蓝发散）

- **★★ PPT(pptx-gen) 当前默认**（2026-07-02 用户指定，替代下方 No.001）。
- **★★ SCI 绘图当前默认**（2026-07-02 用户明确指定，替代 NPG 作为默认；用户原话："我希望你将其作为我们PPT绘图以及SCI论文绘图的标准配色，除非我需要更换，要不然这就是默认配色"）。

来源：「阿昆的科研日常」《SCI论文插图 遇上浮世绘配色》。暖端红橙黄 → 冷端蓝，10 级连续发散，适合分类对比 / 堆叠 / 发散数据 / 排序条形图。

**10 色完整版（暖→冷）：**

| 序 | HEX | RGB | 角色 |
|---|---|---|---|
| 1 | `#E76254` | 231,98,84 | 珊瑚红（暖端 / 最高） |
| 2 | `#EF8A47` | 239,138,71 | 橙 |
| 3 | `#F7AA58` | 247,170,88 | 浅橙 |
| 4 | `#FFD06F` | 255,208,111 | 黄 |
| 5 | `#FFE6B7` | 255,230,183 | 浅黄（暖中性） |
| 6 | `#AADCE0` | 170,220,224 | 浅青（冷中性） |
| 7 | `#72BCD5` | 114,188,213 | 天蓝 |
| 8 | `#528FAD` | 82,143,173 | 中蓝 |
| 9 | `#376795` | 55,103,149 | 深蓝 |
| 10 | `#1E466E` | 30,70,110 | 深藏青（冷端 / 最低） |

**4 色精简版（≤4 分类，对比鲜明，图中高亮版）：** `#376795`（深蓝）· `#72BCD5`（天蓝）· `#FFD06F`（黄）· `#E76254`（红）

**matplotlib 直接用：**
```python
PALETTE_UKIYOE_10 = ["#E76254","#EF8A47","#F7AA58","#FFD06F","#FFE6B7",
                      "#AADCE0","#72BCD5","#528FAD","#376795","#1E466E"]
PALETTE_UKIYOE_4  = ["#376795","#72BCD5","#FFD06F","#E76254"]
```

**PPT(pptx-gen) 提示词用（把 hex 钉死，防 AI 偏色）：** 在每页 description 里写——
> "Color system ONLY these exact hex values — deep navy #1E466E, deep blue #376795, mid blue #528FAD, sky blue #72BCD5, pale teal #AADCE0, pale yellow #FFE6B7, yellow #FFD06F, light orange #F7AA58, orange #EF8A47, coral red #E76254. Pure white #FFFFFF background — never off-white, never a dark or gradient background panel."

## 色板：阿昆 SCI No.001（蓝→红发散，历史默认）

- 2026-06-30～2026-07-02 曾为 PPT 默认，现由上方浮世绘配色替代。保留供回滚。

来源：「阿昆的科研日常」SCI 论文插图经典配色 No.001。蓝-白-红发散，适合分类对比 / 堆叠 / 发散数据。

**7 色完整版（发散 / 多分类）：**

| 序 | HEX | RGB | 角色 |
|---|---|---|---|
| 1 | `#3951A2` | 57,81,162 | 深蓝（冷端 / 最低） |
| 2 | `#72AACF` | 114,170,207 | 中蓝 |
| 3 | `#CAE8F2` | 202,232,242 | 浅蓝 |
| 4 | `#FEFBBA` | 254,251,186 | 浅黄（中性 / 中点） |
| 5 | `#FDB96B` | 253,185,107 | 浅橙 |
| 6 | `#EC5D3B` | 236,93,59 | 橙红 |
| 7 | `#A80326` | 168,3,38 | 深红（暖端 / 最高） |

**4 色精简版（≤4 分类，对比鲜明）：** `#3951A2` · `#72AACF` · `#FDB96B` · `#EC5D3B`

**matplotlib 直接用：**
```python
PALETTE_AKUN_001   = ["#3951A2","#72AACF","#CAE8F2","#FEFBBA","#FDB96B","#EC5D3B","#A80326"]
PALETTE_AKUN_001_4 = ["#3951A2","#72AACF","#FDB96B","#EC5D3B"]
```

**PPT(pptx-gen) 提示词用（把 hex 钉死，防 AI 偏色）：** 在每页 description 里写——
> "SCI diverging palette ONLY: deep blue #3951A2, mid blue #72AACF, pale blue #CAE8F2, pale yellow #FEFBBA, light orange #FDB96B, orange-red #EC5D3B, deep red #A80326. Pure white #FFFFFF background. Deep blue #3951A2 / navy for headers and titles."

## 用法约定

以下约定针对上方标 ★★ 的当前默认色板（阿昆 浮世绘配色）：

- **PPT**：标题条 / 页眉用深藏青 `#1E466E`（或深蓝 `#376795`）；数据条、强调、等距插画的色阶走 10 色完整版（暖→冷）；背景**纯白**。
- **SCI 图**：分类 ≤4 用 4 色精简版（`#376795` · `#72BCD5` · `#FFD06F` · `#E76254`）；≥5 或发散数据用 10 色完整版；**保持色板既定顺序**（完整版暖→冷），别打乱。
- **色盲安全备选**：需要严格色盲安全或灰度印刷时，改用 Okabe-Ito（见 `publication-styling.md`），并加线型/marker 冗余编码。

## 维护（用户自己改）

- **换默认色板**：把"★★ SCI 绘图当前默认"这条 bullet 挪到你要的那节，并把「用法约定」一节的色值同步成新默认。
- **代码入口**：`figure_standards.py` 的 `default_palette(n)` 直接读本文件——按以粗体「★★ SCI 绘图当前默认」开头的 bullet 定位默认节，再读该节 `matplotlib 直接用` 代码块里的色值列表（最长的当完整版，最短的当精简版）。所以全文件只能有一节带这条 bullet，默认节必须保留这个 python 代码块；否则函数会直接报错。代码里不要再抄一份色值。
- **加新色板**：在下方新增一节（命名 + hex + RGB + 来源 + 适用场景）。
- 不删历史色板，便于回滚。

<!-- 在此追加更多色板节，如 ## 色板：Nature NPG / ## 色板：自定义XXX -->
