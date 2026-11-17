# BondFusion-TCN 四大标准实验体系汇总报告（PyTorch Lightning + 6井对称嵌套 LOWO）

统一评估口径：6 口真实测井（A–F，共 55,155 采样点，170 个断gap感知连续同标签段，139 个连续测井段），全部模型采用验证集早停体制输出每口留一测试井的无泄漏预测概率。


## 一、对比实验（Comparison Experiments，按 Pooled Macro-F1 降序）

| 排名 | 模型 | Pooled Macro-F1 | 6井均值 | BalAcc | Kappa | F1_Good | AUC(G vs M) | 总误判数 | Run170 F1 | 数据来源 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | BondFusion-TCN (Ours) | **0.8629** | 0.8609 | 0.9081 | 0.9038 | 0.6788 | 0.9402 | 2521 | 0.8856 | Full Nested v04 + Lightning 4-Slot |
| 2 | 2D-CNN +VDL | **0.8355** | 0.8101 | 0.8608 | 0.8893 | 0.6087 | 0.9107 | 2883 | 0.8438 | Full Nested v04 |
| 3 | Reformer +VDL | **0.8256** | 0.8115 | 0.8775 | 0.8905 | 0.5745 | 0.8872 | 2863 | 0.8602 | Full Nested v04 |
| 4 | ModernTCN +VDL | **0.8022** | 0.7932 | 0.8138 | 0.8726 | 0.5240 | 0.9171 | 3284 | 0.8111 | Full Nested v04 (fine b=10) |
| 5 | VDL-ModernTCN (mean pool) | **0.7964** | 0.7751 | 0.8061 | 0.8900 | 0.4823 | 0.9189 | 2852 | 0.8259 | Full Nested v04 |
| 6 | HistGBDT (+VDL, Window) | **0.7881** | 0.8227 | 0.8142 | 0.8747 | 0.4733 | 0.8851 | 3253 | 0.8926 | Full Nested v04 |
| 7 | GRU +VDL | **0.7806** | 0.7766 | 0.7773 | 0.8784 | 0.4493 | 0.8630 | 3116 | 0.8589 | Full Nested v04 |
| 8 | HistGBDT (+VDL, Point) | **0.7710** | 0.8011 | 0.8048 | 0.8508 | 0.4477 | 0.8470 | 3906 | 0.8732 | Full Nested v04 |
| 9 | TimesNet +VDL | **0.7543** | 0.7719 | 0.7489 | 0.8218 | 0.4341 | 0.8859 | 4481 | 0.7939 | Full Nested v04 |
| 10 | Transformer +VDL | **0.7496** | 0.7501 | 0.7516 | 0.8394 | 0.3972 | 0.8496 | 4101 | 0.7700 | Full Nested v04 |
| 11 | TimeMixer +VDL | **0.7480** | 0.7420 | 0.7686 | 0.7882 | 0.4580 | 0.8846 | 5371 | 0.7716 | Full Nested v04 |
| 12 | LSTM +VDL | **0.7468** | 0.7584 | 0.7489 | 0.8619 | 0.3608 | 0.8883 | 3551 | 0.8283 | Full Nested v04 |
| 13 | TimeMixer (Curves only) | **0.7171** | 0.6966 | 0.7120 | 0.8517 | 0.2817 | 0.7718 | 3832 | 0.6688 | Full Nested v04 |
| 14 | Crossformer +VDL | **0.7024** | 0.6792 | 0.6938 | 0.7511 | 0.3567 | 0.8721 | 6203 | 0.7035 | Full Nested v04 |
| 15 | ModernTCN (Curves only) | **0.6857** | 0.6708 | 0.6870 | 0.8346 | 0.2004 | 0.6556 | 4280 | 0.6593 | Full Nested v04 |
| 16 | Crossformer (Curves only) | **0.6707** | 0.6694 | 0.6714 | 0.8323 | 0.1589 | 0.6284 | 4322 | 0.5978 | Full Nested v04 |
| 17 | Reformer (Curves only) | **0.6648** | 0.6622 | 0.6702 | 0.7998 | 0.1714 | 0.6697 | 5216 | 0.6752 | Full Nested v04 |
| 18 | GRU (Curves only) | **0.6593** | 0.6830 | 0.6618 | 0.8390 | 0.1129 | 0.7218 | 4149 | 0.6675 | Full Nested v04 |
| 19 | HistGBDT (Curves only) | **0.6419** | 0.6308 | 0.6452 | 0.7737 | 0.1300 | 0.6216 | 5906 | 0.5978 | Full Nested v04 |
| 20 | Transformer (Curves only) | **0.6369** | 0.6278 | 0.6384 | 0.7976 | 0.0862 | 0.7130 | 5261 | 0.6401 | Full Nested v04 |
| 21 | LSTM (Curves only) | **0.6324** | 0.6302 | 0.6406 | 0.8355 | 0.0342 | 0.5538 | 4240 | 0.6328 | Full Nested v04 |
| 22 | CBL Two-Threshold Rule | **0.6285** | 0.6148 | 0.6293 | 0.7665 | 0.0947 | 0.4659 | 6116 | 0.6250 | Full Nested v04 |
| 23 | TimesNet (Curves only) | **0.6209** | 0.6191 | 0.6272 | 0.8182 | 0.0166 | 0.5123 | 4684 | 0.5845 | Full Nested v04 |

## 二、消融实验（Ablation Experiments，单因素逐项消融）

| 变体配置 | Pooled Macro-F1 | 相对下降 Δ | 6井均值 | BalAcc | Kappa | F1_Good | AUC(G vs M) | 总误判数 | 边界误判(/155) | Run170 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BondFusion-TCN (Full) | 0.8629 | 0.0000 | 0.8609 | 0.9081 | 0.9038 | 0.6788 | 0.9402 | 2521 | 65 | 0.8856 |
| w/o Zero-phase filter | 0.8608 | -0.0021 | 0.8566 | 0.9049 | 0.9032 | 0.6728 | 0.9406 | 2535 | 68 | 0.8856 |
| w/o Trace contrast | 0.8605 | -0.0023 | 0.8591 | 0.9046 | 0.8994 | 0.6769 | 0.9400 | 2634 | 65 | 0.8856 |
| w/o CBL-gated bridge | 0.8484 | -0.0145 | 0.8440 | 0.9043 | 0.8885 | 0.6509 | 0.9283 | 2920 | 67 | 0.8812 |
| w/o Hilbert envelope | 0.7522 | -0.1107 | 0.7875 | 0.7885 | 0.8244 | 0.4212 | 0.8633 | 4610 | 72 | 0.7968 |
| w/o VDL (Curve-Prior only) | 0.6851 | -0.1778 | 0.6707 | 0.6864 | 0.8347 | 0.1984 | 0.6561 | 4278 | 65 | 0.6593 |

## 三、敏感性实验（Sensitivity Experiments）

| 分组 | 实验编号 | 声波表征 | Bin宽度 | Pooled Macro-F1 | 6井均值 | BalAcc | F1_Good | AUC(G vs M) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BondFusion-TCN Representation & Bin Width | bondfusion-hilbert-bin10 | hilbert | 10 | 0.8531 | 0.8163 | 0.8848 | 0.6454 | 0.9363 |
| BondFusion-TCN Representation & Bin Width | bondfusion-hilbert-bin20 | hilbert | 20 | 0.8629 | 0.8609 | 0.9081 | 0.6788 | 0.9402 |
| BondFusion-TCN Representation & Bin Width | bondfusion-hilbert-bin100 | hilbert | 100 | 0.7963 | 0.8090 | 0.8144 | 0.5043 | 0.8932 |
| BondFusion-TCN Representation & Bin Width | bondfusion-rectified-bin10 | rectified | 10 | 0.8178 | 0.8057 | 0.8279 | 0.5522 | 0.9289 |
| BondFusion-TCN Representation & Bin Width | bondfusion-rectified-bin20 | rectified | 20 | 0.7522 | 0.7875 | 0.7885 | 0.4212 | 0.8633 |
| BondFusion-TCN Representation & Bin Width | bondfusion-rectified-bin50 | rectified | 50 | 0.8045 | 0.7990 | 0.8210 | 0.5271 | 0.9095 |
| BondFusion-TCN Representation & Bin Width | bondfusion-rectified-bin100 | rectified | 100 | 0.7198 | 0.7532 | 0.7442 | 0.3206 | 0.8430 |
| BondFusion-TCN Representation & Bin Width | bondfusion-rectified-bin190 | rectified | 190 | 0.7270 | 0.7596 | 0.7415 | 0.3309 | 0.8589 |
| Backbone Representation & Bin Width | hilbert-bin10 | hilbert | 10 | 0.8388 | 0.8103 | 0.8838 | 0.6056 | 0.9195 |
| Backbone Representation & Bin Width | hilbert-bin20 | hilbert | 20 | 0.8508 | 0.8487 | 0.9025 | 0.6614 | 0.9282 |
| Backbone Representation & Bin Width | hilbert-bin100 | hilbert | 100 | 0.7814 | 0.7970 | 0.8034 | 0.4826 | 0.8692 |
| Backbone Representation & Bin Width | rectified-bin10 | rectified | 10 | 0.8022 | 0.7932 | 0.8138 | 0.5240 | 0.9171 |
| Backbone Representation & Bin Width | rectified-bin20 | rectified | 20 | 0.7499 | 0.7645 | 0.7725 | 0.4590 | 0.8510 |
| Backbone Representation & Bin Width | rectified-bin50 | rectified | 50 | 0.7994 | 0.7948 | 0.8185 | 0.5249 | 0.9030 |
| Backbone Representation & Bin Width | rectified-bin100 | rectified | 100 | 0.7121 | 0.7431 | 0.7362 | 0.3145 | 0.8165 |
| Backbone Representation & Bin Width | rectified-bin190 | rectified | 190 | 0.7128 | 0.7457 | 0.7279 | 0.3055 | 0.8332 |
| Curve Channel & Missingness | diagnostic_ac/tuned | curves+AC | 0 | 0.6527 | 0.6160 | 0.7816 | 0.4636 | 0.8487 |
| Curve Channel & Missingness | diagnostic_missing/tuned | curves+missing_mask | 0 | 0.3743 | 0.3713 | 0.6382 | 0.2168 | 0.6045 |
| Early-Stopping Schedule | legacy-stopping-ModernTCN | curves_only | 0 | 0.6857 | 0.6708 | 0.6870 | 0.2004 | 0.6556 |
| Early-Stopping Schedule | legacy-stopping-ModernTCN_plusVDL | rectified | 100 | 0.7244 | 0.7499 | 0.7404 | 0.3481 | 0.8215 |

## 四、泛化性与机理分析实验（Generalization & Mechanism Analysis Experiments）

| 分析类别 | 对象 / 井别 | Pooled/Well F1 | BalAcc | Kappa | F1_Good | 边界误判/总数 | 内部误判/总数 | 说明 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Per-Well LOWO Generalization | BondFusion-TCN (Ours) - Held-out Well A | 0.9634 | 0.9450 | 0.9749 | 0.9129 | 9/30 | 65/6001 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | BondFusion-TCN (Ours) - Held-out Well B | 0.9976 | 0.9972 | 0.9931 | 1.0000 | 7/15 | 9/4595 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | BondFusion-TCN (Ours) - Held-out Well C | 0.6848 | 0.8702 | 0.5823 | 0.2936 | 8/15 | 1002/3979 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | BondFusion-TCN (Ours) - Held-out Well D | 0.9266 | 0.9397 | 0.9060 | 0.8582 | 8/20 | 396/6836 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | BondFusion-TCN (Ours) - Held-out Well E | 0.7957 | 0.8714 | 0.9466 | 0.4378 | 13/35 | 222/14538 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | BondFusion-TCN (Ours) - Held-out Well F | 0.7975 | 0.7698 | 0.8761 | 0.5172 | 20/40 | 762/19051 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | 2D-CNN +VDL - Held-out Well A | 0.8088 | 0.7758 | 0.8970 | 0.5115 | 14/30 | 286/6001 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | 2D-CNN +VDL - Held-out Well B | 0.9786 | 0.9699 | 0.9646 | 0.9718 | 5/15 | 77/4595 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | 2D-CNN +VDL - Held-out Well C | 0.6574 | 0.8493 | 0.5348 | 0.2704 | 8/15 | 1155/3979 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | 2D-CNN +VDL - Held-out Well D | 0.9215 | 0.9195 | 0.9008 | 0.8502 | 9/20 | 412/6836 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | 2D-CNN +VDL - Held-out Well E | 0.6484 | 0.6541 | 0.9431 | 0.0000 | 14/35 | 236/14538 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | 2D-CNN +VDL - Held-out Well F | 0.8461 | 0.8245 | 0.8938 | 0.6442 | 19/40 | 648/19051 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | ModernTCN +VDL - Held-out Well A | 0.8896 | 0.8516 | 0.9346 | 0.7249 | 12/30 | 179/6001 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | ModernTCN +VDL - Held-out Well B | 0.9971 | 0.9975 | 0.9918 | 1.0000 | 6/15 | 13/4595 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | ModernTCN +VDL - Held-out Well C | 0.6448 | 0.8444 | 0.5117 | 0.2389 | 8/15 | 1279/3979 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | ModernTCN +VDL - Held-out Well D | 0.8708 | 0.8399 | 0.8393 | 0.7513 | 9/20 | 651/6836 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | ModernTCN +VDL - Held-out Well E | 0.6163 | 0.6010 | 0.8485 | 0.0000 | 14/35 | 605/14538 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | ModernTCN +VDL - Held-out Well F | 0.7408 | 0.7204 | 0.9167 | 0.3017 | 20/40 | 488/19051 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | CBL Two-Threshold Rule - Held-out Well A | 0.6276 | 0.6606 | 0.8495 | 0.0000 | 15/30 | 418/6001 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | CBL Two-Threshold Rule - Held-out Well B | 0.6473 | 0.6517 | 0.9245 | 0.0000 | 5/15 | 167/4595 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | CBL Two-Threshold Rule - Held-out Well C | 0.6353 | 0.6423 | 0.8223 | 0.0000 | 7/15 | 290/3979 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | CBL Two-Threshold Rule - Held-out Well D | 0.6273 | 0.6415 | 0.6169 | 0.1502 | 12/20 | 1556/6836 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | CBL Two-Threshold Rule - Held-out Well E | 0.6203 | 0.7820 | 0.7755 | 0.0977 | 11/35 | 1014/14538 | Nested LOWO blind well evaluation |
| Per-Well LOWO Generalization | CBL Two-Threshold Rule - Held-out Well F | 0.5312 | 0.5699 | 0.6025 | 0.0812 | 20/40 | 2601/19051 | Nested LOWO blind well evaluation |
| Mechanism Check | CBL-Gated Bridge Gain (Full vs w/o CBL-gated bridge on High-CBL Well E) | 0.8629 | 0.9081 | 0.9038 | 0.6788 | 65/155 | 2456/55000 | Zero-false-Good leakage gating |
| Mechanism Check | Early-Window Reference Drift (early-window-reference) | 0.6874 | 0.6790 | 0.7487 | 0.3122 | 77/155 | 6179/55000 | Tests casing-head subtraction fragility across boreholes |
| Mechanism Check | Target-Domain Leakage Diagnostic (auxiliary-test-centering) | 0.5669 | 0.5254 | 0.4564 | 0.2524 | 88/155 | 11978/55000 | Excluded auxiliary mode proving naive median shift fails |
| Mechanism Check | Boundary vs Interior Error Profile (w/o Zero-phase filter) | 0.8608 | 0.9049 | 0.9032 | 0.6728 | 68/155 | 2467/55000 | Boundary error check under causal group-delay filter |
