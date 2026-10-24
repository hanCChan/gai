# 创新边界与基线审计（NOVELTY_AND_BASELINE_AUDIT）

日期 2026-10-24。分支 `experiment/rffi-budget-calibration-v1`（基于 `b6ce3c2`）。本文件是**归属与对照清单**，不是实验结果。

## 0. 判定原则（按用户 2026-10-24 纠正）

1. **跨领域迁移是允许且有价值的创新形式。** 某算法在 CV / HAR / 通用 ML 中存在，不构成否决 RFFI 方法贡献的理由。要求只有两条：正确归属原始来源；与 RFFI 中同任务、同信息预算的最近工作公平比较。
2. **不得把原领域公式伪称原创。** 后验预测、收缩、ProtoNet episode 等数学组件都写明出处。
3. **不得在未检索的情况下声称“RFFI 首次”。** 本轮外网检索（Crossref/WebFetch）被环境拒绝，只能写“在已记录的检索范围内未发现同任务同机制工作”。

主张分三层，每层所需证据不同：

| 层级 | 主张形式 | 需要的证据 |
|---|---|---|
| L1 直接迁移 | “将 X（来源）首次/系统地用于已知 TX 跨 RX 少样本再校准” | 同任务 RFFI 对照、三数据集、成本曲线 |
| L2 面向 RX 变化的实质改造 | “把类中心的跨接收机变化 K 与包内散布 Σ 分开估计，并用有限源中心修正 U 形成预算相关后验” | L1 + 消融证明分解/修正本身有增量（H4 vs H2*/H3、A=K vs A=K+U、pooled vs class-shrunk） |
| L3 新训练机制 | “以后验预测风险在源 RX 伪目标 episode 上训练表示（T3）” | L2 + T3 vs T2、T3/H2 vs T3/H4 的 encoder×head 交叉 |

L1 已有实证基础（P2E proto：WiSig 53.5→82.0 @5-shot；Shen ccalign 87.0→89.8）。L2/L3 **尚未实验**。

## 1. 本地已实现且已实证的基线（不可遗漏）

`scripts/receiver_p2/p2e_fewshot.py` + `p2e_eigenrx.py`：

| 名称 | 规则 | 本仓库对应 |
|---|---|---|
| `proto` | 源 PCA(99.5%, ≤32) 空间，接收机等权类均值 μ，包合并 within-(class,RX) 全协方差收缩 10%，`μ + m/(m+2)(ḡ−μ)` | **H2L**，`LegacyP2EProto`；测试 `test_legacy_proto_matches_p2e_eigenrx_bitwise` 证明与历史 `EigenRX.loglik` 逐类得分一致（差 1e-8 内） |
| `span` | 源类 shift span 内保留 m/(m+0.5)，外部 m/(m+8) | 未移植（P2E 显示 ≤1 pp）；保留历史脚本 |
| `eig` / `eig+res` | 共享 eigen-receiver 系数 + 残差 m/(m+2) | 未移植；理由同上，P2E 结论“共享系数≈无增益（WiSig）” |
| `headft` | 冻结特征上源包 + 20× 支持包重训线性头 | 未移植；P2E 显示劣于 proto |

标签泄漏历史：P2E 审计 A1 的 `cc` 解码用了 query 真标签。新代码有回归测试 `test_uninformative_features_stay_at_chance`（无信息特征下所有头 <45%，4 类机会 25%）与 `test_query_label_permutation_changes_metrics_only`（打乱 query 标签，所有 logits 逐位相同）。

## 2. 跨领域可迁移算法

| 头 | 原始领域 / 来源 | RF 中是否已有类似机制 | 本任务适用性 | 部署前提 | 本实现与原文差异 |
|---|---|---|---|---|---|
| H1 Target prototype / pooled LDA | ProtoNet 决策规则 (Snell 2017, CV)；LDA | 有：INFOCOM'24 改进 ProtoNet [W11]、CMES'26 episodic proto [W12] | 直接适用 | 每 TX ≥1 支持包 | 冻结特征，无 episode 训练 |
| H2 标量收缩 | 经验贝叶斯 / James-Stein 型收缩；P2E 本地 | 本地 P2E | 直接适用 | 源类均值 | α 仅在 source_selection RX 上选（网格 {0,.25,.5,.75,1}），报告时交叉拟合 |
| H3 Gaussian 协方差收缩 | Ledoit-Wolf 类收缩；Simple CNAPS 层级收缩思想 | Gaussian Prototype RFFI（IoT-J 2025，少样本开集，LoRa）—**未取得全文，机制待核** | 直接适用 | 源样本 | 不分解 between/within；对角 |
| H4 后验预测 | 高斯共轭/经验贝叶斯；MetaQDA 的贝叶斯 QDA 思想 | 未在已记录检索中发现“已知 TX、跨 RX 中心变化 K + 包噪声 Σ + 有限源中心 U”的同机制工作 | 待检验 L2 候选 | ≥2 源 RX；OSU 单源 RX 只能用迁移变体 | 对角；插件估计，不积分 K/Σ 的估计不确定性，不叫 exact Bayesian |
| H5a SimpleShot CL2N | Wang 2019 (CV) | — | 强简单对照 | 源均值 | 忠实（最近质心） |
| H5b Simple CNAPS 分类器 | Bateni 2020 (CV) | — | 对照 | 仅支持集 | **只移植 Mahalanobis 头（λ=n/(n+1)，β=1，无 logdet），无 FiLM 自适应 —— 不是完整复现** |
| H5c NIW 预测（MetaQDA 风格） | Zhang 2021 MetaQDA (CV) | — | 对照 | 源矩 | 先验来自源矩 + κ0 网格 {1,4,16}，**未元学习先验 —— 不是 MetaQDA 复现** |
| T2 episodic ProtoNet | Snell 2017 | [W11][W12] | 训练对照 | ≥3 拟合 RX | 以“接收机”为 episode 单位 |
| T3 后验风险 episode | MetaQDA / CNAPS 端到端思想 | 未发现同机制 | L3 候选 | ≥3 拟合 RX | 源统计仅来自当步 reference RX |

## 3. RFFI 同任务对照（必须在正式论文中处理）

| 工作 | 任务 / 信息预算 | 与本任务关系 | 当前状态 |
|---|---|---|---|
| Few-shot cross-receiver RFFI based on feature separation, IET Commun. 2024 [W8] | 新 RX 少样本微调，~30 samples/TX | 最直接同任务；预算更大 | 摘要级核对；需全文与可复现协议 |
| A Novel RFFI Scheme for Few-Shot Open-Set Recognition, IEEE IoT-J 2025（IEEE 10960433） | LoRa，Gaussian prototype + Mahalanobis + 元学习，少样本开集 | 机制相近（Gaussian 原型），任务不同（开集/新设备） | **未核验全文**（本轮外网被拒）；正式比较前必须取得 |
| Few-Shot RFFI for UAVs Based on Gaussian Prototypical Network（DOI 候选 10.1109/LCOMM.2025.3591973） | UAV 少样本 | 机制相近 | **未核验**；不得按标题自行实现并称复现 |
| Calibrated Open-Set Prototypes [W28]、Prototypical Contrastive DA [W31]、CSCNet source-free [W10]、GAN-RXA [W7] | 开集/无标签目标/校准提取器 | 信息权限不同 | 单独分组报告，不与 inductive 少样本混比 |

在正式论文中，H4 首先要和 RF 内的 Gaussian prototype 类方法比（同任务、同 m），跨领域方法用于归属和消融。

## 4. 反证条件（预先写定）

- 若 H4 相对 max(H1, H2*, H2L, H3) 在 source_selection RX 上未达到门槛（WiSig m=1 +2 pp；Shen 剩余错误相对下降），**停止把 between/within 分解作为主创新**；保留 L1（简单原型校准的 RF 部署方案 + 成本曲线 + P2E 诊断）。
- 若 H2* 已解释大部分收益且 T3 不优于 T2，优先简单方案。
- 不因 H4 未赢而推翻路线：L1 本身仍成立。
