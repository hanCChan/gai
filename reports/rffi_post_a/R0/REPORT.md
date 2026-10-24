# R0：WiSig 三臂可校准表征小试验 — `NO_GAIN_IN_FIXED_PILOT`

日期 2026-10-24。分支 `experiment/rffi-post-stage-a-diagnostics-v1`（激活提交 `75e3964`）。用户批准：
“根据这个方案继续进行，找到我们的有效算法”。本轮只激活 R0；D1 与逐 query 分数 replay **未运行**。

## 第一页

**做了什么**：WiSig 两折 × 三臂（C_MATCH 配对 CE、P_MIX 接收机混合 ProtoNet、P_RX 接收机成组 ProtoNet）
× 一个 seed = **6 次拟合**，每次 4800 步。之后对 `last.pt`（主）和 `native_best.pt`（次）做 12 次固定头评估，
用的是 Stage A 原 held-RX manifest（与 Stage A 的 manifest SHA 前缀一致才放行）。
6/6 拟合完成，无失败、无重试。同折三臂的初始化 SHA 相同，all_ids 链 SHA 相同。
独立 score→accuracy 重算 24 个目录全部一致（最大差 0）。预留 **0.1965 A100 GPUh**，上限 1.5。
reservation 已关闭。

**预登记判定**（`verify_results.py` 独立实现）：**NO_GAIN_IN_FIXED_PILOT**。

| 条件 | w0 | w1 | 通过 |
|---|---:|---:|---|
| P_RX/H1e − C_MATCH/H1e，lowshot（1/5-shot 均值）≥ +2 pp | **+4.08** | **−3.93** | 否 |
| 1-shot / 5-shot 分别 ≥ 0 | +4.20 / +3.95 | −2.80 / −5.07 | 否 |
| P_RX − P_MIX ≥ +1 pp | +11.63 | +1.91 | 是 |
| 8 个 held RX 中 ≥6 个对 CE 为正 | 3/4 | 1/4 | 否（4/8） |
| worst-RX 下降 ≤3 pp，H0 下降 ≤5 pp | 10.25 / 2.75 | 16.24 / 5.38 | 否 |
| P_RX/H1e 不低于 C_MATCH 最好的固定次要头 | 63.31 vs 62.47 | 55.55 vs 60.47 | 否 |

**结论**：在这个固定配方、一个 seed 下，没有证据表明接收机成组的 ProtoNet 训练稳定改善了少样本原型校准。
w0 的 +4 pp 和 w1 的 −4 pp 互相抵消。按规则停止扩矩阵，不做任何追加拟合。

## 三个能确定的现象

**1. P_MIX 在两折、所有头上都差于配对 CE**（H1e lowshot：w0 −7.6、w1 −5.9 pp；`native_best` 同向）。
这是本轮唯一在两折都一致的结果。P_MIX 的 support 和 query 来自不同接收机，ProtoNet loss 因此在把
同一 TX 在不同 RX 上的特征往一起拉。这本质上是一种跨 RX 不变性约束，与 P2E 以来“不变性用身份信息买单”的观察一致。
P_RX 的 support 和 query 来自同一 RX，没有这种约束，所以两折都好于 P_MIX（+11.6 / +1.9 pp）。
“成组优于混合”只是相对 P_MIX 成立，并不等于成组优于普通 CE。

**2. 同一批数据、同一初始化训练出的三个 encoder，在单个 held RX 上的差距可达 ±16–24 pp**
（H1e lowshot，例如 w0 RX16：C_MATCH 49.0、P_RX 68.2；w1 RX26：65.4 对 49.2）。
因此单 seed 的整折均值差 ±4 pp，落在训练轨迹本身的波动范围内。这不是“方法无效”的证明，
而是说明当前 pilot 的分辨率不足以检测几个 pp 的方法差。按计划不能用加 seed 来“救”结果；
是否单独申请一次方差测量由你决定（见下）。

**3. ProtoNet 损失改变了特征几何**：特征范数中位数从约 17（C_MATCH）降到 3–5（P_RX/P_MIX）。
native 训练 CE 变高（0.14–0.18 → 0.26–0.40），源验证 CE 略变差（0.20 → 0.21–0.25）。
P_RX 的 native 与 ProtoNet 梯度夹角余弦约 0.1–0.14，近乎正交；P_MIX 约 0.36–0.42。
所有臂 59–91% 的步数触发了 clip=5（配方继承自历史 WiSig 训练器）。

## 与 Stage A 的关系

C_MATCH（4800 步、新 episode 采样）的 H1e 1/5-shot 为 w0 54.86/63.62、w1 55.94/63.02。
Stage A T0（2400 步）的 8 RX 平均为 55.98/62.73。加倍步数没有带来可见收益。所有固定次要头的排序与 Stage A 相近：
5-shot 时 H5b/H5c 略优，1-shot 时 H5a 略优，H2L 在 1-shot 最弱。

## 限定

- 单 seed、两折拟合 RX 重叠、每折 4 个 held RX；区间与计数只作描述。
- 这些 held RX 已在 Stage A 使用，属于开发数据，不是盲测。
- `INCONCLUSIVE_OPTIMIZATION` 须人工判定。训练曲线没有发散，ProtoNet 训练准确率 0.87–0.95，
  因此不判为优化失败；但高 clip 比例和特征范数收缩是本配方的已知特征，未做任何调整。
- 没有温度拟合、没有 head/epoch 选择、没有加 seed。

## 文件

`execution_ledger.json`、`jobs.jsonl`、`verification.json`（独立 replay 与判定）、`evaluation/*/rows.csv` 与 `meta.json`、
`R0_*.log`、`access_log.jsonl`、`decision.json`。
分数文件 `evaluation/*/scores.npz` 和权重 `outputs/rffi_post_a/` 只留本地。
