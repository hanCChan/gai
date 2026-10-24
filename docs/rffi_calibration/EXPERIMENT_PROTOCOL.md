# 实验协议（EXPERIMENT_PROTOCOL）v1

冻结于 2026-10-24，早于任何真实数据校准分数。机器可读版本：`configs/rffi_calibration/{registration,roles,datasets,formal_matrix}.json`。

## 1. 任务

闭集、已登记 TX 在新接收机上识别。部署到目标 RX 时，每个 TX 提供 `m ∈ {0,1,2,5,10}` 个可信带标签支持包（总标签 `m·K`）。query 归纳式打分：预测只依赖 query 特征、源统计和该 RX 的支持集；query 标签只进指标。类先验均匀。BN/Dropout 冻结；不按 query 重估任何统计。

信息权限分组：source-only（H0/H0S，m=0）｜few-shot supervised（H1–H5）｜transductive / 无标签目标方法（单独表，不混比）。

## 2. 数据角色与表示

见 `roles.json`。要点：

- **WiSig**：source 14 RX，两折 w0/w1（fit 10 / source_selection 4）；development 10；final 8 + 日期 3 + equalized 封存。fit = 日期 0–1、位置 0:120；选 epoch = fit RX 位置 120:160。
- **Shen**：source 10 RX = 组 A/B；Stage A selection 任务 A/b210_1、A/rtl_1、B/b200_mini_1、B/pluto_1（组内其余 4 RX 拟合）；dev 4、reserved 6 封存。
- **OSU**：TRANSFER_STRESS_ONLY，双向 RX1↔RX2，原始样本块 fit/support/query。

表示层：WiSig `OfficialCNN.encode()` 80-D（fc2 ReLU 后、dropout 前）；Shen 128-D `relu(tx_input(normalize(feature)))`（native classifier 的输入，前向 hook 捕获，原 forward 不改）；512-D 只作历史诊断。OSU 25 类在 Shen trunk 上丢弃 10 类头。

## 3. 支持 / 查询

每 (目标 RX, 类)：固定 query 集（WiSig/OSU 20、Shen 40，按位置均匀取，与 m 和 draw 无关）+ 不相交支持池；20 个 draw，每个 draw 是池内长度 10 的有序随机子集，种子 = sha256(dataset, rx, class, draw, tag)；m 取前 m 个 → 1⊂2⊂5⊂10 嵌套。所有头/encoder 用同一组 ID。覆盖门槛：支持池 ≥10 且 query 达标，否则进入部分覆盖表（缺类不按 0 计）。draw 不是独立 RX。

## 4. 统计模型与 H4

```
z[t,r,i] = μ[t] + d[t,r] + ε[t,r,i],  d ~ N(0,K[t]),  ε ~ N(0,Σ[t])      （全部对角）
u[t,r] 单元均值, v 单元方差(ddof=1), n 单元大小, R[t] 可用单元数(n≥2)
μ = mean_r u                       （接收机等权）
C = v/n;  B = var_r(u, ddof=1);  K_raw = B − mean_r C
K = a_K·max(K_raw,0) + (1−a_K)·K_pool,  a_K=(R−1)/(R−1+8)
Σ = a_S·mean_r v + (1−a_S)·Σ_pool,  a_S = df/(df+d),  floor = 1e-8 + 1e-5·mean(Σ_pool)
U = K/R + Σ_r C/R²;  A = K + U        （ablation H4-AK: A = K；H4-pooled: K_pool, Σ_pool）
m>0:  G = A/(A+Σ/m),  θ = μ + G(s−μ),  V = A·(Σ/m)/(A+Σ/m)
m=0:  θ = μ,  V = A
score_t(z) = log N(z | θ_t, Σ_t+V_t)  （完整 Mahalanobis + logdet），softmax 全部类
```

单源 RX：报错，不置 K=0。OSU 迁移变体：`μ_anchor = u_osu[t]`，`A = 2K_pool + Σ_pool/n_src[t]`（因子 2 = 两 RX 独立同分布偏移的建模假设）。这是经验贝叶斯插件近似，不积分 K/Σ 的估计不确定性。

## 5. 冻结头（Stage A）

H0 native｜H0S 源均值对角 LDA｜H1 目标均值 + 池化方差（H1e 欧氏）｜H2fix `m/(m+2)`｜H2L 历史 P2E proto（逐位一致）｜H2[α] α∈{0,.25,.5,.75,1}｜H3 普通 Gaussian 收缩（κ_b=8）｜H4 / H4-AK / H4-pooled｜H5a SimpleShot CL2N｜H5b Simple CNAPS 分类头（无 FiLM）｜H5c NIW 预测（κ0∈{1,4,16}，非元学习）。

参数族（H2、H5c）每个 m 在 source_selection RX 上按接收机等权准确率选；门槛表用**留一 selection RX 交叉拟合**（RX q 的分数用其他 RX 选出的成员），记作 `H2*`、`H5c*`。温度只影响 NLL/Brier/ECE，同样只在 selection 上拟合。

指标：接收机等权 Accuracy、macro-F1、worst-RX、每 TX recall、NLL、Brier、ECE、支持标签总量、adapt/predict 耗时；配对差逐 RX（draw 先平均），t 区间 + 改善 RX 数。

主比较：H4−H2*、H4−H3、H4−H2L、H4−H1、H4−H4-AK、H4−H4-pooled；另报 H2*−H2L、H5c*−H4、H5b−H4。

## 6. 训练（Stage B，未激活）

T0 native CE｜T1 CE + 历史 `p2_losses.ccalign`｜T2 CE + 接收机 episode ProtoNet query CE｜T3 CE + episode 后验预测 query CE（源统计只来自当步 reference RX，全程可微）。所有臂共用 `EpisodeSchedule`：伪目标 RX 均匀轮换，reference 每类 16，伪目标每类 m 支持 + 8 query，m∈{1,2,5,10}；同一步所有记录参与 native CE。权重 `CE + 1.0·CE_episode`。Adam、clip 5、按 fit RX 留出位置的接收机等权 CE 选 checkpoint。每个 encoder 都用 H2* 与 H4 双头评估。

## 7. 必须通过的测试

`pytest tests/rffi_calibration`（合成数据）：单元矩 / K 扣噪 / 单 RX 拒绝 / 缺类报错 / 后验闭式 + Monte Carlo + 对角=全协方差条件化 / m=0、m→∞ 极限 / logdet 与 scipy 一致 / Cholesky 抖动 / 迁移先验因子 / 与 P2E proto 逐位一致 / query 标签打乱不改 logits / 批大小与顺序不变 / 无信息特征处于机会水平 / 嵌套支持、query 不随 m 变 / 窗口块完整性 / 封存 RX 与日期、symlink、路径穿越拒绝 / 未激活预约拒绝数值访问 / 队列不重复领单、预算、校验失败不算完成、SIGTERM=cancelled / wrapper native logits 对齐、80/128/512 不混淆、类数不匹配拒绝 / T3 gradcheck、T0–T3 单步训练。

## 8. 门槛（工程门槛，不是创新定义）

WiSig：H4 − max(H1,H2*,H2L,H3) ≥ +2.0 pp @ m=1（同时报告 m=5、worst-RX、改善 RX 数）。Shen：m=1 与 m=5 最强简单头剩余错误的相对下降 + worst-RX。未通过 → 停止“分解必要”主张，保留简单校准成果。

## 9. 封存

final 8 RX、日期 3、equalized、Shen dev/reserved 默认不可数值访问（`AccessGuard`）。解封需在方法、超参、预处理、ckpt、manifest 全部冻结后单独授权。
