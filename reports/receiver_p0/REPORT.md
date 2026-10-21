# P0 报告:设备对判别失败定位 + 三分支 PoC — 2026-10-21

范围:P0-A/B/C + WiSig mirror(纯推理,~0.15 GPUh)+ PoC 42 fits(CE 锚点 6 + 36 arms,~1.4 GPUh)。
规则在任何模型输出前冻结于 `docs/receiver_p0/PROTOCOL_P0.md`。基线提交 `22e43743`,分支 `experiment/p0-pair-margin`。
数据边界:Shen 仅 Strict CIS 缓存;WiSig 仅开发池;最终 8 RX / 日期 2+ / 等化版 / 原始 IQ 未访问。

## 0. 结论先行

| 预注册问题 | 回答 | 证据 |
|---|---|---|
| 哪些设备对在未见 RX 上判别塌缩? | 高度集中于 ~6 对相近设备(两折一致:(0,1)(1,4)(2,3)(7,8) 族),塌缩单向、RX 特异 | P0-A 90 对 × 10 RX × 30 模型 margin 矩阵 |
| 错误为何单向? | 塌缩对在源域 q10 已最低(rank 2–10/90),但源域几乎不翻转(conf<0.08);单向由外层 RX 的 TX 特异畸变决定,与公共 RX 偏移无关(与 D2+ 一致) | P0-C Pluto 案例:1→0 rank 4、3→2 rank 2、4→1 rank 6、8→7 rank 10 |
| 源域能否预测? | **能,且不只是"本来就难"**:R_q ρ=0.72–0.73(两折,置换 p<1e-4),对普通难度基线 R_conf 的偏相关 +0.32/+0.37;seed 间风险排序稳定(ρ 0.92–0.95) | risk_correlation.csv |
| 编码器还是分类头? | **主要是编码器特征塌缩**:源域重拟合头在 Pluto 四个塌缩方向混淆率 0.53–0.68(冻结头 0.45–0.61),d_class(Pluto/源)= 0.61–0.97;重拟合头无法恢复 | refit vs frozen 分解 |
| 模型瓶颈? | eff_rank 3.4–3.8(512 维)、rank16 瓶颈下 128 维 3.7–3.8;但 (8,7) 的 d_class 比几乎不变仍塌缩 → 不能全归因容量 | geometry + (8,7) 案例 |
| WiSig 同样成立? | 源风险同样预测内部未见 RX 失败:R_q ρ=0.67–0.77、R_conf 0.67–0.82;跨数据集框架成立,物理成因不预设相同 | wisig_risk.csv |
| **PoC:三分支哪个有效?** | **drift(判别方向投影约束)与 pair(margin 稳定性)λ=0.1 有效;drift λ=1.0 在 fold 1 最强;pair λ=1.0 训练崩溃(CE 主导损失被淹没)出局;sub(原型间距)两折不一致,弱** | 42 fits,seed-mean |

**对下一阶段的决定**:
1. 以 **drift(λ∈{0.1,1})为主力机制**推进:平均外层精度不劣于 CE(部分 +2 pp),P 对塌缩数 8.3→5.0(fold 0 CE 锚点 8.3 → drift0.1 5.0)、4.7→2.7(fold 1),且与全分布对齐(CIDG/ConstStyle 类)有明确的机制区分(只约束 (w_t−w_u) 方向的投影)。
2. pair λ=0.1 与 drift 同量级但 Pluto 上更差(见 §3.4),作为 drift 的消融对照保留,不单独立项。
3. sub 分支(fold 0 掉 1.4–2.2 pp)与 H2 "特征塌缩→sub 应最好"的预测不符——塌缩在特征里,但直接推开原型间距不是正确的修复(它破坏了源域已学好的判别几何);这否定"特征间隔保持型表征学习"作为独立方法。
4. Pluto 上没有一个 arm 修复塌缩方向(全部 >0.5):**判别方向稳定性约束在源域成立、在极端外推 RX 上不足**——这是 target-free 方法的边界,写进论文(不掩盖)。

## 1. 核验

- 30/30 Shen checkpoint 哈希通过(与 D2 相同守卫);Strict CIS 输入 SHA 与 registration 一致(复用 R0/R1 verifier,本阶段未重算全部 40 资产,D 阶段已核)。
- logit 重实现与 `model(x)` 逐位一致(run_p0a 内断言)。
- **CE 锚点 6/6 逐位复现冻结 R0/R1 checkpoint**(0 tensor 偏差、同选中 epoch;`anchor_check_*.json`)——PoC 的对比基准可信。
- 复现检查:CE 锚点在 Pluto 1→0 三 seed 混淆 0.33/0.70/0.805(均值 0.61)与 R0/R1 存档 CSV 逐值一致。

## 2. P0-A/C:margin 矩阵与可预测性

(表格见 REPORT_SKELETON.md;此处只列判定)

- 冻结规则 R_q(源域最差 margin 风险)与外层失败:fold 0 ρ=0.735、fold 1 ρ=0.718(90 对,置换 p=5e-5)。
- 偏相关(R_conf 残差化后):+0.374 / +0.324——**margin 风险携带超出普通源域难度的信息**。
- R_spread(跨源 RX 波动)不提供增量(偏相关 ≈ 0/负):"跨 RX 不稳定"本身不预测,预测力来自绝对 margin 水平。
- 源域 top-10 风险对捕获外层 top-10 失败对 9/10(fold 0、fold 1 均 0.9)。
- seed 稳定性 0.92–0.95;两折排序一致(0↔1、1↔4、2↔3、7↔8 族)。
- SWAD vs CE 逐对:4 对受损 / 5 对受益,平均 |Δ|=0.006——SWAD 的 pair 级效应是集中的(与 D2+ 一致),不是均匀改善。
- WiSig mirror:R_q ρ official 0.775 / resnet 0.670,均 p<1e-12;框架跨数据集成立。

**判定:target-free 可预测性成立(预注册判据 ρ≥0.5、两折同向、偏相关>0 全部满足)。**

## 3. P0-B:分解 + PoC

### 3.1 编码器 vs 分类头

Pluto 四个塌缩方向(CE 3-seed 均值):冻结头 0.61/0.59/0.47/0.45 → 源域重拟合头 0.68/0.67/0.65/0.53。
**重拟合头不缓解(反而略差)**:特征在极端 RX 上已不可分,不是头边界过拟合源 RX。
d_class(pluto/source)= 0.65/0.61/0.97/0.75——(8,7) 特征间距几乎不变但 margin q10 从 3.66 → −1.94,说明该对的塌缩部分发生在头/判别几何层面;其余三对以特征塌缩为主。混合机制。

### 3.2 容量

eff_rank(512)=3.4–3.8(所有方法),128 维头前 3.7–3.8。按 PROTOCOL 解释禁令,只报告为编码器有效秩,不解释为身份信息维度。

### 3.3 PoC 主结果(外层 5 RX,3 seed 均值)

| arm | fold 0 Δacc | fold 0 Δconf_P | fold 1 Δacc | fold 1 Δconf_P | 塌缩数 f0/f1 |
|---|---:|---:|---:|---:|---:|
| CE 锚点 | — | — | — | — | 8.3 / 4.7 |
| sub λ0.1 | −1.37 pp | −0.002 | +0.66 pp | −0.011 | 7.3 / 5.7 |
| sub λ1.0 | −2.17 pp | −0.013 | +1.66 pp | −0.026 | 6.0 / 3.3 |
| drift λ0.1 | +0.42 pp | −0.005 | **+2.04 pp** | **−0.033** | **5.0 / 3.0** |
| drift λ1.0 | +0.35 pp | −0.010 | **+2.63 pp** | **−0.029** | 6.7 / **2.7** |
| pair λ0.1 | +0.50 pp | −0.002 | **+2.18 pp** | −0.024 | 6.0 / 3.0 |
| pair λ1.0 | **−55 pp(训练崩溃)** | — | **−64 pp** | — | 出局 |

- fold 1 所有三个机制在全部 3 seed 上 Δacc>0(drift λ1.0 最小 +2.3 pp)——不是 seed 运气。
- fold 0(含 Pluto)drift/pair λ0.1 微升 +0.4/+0.5 pp,Pluto 本身 −4 至 −5 pp(见 3.4)。
- pair λ1.0 崩溃原因:logsumexp 聚合的批内 min margin 在 P 对上的梯度规模远超 CE(源 val CE 从 1.5 升到 8),λ 需按损失规模重标定——H4 触发,该配置出局,不是机制本身的判决(PROTOCOL §6 H4 已按配置记录)。

### 3.4 Pluto 边界(不掩盖)

Pluto 四方向 conf:CE 0.61/0.59/0.47/0.45;drift λ1.0 0.55/0.58/0.52/0.49;pair λ0.1 0.75/0.54/0.70/0.57;sub λ1.0 0.79/0.65/0.43/0.57。
**没有任何 arm 修复 Pluto**;部分 arm 在 Pluto 上更差但被其他 RX 的增益抵消。
解读:源域判别方向约束能把"源域风险可预测的中等偏移"守住,守不住 Pluto 级别的极端外推(其带内 d_class 只有源域的 0.61–0.65)。
这正是 GPT 方案中"若源域无法预测损害,则明确可识别性边界"的实证版本:边界不是在"源域完全无法预测"(P0-C 已证可预测),而是在"可预测范围内的最极端尾部"。

## 4. 机制定位结论(对应 PROTOCOL §6 预注册分支)

- **特征间隔塌缩**(refit 头复现塌缩 + d_class 收缩):成立——但 sub 分支失败证明"直接推开原型"不是修复;正确形态是 drift(在判别方向上约束偏移,不重排几何)。
- **分类边界漂移**:在 (8,7) 上成立(特征间距不变、margin 塌缩)——drift 对此有效。
- **联合**:pair λ0.1 与 drift 同量级且在 Pluto 更差;下一步不再需要独立 pair 损失。
- **下一阶段方法**(待另行批准):drift 机制的正式化——接收机留域的判别方向 margin 稳定性,消融 λ、γ_d、P 构造(top-K vs R_neg),对照 CIDG/ConstStyle/GroupDRO/SWAD 全基线;Pluto 尾部作为"可预测性边界"写入分析,不作为方法失败隐藏。
- 已有方法直接解决?P0-B 显示重拟合头/擦除(D2)/SWAD/GroupDRO 均不能保护塌缩对;MLDG 在 Pluto 塌缩对上最好(0.29/0.28/0.52/0.13,D2+)但平均精度更低——drift λ1.0 在 fold 1 同时 +2.6 pp 与塌缩数 4.7→2.7 是首个两者兼得的信号(单折,exploratory)。

## 5. 产物

```
outputs/receiver_p0/p0a/{pairs,geometry,refit,recall}_shard*.csv   30 模型 margin/分解
outputs/receiver_p0/p0c/{risk_correlation,top_pairs.csv,answers.json}
outputs/receiver_p0/p0_wisig/{wisig_pairs.csv,wisig_risk.csv}
outputs/receiver_p0/pairs/pairs.json                              冻结 P 集
outputs/receiver_p0/poc/*/                                        42 fits(含 6 CE 锚点)
outputs/receiver_p0/poc/anchor_check_*.json                       锚点逐位复现
outputs/receiver_p0/poc_eval/{runs.csv,pairs.csv}                 外层评估
reports/receiver_p0/{REPORT.md,REPORT_SKELETON.md}
```
代码 SHA:各 manifest;GPU 消耗 P0 ≈ 0.15 h + PoC ≈ 1.4 h < 2 h 上限。停止条款生效:本报告后不自行扩展新诊断阶段。
