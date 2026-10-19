# rl1 代码审查与科研判断（2026-10-19）

> 范围：只读审查，没有训练、没有 GPU、没有访问任何封存数据（本机本来也没有数据和 IQ）。
> 审查对象见第 0 节。凡写"已核实"的，都是我实际读了代码或重算了公开 CSV；凡写"子代理发现、已抽查"的，是另一个审查代理先找出、我再打开对应行确认过的。

---

## 0. 审查对象与版本（先读）

| 位置 | 内容 | 说明 |
|---|---|---|
| `/workspace/rl1/keyan/` | 服务器 zip 解出的 7 个工作区（code-y-b1 / c-audit / d0 / d05-b0 / m0 / p0-sync / p2a），3534 个文件 | **不含 R0/R1、R2、R2-C0、WiSig W0/W1、ReceiverTransform 的源码**。zip 最后一个文件 `code-y-p2a/.../A5_n210_1_ib_oob_gate_seed20261013/best.pt` 被截断（缺约 481 KB），已损坏 |
| `/workspace/rl1/gai/` | 从 GitHub 克隆的 `hanCChan/gai`，**当前检出 `bc05642`** | 交接文档冻结在 `aaa45ea`，但远端在它之后还有一次提交 `bc05642`（分支 `experiment/r2-c0-cross-rx-audit`，2026-10-13 21:55）：**R2-C0 跨 RX/日期风险转移审计 + 控制器等价加速与计时**。交接文档没有提到这一次，而它正好回答了交接要求"先评估风险能否跨 RX 预测"的问题。 |
| 一致性 | zip 里 `code-y-b1`、`code-y-m0` 与 gai 共有文件**逐字节一致**（子代理 `diff -rq`）；B1 和 ReceiverTransform 文件与 pre-execution seal 的 SHA 一致 | 没有代码漂移 |

**R2-C0 的结论（交接文档缺失的部分）**：在 80 个真实 Adam 快照上，WiSig 日期-MLDG 的"跨 RX 有害判断"平衡准确率 **0.50**（等于瞎猜），跨 RX+日期 0.506；同 RX 不同包则是 0.92。Shen 跨 RX 是 0.857，但判断侧只有 1 台 RX，此时多 RX 保护在数学上等于普通均值回溯。多 RX 规则与均值回溯只在 4/320 个轮换观测中选择不同。决定 `MODIFY`，18/54 项训练都未启动。

---

## 1. 一句话结论

> **勘误（2026-10-19，GPT 复核后）**：
> - E0 的"已有 logits"说法有误：评价脚本只保存了 argmax。
> - A3 中 r=0.97 不能作为选择依据。
> - A4 应收窄为"只影响方法间比较"。
> - R2-C0 平衡准确率 0.50 不代表风险分数没有信息（AUC 0.638）。
>
> 修正后的方案见 [RESEARCH_PLAN_link_shortcut_2026-10-19.md](../../../docs/interaction_l0/RESEARCH_PLAN.md)。


**没有发现会推翻已发表数字的代码 bug**；数据切分、封存、scaler 和种子配对都经得起检查。真正的问题出在**实验设计和解释层面**，有 5 条会改变"下一步该怎么做"的判断（A1–A5），其中 A2/A3 是我用公开 CSV 新算出来的，以前的报告里没有。

---

## 2. 问题清单

### A. 影响结论解释的问题（重要）

#### A1【结构性】R2 控制器测到的不是"未见 RX 风险"，从设计上就测不到
- **代码**：[train_receiver_r2.py:45-46](../../../scripts/train_receiver_r2.py#L45-L46) 每一步都重新 `prng.permutation(domains)` 划分 A/B/C；C1 再取当步 C 组 RX 的留出包（[train_receiver_r2.py:99](../../../scripts/train_receiver_r2.py#L99)）。
- **事实**：每台 RX 在大多数步骤里都会进入 A/B 参与梯度。所以 C1 其实是"**已经训练过的 RX 的留出包**"，控制器做的是"训练分布内留出包验证 + 回溯"。R2-C0 跨 RX 审计里的"目标 RX"也同样是训练 RX。
- **影响**：控制器的原始动机是保护未见 RX，而它的信号来源和这个目标在结构上就不一致。WiSig 跨 RX 平衡准确率 0.50 和这一点吻合。要真正测未见 RX 风险，必须让一组 RX 在整个训练过程中都不进入梯度（嵌套留出）。可是 WiSig 只有 14 台源 RX、Shen 只有 5 台，这样做会直接减少训练域。这个矛盾在报告中没有被正面写出。

#### A2【新分析】WiSig 的未见 RX 成绩主要由"seed 层面的全局因子"决定，3 个 seed 不够
用 [per_RX_date.csv](../../../reports/receiver_r01/2026-10-13/per_RX_date.csv)（官方 CNN，internal RX、未见日期）重算：
- **方差分解（CE）**：seed 主效应方差 **72.7**，大于 RX 主效应 **57.7**，RX×seed 交互 32.1（单位 pp²）。
- **seed 效应是全局的**：seed 20261014 在**全部 6 种方法**上都最好（CE 59.1、MLDG-date 70.3、PCGrad 65.4、GroupDRO 64.9…），seed 20261013 在全部方法上都最差（30–38%），而且几乎每台 RX 都同向。
- **方法差异依附于 seed**：MLDG−CE 在 seed 14 上对 10/10 台 RX 都是 +6~+18pp，在 seed 13 上 7/10 台为负。所谓"MLDG +3.54pp"基本就是"seed 14 上的大幅提升"。
- **影响**：方法比较被一个跨方法共享的、seed 级别的因子主导（初始化和/或采样顺序；目前两者分不开，因为它们按 seed 配对）。这是典型的**欠定问题（underspecification）**：源域上几乎同样好的解，在未见 RX 上相差 20–30pp。WiSig CE 一次拟合只要约 135 秒（≈0.04 GPUh，见 [training_cost.csv](../../../reports/receiver_r01/2026-10-13/training_cost.csv)），所以 3 个 seed 并不是成本逼出来的。

#### A3【新分析】现行 checkpoint 选择看不到 A2 的因子，而带分布偏移的源域代理能看到
18 个（方法×seed）模型上，与 internal-RX 未见日期准确率的相关：

| 源域量 | Pearson | 去掉方法均值后（只剩 seed 效应） | 去掉 seed 均值后（只剩方法效应） |
|---|---:|---:|---:|
| 源 RX、已见日期（≈现行同分布验证） | 0.34 | 0.65 | 0.47 |
| 源 RX、**未见日期 2** | **0.81** | **0.97** | 0.68 |

- **代码**：现行选择规则是同 RX 留出包的 receiver-equal CE 最小（[train_receiver_r01.py:149-151](../../../scripts/train_receiver_r01.py#L149-L151)，即 DomainBed 的 training-domain validation）。
- **注意**：这里的"源 RX 未见日期"用的是日期 2，和评价日期相同，**不能直接拿来做选择**，否则就是日期泄漏。它只说明：**有偏移的源域信号能预测 seed 级因子，同分布验证基本看不到**。合规的版本是源内留 RX 或留日期（日期 0/1 之间）做验证；这需要新的少量拟合才能检验（见第 4 节 E1）。样本只有 18 点、3 个 seed，这只是线索，不是结论。

#### A4【子代理发现、已核实】B1 第 0 折只有 CE 选到 LR=1e-4，其余全是 3e-4
- **证据**：[all_job_status.csv](../../../reports/b1_baselines/2026-10-13/all_job_status.csv) 第 16–17 行。选择流程本身合规（源内搜索，平手取小 LR），但结果是 fold 0 的比较混入了 LR 差异。
- **在 LR 一致的 fold 1 上**（[per_RX.csv](../../../reports/b1_baselines/2026-10-13/per_RX.csv)）：CE 的准确率 **87.38%** 和平均最差 TX 召回 **62.80%** 都是四种方法中最高（GRL 85.85/57.05，DRO 82.72/52.30，SupCon 83.56/53.10）。GRL 报告的 +2.85pp 最差 TX 召回全部来自 fold 0（+11.45），在 fold 1 是 −5.75。
- **影响**：B1 中"seed13/14 反转、某些方法兼顾平均和尾部"的叙述，有相当一部分是 LR 造成的。更准确的说法是：**在 LR 匹配的比较里，Shen 上没有标准方法胜过 CE**。另外，LR 搜索用 10 轮、正式训练用 20 轮，horizon 也不一致。

#### A5 训练时长截断贯穿 R0/R1 与 R2-A
- R0/R1：11 个普通 checkpoint 落在第 99/100 轮；R2-A：CE 在 200 轮时的最佳 epoch 是 194–196，**依然贴着边界**。
- **影响**："200 轮 CE 51.74% > MLDG 50.78%"同样不是收敛后的比较。CE 很便宜，应当训练到源验证出现平台（例如 400 轮 + patience 早停）再比较；MLDG 的最佳 epoch 在 150–165，已经进入平台。

### B. 统计与实现层面

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| B1 | [controller.py:69-71](../../../src/receiver_r2/controller.py#L69-L71) | 均值门槛用的是"各 RX 噪声带 2SE 的**平均**"，而不是"均值的 2SE"。R 台 RX 独立时，前者约是后者的 √R 倍（WiSig C 有 4 台 → 宽松 2 倍；Shen C 有 2 台 → 1.41 倍） | 均值保护比文档描述的宽松。每 RX 的 SE 把 10 个 TX × 日期的包混在一起算（[measurement.py:191-195](../../../src/receiver_r2/measurement.py#L191-L195)），SE 的抽样单位也不明确 |
| B2 | [measurement.py:140-152](../../../src/receiver_r2/measurement.py#L140-L152) | 倍率 0.5/0.25 时保留完整的 Adam m、v 和 step，但参数只走 s·δ；倍率 0 时整个梯度被丢弃 | 控制器约等于"按步自适应缩小 LR，且动量不一致"。必须与固定小 LR、普通回溯做对照（R2_C_FAIR_MATRIX 里已经计划了），否则收益无法归因 |
| B3 | 设计 | 控制器每步读取 C1 的真实标签（SOURCE_AUDIT 已披露） | 发表时需要"CE/MLDG 直接训练 fit+C1"的同标签预算对照 |
| B4 | [representations.py:16-17](../../../src/shen_p2a/representations.py#L16-L17)、[g1_data.py:52,57](../../../src/shen_p2a8/g1_data.py#L52) | Strict CIS 先滤到 ±62.5 kHz，再裁 52 个 bin（约 ±203 kHz，7.8125 kHz/bin）：只有 **17/52 行在带内**，其余 35 行是 boxcar 窗泄漏的比值，而且裁剪不对称一个 bin | 所有方法都用同样的输入，所以不影响 B1/R0/R1 内部比较。但"Strict 比 Full 高 13–18pp"不能解释成"去掉 OOB 的因果效应"。Strict 里 2/3 的输入是确定性泄漏，这本身可能就是噪声源，值得作为一个便宜的输入对照 |
| B5 | [run_m0_m2_signal.py:187-189](../../../scripts/run_m0_m2_signal.py#L187-L189)、[:87](../../../scripts/run_m0_m2_signal.py#L87) | (a) AWGN 功率按全带 1 MHz 信号功率设定，随后 `strict()` 只保留约 1/8 的噪声，所以带内 SNR 比标注高约 9 dB（标 40/20 dB，实际约 49/29 dB）。(b) 第 87 行 `array_equal(x+.15j*x.conj(), x+.15j*x.conj())` 是拿同一个表达式和自己比较，恒为 True | (a) M0 的绝对敏感度被低估，相对占比结论基本不受影响。(b) 真正的级联等价检验在 analyze 脚本里（9.93e-16），因此不影响结论 |
| B6 | [SOURCE_AUDIT.md](../../../reports/receiver_c0/SOURCE_AUDIT.md) 的数据 | CE 裁剪前梯度范数 7.3–10.0，裁剪率 73–98%；MLDG 只有 1.6–2.0，裁剪率 1–2% | 不是 bug。但有一个可检验的机制假设：MLDG 的一阶展开是 F+G−α∇F·∇G，相当于隐式的**梯度对齐/梯度范数正则**，它让轨迹远离高梯度、高曲率区域。若如此，MLDG 在 WiSig 上的收益应该能被 SAM/SAGM/Fish 一类方法复现，这也会决定它的真正对手是谁（见第 4 节 E3） |

### C. 潜在或轻微问题

- [measurement.py:174](../../../src/receiver_r2/measurement.py#L174) `mask_seed` 里没有 `step`。R2-A 每个 epoch 只测一次，控制器里 `train_replicates=0`，所以**目前没有影响**；以后若每步测 train 模式，同一 epoch 内的 Dropout 掩码会重复。
- [wisig_w01_common.py:20-23](../../../scripts/wisig_w01_common.py#L20-L23) 忽略了 pickle 中 Fortran-order 标志（子代理发现）。是潜在风险，建议加一个断言。
- W1 的 "TX×RX 15–25%" 计算正确，但份额里包括编码器从没训练过的 TX 8–9；日期 0/1 是编码器见过的日期，所以"日期效应"和"编码器泛化"混在一起（子代理发现）。
- ReceiverTransform 的 rank-2 映射是在 8 个类均值上做 ridge 拟合（秩 ≤7、维度 32），本质上欠定，与报告的说法一致；对照组是公平的（子代理发现）。
- M0 已提交的 REPORT.md 与脚本生成的版本不一致，说明有人工编辑，属于可追溯性问题（子代理发现）。

### D. 已核实无问题的部分
- MLDG 是完整二阶实现（`create_graph=True`），内层只用 train 子集，外层除以 (1+β)：[core.py:37-46](../../../src/receiver_r01/core.py#L37-L46)。
- PCGrad 的两任务对称投影正确：[core.py:49-61](../../../src/receiver_r01/core.py#L49-L61)。
- 所有骨干都没有 BatchNorm，因此 `functional_call`/SWAD 没有 BN 统计问题；[measurement.py:163](../../../src/receiver_r2/measurement.py#L163) 也有断言保护。
- 平衡采样保证每个 batch 中每个 RX×日期×TX 恰好 2 包：[core.py:64-72](../../../src/receiver_r01/core.py#L64-L72)。
- GroupDRO 的指数梯度更新正确，η=0.01，分组为 RX×日期×TX。
- 数据切分：WiSig inner_train/inner_held 都在 14 台源 RX 内，final_rx 和日期 3 有断言拦截；Shen 外层 RX 从不进入训练 keys；scaler 只在拟合包上估计（R2 是前 120 包）。
- 风险测量用 `fork_rng` 隔离 RNG；scale=1/0 与实际损失逐位相等的断言都在。
- C0 的加速版与原规则等价：单倍率输入下 `choose_scale` 的语义不变（[optimized.py:45](../../../src/receiver_c0/optimized.py#L45)）。
- C0 的转移分析中，判断侧只用 left 组的 C1，目标侧只用 right 组的 C2，没有交叉污染：[analyze_receiver_c0.py:35](../../../scripts/analyze_receiver_c0.py#L35)。

---

## 3. 对 R2 的决策：`STOP_R2_CONTROLLER_BUT_KEEP_CROSS_RX_TASK`

我的判断是**停止把"每步有限倍率控制器"当作论文机制**，保留它的代码作为基础设施，跨 RX 研究任务本身不变。理由是多条独立证据指向同一个方向，并不只是一次小模型失败：
1. **测量对象不对**（A1）：从结构上测不到未见 RX 风险。
2. **WiSig 跨 RX 转移等于瞎猜**（C0，BA 0.50），同 RX 却有 0.92。也就是说，单步风险是 RX 局部的，不能外推。
3. **多 RX 规则几乎没有增量**：Shen 1/1 划分下与均值回溯在数学上恒等，WiSig 只有 4/320 次选择不同。
4. **剩下的部分是经典方法**：均值回溯就是 Armijo/信赖域式的步长回溯，与 CAGrad/MGDA 高度重叠；而且收益与"更小有效 LR"和"额外 C1 标签"两个混杂因素纠缠在一起（B2、B3）。
5. **更重要的是粒度不对**：A2/A3 显示，决定 WiSig 未见 RX 成绩的是"**落在哪个解**"这种运行级（run-level）因子，而不是某一步更新是否有害。在步级别做控制，抓的是次要变量。

这不是说"跨 RX 泛化没有可做的"，而是**研究对象应当从"步"上移到"解"**。

---

## 4. 下一步建议（只是建议，未执行任何训练，需要你批准）

### 主选项 E：运行级欠定（run-level underspecification）诊断 → 方法

**可证伪的核心假设 H**：在跨 RX 的 RFFI 中，未见 RX 性能的主要变异来自运行级的解选择（初始化/数据顺序），而且这一变异可以用**合规的源内偏移代理**（留源 RX 或留日期）预测，或者用集成/平均消除。

| 阶段 | 内容 | 需要的资源 | 通过/失败判据（事先登记） |
|---|---|---|---|
| **E0** | 用服务器上已有的 R0/R1 logits 做 3-seed logit 集成（同方法以及跨方法），在 WiSig internal 和 Shen outer 上评估 | **0 GPU**，CPU 读取已有预测（服务器 `outputs/receiver_r01`，本机没有） | 若 3-seed CE 集成 ≥ 最好单方法均值 + 3pp，说明方差是可平均的，集成必须成为强基线；若无收益，H 的"可平均"部分被削弱 |
| **E1** | WiSig CE，**初始化 4 × 数据顺序 4 的因子设计 = 16 次拟合**，200 轮 + 早停；每次固定留出 2–3 台源 RX（整个训练都不用）作为合规代理；同样在 Shen 做 16 次 | WiSig 约 16×0.075 ≈ 1.2 GPUh，Shen 约 0.3 GPUh；**建议上限 2 GPUh** | (i) 分清 seed 效应来自 init 还是顺序；(ii) 留出源 RX 的准确率与 internal RX 的 Spearman ≥ 0.6 才算代理可用。若 <0.3，就是代理不存在，H 的"可预测"部分被否定 |
| **E2** | 只有 E0/E1 至少一项通过才做：在代理上做选择或加权集成，与 CE/SWAD/MLDG/EoA/DiWA 同预算比较 | 视 E1 结果另行估算 | 跨两个数据集、≥10 seed 稳定优于 EoA 式集成，才考虑作为方法提出 |

**与先行工作的关系（诚实说明）**：集成和平均（EoA、DiWA、SWAD）、DG 中的模型选择（DomainBed）、欠定问题（D'Amour 等 2020）都已经有了。可能的新意只在于：(1) 在 RFFI 跨 RX 场景下证明失效是运行级的，并且能被源内偏移看到；(2) 设计利用 RX/日期偏移结构的选择/组合算法。如果 E2 只能做到"集成更好"，那就是一个强基线结论，不是新方法。这一点必须事先讲清。

### 备选 E3：检验"MLDG 的收益 = 隐式梯度对齐/平坦化"（由 B6 引出）
- **做法**：在 WiSig 上用同样的 horizon 和 ≥8 seed，比较 CE、CE+clip 放宽、SAM、Fish/SAGM 与 MLDG。约 1–1.5 GPUh。
- **判据**：若 SAM/Fish 能复现 MLDG 的收益，说明 MLDG 的"元学习"叙事不成立，真正的对手是平坦化方法；若不能复现，二阶留域项可能确实有特定作用。
- 这一项更像机制澄清，不直接产出方法；建议排在 E 之后。

### 暂时不建议做的
- 不做 R2-C 的 54 项或 R2-C0 条件下的 18 项拟合（理由见第 3 节）。
- 不用 C2 调阈值；不触碰 WiSig 最终 8 台 RX、日期 3、等化版，也不碰 Shen 开发/锁定 RX。

---

## 5. 附：论文与代码
- 本地下载于审查工作区（PDF 不上传，受第三方版权约束）；清单、来源、SHA256 和获取状态见 [LITERATURE_INDEX.md](../../../docs/interaction_l0/LITERATURE_INDEX.md)。共 34 篇，全部通过完整解析，标题已核对；
- 5 个作者/基准代码仓库（Shen receiverAgnosticRFFI、IB-RFF、wisig-examples、DomainBed、SWAD）放在 `papers/code/`，commit SHA 也记录在 README 里。
- **没有拿到**：A3 IB-RFF（TIFS 2026，没有开放版本，只有代码）、A4 RIFLE（ICCCN 2026）。这两篇需要你用图书馆权限获取。
- 交接清单之外，检索补充了 7 篇 2023–2026 年的跨接收机 RFFI 论文（C2a–C2g，其中包括一篇 2026 年的综述）。我对这些论文和 DRIFT/RIEI 做了**关键词扫描**（不是精读）：都没有出现 underspecification 或源域模型选择的讨论；DRIFT 用 5 个 seed、取最终 epoch，而且只在 ManySig Day-1 上做（不跨日期）。第 4 节 E 方向在 RFFI 文献中**看起来**还没有被直接做过，但在提出方法之前，仍需按交接文档 §7 的要求精读并填写文献矩阵。
