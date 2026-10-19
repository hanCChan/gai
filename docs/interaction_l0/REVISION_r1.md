# L0 修订记录 r1（2026-10-19，基于 `05340f7` 的外部审查）

外部审查结论为 `MODIFY_THEN_GO_L0`。本次只修正**工程正确性与统计公平性**问题；冻结的科学判定阈值（`summarize_l0.py` 中的 `RULES`）**一条未改**。所有修订发生在任何服务器 L0 输出产生之前。

| # | 审查发现 | 修正 | 研究影响 | 证据 |
|---|---|---|---|---|
| P0-1 | 内部 10 台 RX 的完整类别交集只有 TX {1,4,5,7}（日期 0/1）和 {0,1,4,5,7}（日期 2），源域是完整的 10×14；`complete_grid()` 会静默丢掉其余 TX | 新增 `coverage()`，并在图谱里输出 A0 覆盖审计（逐 RX×TX 计数、被丢弃的 TX、不足的单元）。A1 分成三栏：`source_all`（10 TX，不与内部比较）、`source_matched` 与 `internal_matched`（同一组交集 TX）。A3 记录 `n_tx_used` | 避免把类别组成的差异误判为接收条件变化 | 用 `internal_RX_class_coverage.csv` 复算确认；冒烟测试模拟同样的缺口，得到交集 {1,4,5,7}、`n_tx`=4 |
| P0-2 | "只推理"的说法不准确：每个模型要重拟合 9 个小分类头 | 文档和 manifest 中明确写为 `network_parameters_changed=False, head_refits=True, head_refits_per_model=9`；记录每次重拟合的首末损失 | 实验性质如实披露 | `checks.csv` 中有 `refits` 和 `refit_loss_decreased` |
| P0-3 | 物理干预缺少"请求施加量 vs 实际测得量"的验证；谱形对 CFO 的不变性只是声称 | 新增 `measured_change()`。对每种干预重新提取 CFO、谱形、SNR 代理和周期性，输出 `intervention_validation.csv`（误差、非有限值、低周期性、输入相对变化量、**CFO 干预造成的谱形泄漏**） | 防止把没有准确实现的干预当作阳性或阴性证据 | 合成数据实测：30 kHz 的 CFO 平移会让谱形变化 **0.064 RMS（约 0.28 dB）**，确认"严格 CFO 不变"不成立 |
| P0-3 | 擦除方向和源域身份保持用的是同一批样本（160–200 号包） | 方向和重拟合头**只用 `source_fit`**（WiSig 0–120 号包；Shen 每个 TX 的前 160 包）；身份保持在**互斥的** 160–200 号包上评价；内部 RX 只做评价 | 身份保持检查变为独立检查 | 见 `run_l0.py` 文档字符串中的数据分区说明 |
| P1 | 同秩不等于同强度：斜投影与正交投影的扰动幅度不同 | 每个擦除臂报告 `feature_change_DT` = E‖T(z)−z‖²/E‖z−z̄‖²，以及 TX/RX/交互能量保留率；`summarize_l0.py` 中只作描述性报告 | 干预强度差异在报告中可见 | 端到端测试中 `none` 臂的 D_T 恒为 0 |
| P1 | `spearman()` 用两次 argsort，没有处理并列值 | 改为平均秩（`average_ranks()`），常数输入返回 NaN；汇总脚本也改用它 | 物理图谱的相关系数和置换检验在有并列值时才正确 | 并列值单测：[1,2,2,3] vs [1,2,3,4] 得 0.9487（旧实现会给 0.8） |
| — | 硬件型号与空间位置互相混杂（源 RX 中只有 1 台 X310） | A2 输出 `A2_confounds`：各硬件型号的数量、同型号与异型号 RX 对的平均距离、两者的相关 | 任何 A2 的 p 值都必须和混杂情况一起解读 | 见 `atlas.json` |
| — | `--limit 2` 只覆盖两个 WiSig CE 模型 | 新增 `--preflight`：每种结构各选一个作业（WiSig 官方 CNN、WiSig ResNet、Shen CNN，有 R2-A 时再加一个） | 三种结构都经过真实检查 | GPU 端到端测试中三种结构都通过 |
| — | `phase_l0.sh` 冒烟后会自动开始全量 | 拆成 `phase_l0_preflight.sh`（到 `preflight.json` 为止）和 `phase_l0_full.sh`。`--full` 必须满足两个条件才会启动：preflight 判定为 `GO_FULL_L0`，且代码 SHA 与 preflight 时一致 | 先做人工核查，再花全量算力 | 端到端测试覆盖了"未获 GO 时拒绝启动" |
| 新发现 | GPU 上 `use_deterministic_algorithms(True)` 需要设置 `CUBLAS_WORKSPACE_CONFIG`；擦除后的特征是 float64，而分类头是 float32 | 在 torch 导入前设置环境变量默认值 `:4096:8`（与所有历史 launcher 一致）；进入 torch 前统一转成 float32 | 不修的话，第一个 GPU 作业就会崩溃 | 只用 CPU 测试发现不了；在本机 4×V100 上跑端到端时发现并修复 |
| 新增 | 用户要求使用 4 张 GPU | 全量阶段 `--shard i/N`，每张 GPU 一个分片；`merge_l0.py` 合并各分片（核对作业无重复、代码 SHA 一致、各分片的无模型验证一致），并从保存的概率计算集成和错误重叠（断言包对齐） | reserved GPUh 按各分片时间之和计 | 端到端测试：4 个分片、12 个作业，墙钟 40 s，合计 0.037 GPUh，峰值显存 5.6 GB |
| 工程门槛 | 合成数据量只有真实数据的 1/8 到 1/5 | 工程预算上限从 1.0 提高到 2.0 GPUh（在任何服务器输出之前，含 1.5 倍余量）；这不是科学阈值 | 避免只因预算估计导致 HOLD | `preflight.json` 中有 `full_estimate_reserved_gpuh_with_margin` |

**测试**（本机）：
- 5 组 CPU 测试全部通过。
- `smoke_run_l0_e2e.py` 在 **4×V100、torch 2.6.0+cu124** 上跑通真实 `run_l0.py` 的完整流程：preflight → 受保护的全量（4 个分片）→ 合并 → 汇总。
- 该测试使用合成缓存和随机初始化的 checkpoint，只检验工程行为，不检验科学结论。

**仍未验证**：真实 R0/R1 checkpoint 和真实缓存在服务器上的运行情况（本机没有这些资产）。
