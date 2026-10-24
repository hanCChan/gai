# 实现审计（rffi_post_a，C0）

日期 2026-10-24。工作树 `/workspace/gai/branches/rffi-post-stage-a-diagnostics-v1`，分支
`experiment/rffi-post-stage-a-diagnostics-v1`，基于 `dc805ff2a906e0951e0bfcf371f1b004433567db`。
**未提交、未推送。** Stage A 的代码、报告、decision、失败记录和已关闭的 reservation 字节未改
（`git status` 只显示新增路径和 `AGENTS.md` 末尾追加的本轮边界说明）。旧文档里“未实验/未推送”等语句属于各自写作时点，未改写；当前状态以
`reports/rffi_calibration/stage_A/REPORT.md`、`docs/rffi_calibration/PUBLICATION.md` 和本目录为准。

## 1. 新增文件

| 路径 | 作用 | 关键约束 |
|---|---|---|
| `src/rffi_post_a/diagnostic_stats.py` | H4 参数面板（plan §6.2 全部字段）、H2L 式共享协方差（原坐标与 PCA）、pooled Bayesian 矩 K0/U0/A0、稳定性块（留一 RX、行序前后半、WiSig 日期分解、行序滞后相关） | 只接收源记录；held RX 出现在源中即报错；单源 RX 报错；日期不计为 RX；面板必须逐位重现 H4 的 A=K+K/R+Ω，否则抛错 |
| `src/rffi_post_a/diagnostic_heads.py` | 20 个固定 slot；`adapt_variant` 只接收支持集，`score_terms` 只接收浮点特征张量，返回 score/maha/logdet | 未登记 slot 或重复 slot 报错；需要支持的 slot 在 m=0 报 N/A（不记 0）；非有限分数报错；全协方差只用 Cholesky/三角求解 |
| `src/rffi_post_a/diagnostic_matrix.py` | 每任务运行全部 slot + 8 个历史头 replay；锚等价检查；指标层诊断（中心误差、H4/H1 改判） | 标签只在 `_metrics` 与 `metric_layer_*` 中使用；manifest 不完整覆盖时报错 |
| `src/rffi_post_a/episode_controls.py` | R0 共同 batch + P_RX/P_MIX 路由、ProtoNet 损失、梯度探针 | 每周期覆盖每个 (RX,m) 一次；support 日期 0、query 日期 1；采样只用 numpy RNG，不碰 Torch dropout 流；覆盖不足报错，不换规则 |
| `src/rffi_post_a/artifact_guard.py` | `verify_lineage`、缓存键、只写一次的缓存读写 | 在反序列化权重/读特征前检查：预约激活且阶段为 D1、任务在白名单、checkpoint SHA、manifest SHA 前缀、角色（final/development/未知 RX）、held∩fit；缓存键伪造、符号链接逃逸、`..` 均拒绝 |
| `src/rffi_post_a/reporting.py` | D0 汇总、独立聚合重算、曲线审计、无依赖 SVG | 不打开 .pt/.npz/.npy/.h5/.dat |
| `scripts/rffi_post_a/audit_existing.py` | D0 | 已运行；只读 17 个 CSV/JSON |
| `scripts/rffi_post_a/cache_features.py` | D1 提取 | 未激活 → 不读任何东西退出（已测试） |
| `scripts/rffi_post_a/diagnose.py` | D1 分析（含 `--synthetic`） | 真实模式未激活即退出；结束时无条件关闭 reservation |
| `scripts/rffi_post_a/train_pilot.py` | R0 单拟合 | 只认 pilot_matrix 中的 job；需激活的 R0 预约 + 运行中的队列 claim；`--smoke` 只用随机张量写 /tmp |
| `scripts/rffi_post_a/run_pilot.py` | R0 调度 + 固定评估 | 预约必须恰为六个 job、max_fits=6、上限 ≤1.5 h、不重试；已有队列即拒绝；同折三臂初始化与 all_ids 必须一致才评估；结束关闭预约 |
| `scripts/rffi_post_a/verify_results.py` | 独立 score→metric→RX 聚合→R0 判定，以及 D1 score replay | 不导入被验证的 evaluator/reporting 聚合实现 |
| `configs/rffi_post_a/*.json` | datasets/roles（与 Stage A 字节一致）、registration（inactive）、diagnostic_matrix、pilot_matrix | 所有 allow_* 为 false，allowed_job_ids 为空 |

Stage A 的 `train.py`、`run_stage_a.py` 及其门禁未改；新 runner 读新的 config_dir，不能借旧脚本启动。

## 2. 测试（如实）

`/data/miniforge/envs/env-3.12/bin/python -m pytest tests/rffi_post_a tests/rffi_calibration -q` → **102 passed**
（新套件 41、Stage A 套件 61）。Stage A 套件中的 `test_stage_a_runner` GPU 推理回归在本机可见 GPU 时会运行
（随机张量，几秒，不读数据、不属于任何 reservation）。

| plan §10 组 | 覆盖的测试 |
|---|---|
| 历史等价 | D_CC=M_TQ=H4、M_SP=H1、C_PF_L=H2L（m=0/1/5，去行最大值后 ≤1e-8）；历史 H4-pooled 与 D_PP 只在 U 抽样项上不同（专门测试） |
| 数学 | pooled Bayes full 与 diag 条件化在对角矩时一致；m=0 走先验；1-shot H5b 共享 Q；给定参数下全协方差条件化对可逆仿射的概率不变、log score 差为共同常数 |
| 限定的不变性 | 对角 slot 对共同平移、维度置换、符号翻转不变；不对旋转/缩放下的重新估计管线提出要求 |
| 数值 | 非有限输入报错、类型检查、近共线特征下 K0 半正定与抖动记录 |
| 泄漏 | 打乱 query 标签所有 slot/replay 分数逐位不变；逐条/附加无关 query 不改变输出；scorer 拒收非张量输入 |
| 源统计 | held RX 进源报错；单 RX 报错；日期不增加 RX 数；稳定性块输出有界 |
| Episode | 周期覆盖、确定性、ID 唯一、日期规则、计数、MIX 每类 RX 互异且不固定；覆盖不足报错 |
| ProtoNet | 与作者 `euclidean_dist` 实现一致；support 有梯度；query 标签只改 loss 不改 logits；weight=0 单步与 C_MATCH 参数逐位相同；三臂初始化/dropout RNG/native loss 相同 |
| 选点 | train_pilot 固定写 last.pt 与 native_best.pt（smoke 三臂 init 与 all_ids SHA 相同）；held RX 不被加载 |
| 门禁 | 三个真实入口在未激活时于读取前退出；verify_lineage 的每条拒绝路径；缓存伪造/符号链接/写一次；run_pilot 只接受恰好的六-job 请求 |
| 调度 | 复用 Stage A JobQueue：恰好一次 claim、失败不重试、超预算拒绝 |
| 独立核验 | 合成 R0 输出的 score replay 与判定；篡改 rows.csv 的准确率可被发现；合成 D1 端到端（锚等价 + score replay） |
| BN 防护 | 两个 encoder 均无 BatchNorm |

未作为 CLI 执行：`diagnose.py --synthetic`、`train_pilot.py --smoke`（权限分类器间歇性拦截直接运行脚本；
其函数路径已分别由 `test_synthetic_d1_end_to_end_with_anchor_and_score_replay`、
`test_train_pilot_smoke_runs_three_arms_with_identical_ids` 在 pytest 中执行）。

## 3. 已知限制与风险

1. D1 的 C_RD_S 不等于 H1：前者为包合并残差 + 10% 各向同性收缩，后者为接收机等权、cell ddof 的对角方差；报告中分开列。
2. pooled Bayesian slot 要求每类 R 相同且每 cell n≥2，否则 N/A；当前登记任务满足，但缺 cell 时不会改用别的估计器。
3. 中心误差使用 query 两个固定半集的均值，含抽样噪声；报告其两半差异，不称 oracle。
4. R0 判定中的 `INCONCLUSIVE_OPTIMIZATION` 需人工根据训练日志判断，脚本不自动给出。
5. R0 成本是静态推算（未做 GPU 计时）。
6. `train_pilot.py` 的曝光计数按记录累加；梯度探针每 160 步一次，成本计入预算。
7. 第三套独立多-RX 数据与最近 RF 对照仍未解决（plan §8.2–8.3），本轮没有外网核验。
