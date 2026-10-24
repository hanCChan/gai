# 实现审计（IMPLEMENTATION_AUDIT）

日期 2026-10-24。所有代码在新 worktree `/workspace/gai/branches/rffi-budget-calibration-v1`（分支 `experiment/rffi-budget-calibration-v1`，基于 `b6ce3c2`）。历史 P2E 源码、报告、Errata 未改动。

## Execution Update

The sections below preserve the pre-execution implementation audit. Stage A
was subsequently authorized and executed on 2026-10-24. CPU preflight and
synthetic dry run passed; six T0 fits and twelve corrected A0/A3 evaluations
completed. Initial GPU evaluation device errors and ledger/report closure
errors were fixed and disclosed, not confused with the prior permission issue.
Final tests: 60 passed, 1 skipped on CPU; the GPU integration regression passed
separately. Decision **STOP_DECOMPOSITION**, reservation closed, no push.
See `EXECUTION_AMENDMENT_20261024.md` and
`reports/rffi_calibration/stage_A/REPORT.md` for the authoritative current state.

## 1. 文件

| 模块 | 职责 | 关键点 |
|---|---|---|
| `src/rffi_calibration/statistics.py` | 单元矩、K/Σ/U 估计、对角后验预测、对角/全协方差 Gaussian 打分、PSD 投影、带记录抖动的 Cholesky、OSU 迁移先验 | float64；torch 实现（T3 可反传）；单 RX 抛 `SingleReceiverError`；缺类抛 `MissingClassError`；无显式求逆 |
| `src/rffi_calibration/heads.py` | H0–H5c，统一 `fit / adapt / predict_proba` | `predict_*` 只接收 query 特征；全部类完整得分（Mahalanobis + logdet + 先验）；需要支持而缺类时报错，不静默回退 |
| `src/rffi_calibration/sampling.py` | `SampleRecord` / `RecordTable`、稳定 sample ID、固定 query + 嵌套支持 manifest、窗口块完整性检查 | ID = dataset\|capture\|tx\|date\|原始位置；重复 ID 报错；manifest 是元数据的纯函数（SHA 可复现） |
| `src/rffi_calibration/evaluator.py` | 头 × 预算 × draw 评估、指标、源内参数选择、留一交叉拟合、配对差 | query 标签只在 `score_metrics` 中查表；非有限分数抛错 |
| `src/rffi_calibration/episodes.py` | T0–T3 共享 episode schedule 与 query 损失 | 伪目标 RX 轮换；T3 源统计仅来自 reference；T1 调用历史 `p2_losses.ccalign` 原函数 |
| `src/rffi_calibration/encoders.py` | WiSig 80-D / Shen 128-D wrapper | Shen 用 forward pre-hook 捕获 classifier 输入，原 forward 不改；类数不匹配必须显式 `new_head=True` |
| `src/rffi_calibration/registry.py`, `access.py` | 路径变量解析、标签映射、封存与预约检查 | 数值访问前检查角色/日期/真实路径；未激活预约拒绝一切数值读取；访问日志不含数值 |
| `src/rffi_calibration/datasets/{wisig,shen,osu}.py` | 三数据集适配 | OSU 先定原始样本块再放窗口；窗口不跨块，带 guard |
| `src/rffi_calibration/queue.py` | SQLite 事务队列 | 白名单 + 预算守卫；completed 需 exit 0 + 产物 + verifier；失败不原地重跑；SIGTERM → cancelled |
| `scripts/rffi_calibration/frozen_heads.py` | Stage A 执行器（含 A0 历史 8 类 ckpt、`--synthetic` 干跑） | 未激活预约直接退出、不读数据；ckpt 状态/SHA 校验；评估 RX 若在拟合 RX 中则标 INTERNAL_DIAGNOSTIC |
| `scripts/rffi_calibration/train.py` | T0–T3 训练（只认 `formal_matrix.json` 中且被激活预约列出的 job） | `--smoke` 只用随机张量 |
| `scripts/rffi_calibration/preflight.py` | 元数据预检 | 不读信号数值；封存 RX 只做存在性检查 |
| `configs/rffi_calibration/*.json` | 数据、角色、协议、作业矩阵 | 预约 `active=false`，`allow_feature_extraction=false`，`max_fits=0` |
| `tests/rffi_calibration/*` | 合成测试 | 见 §2 |

## 2. 测试状态（如实）

- **已执行并通过：56 项**（`pytest tests/rffi_calibration -q` → `56 passed in 9.09s`）：`test_statistics.py` 13、`test_heads.py` 11、`test_sampling_and_leakage.py` 10、`test_episodes.py` 6、`test_infra.py` 8、`test_train_smoke.py` 8（WiSig/Shen × T0–T3 真实 wrapper 单步训练，随机张量）。
- **已编写、本轮未能执行**：`preflight.py`、`frozen_heads.py --synthetic`、`train.py --smoke`。原因：环境权限分类器对直接运行脚本反复返回 “Stage 2 classifier error”（不是代码失败）；IDE 诊断无错误。需用户本地运行（STAGE_A_EXECUTION_PLAN §1）。
- 修改历史中被测试抓到并修正的问题：`cell_moments` 高级索引用法 → `index_select`；`SimpleCNAPSFrozen` 缺 `fit`；若干测试本身的错误期望（K 真值应为实际抽到的接收机偏移的方差；Monte Carlo 条件方差需按全方差公式；对角头只声明缩放/置换等变，而非任意仿射）。

## 3. 关键正确性证据

- H2L 与历史 `p2e_eigenrx.EigenRX` + `m/(m+2)` 的逐类对数似然差（减去行最大值后）≤1e-8，argmax 完全一致 → 已实证基线被忠实保留。
- 对角后验 = 独立实现的全协方差 Gaussian 条件化（对角输入）至 1e-10；m=0 → 先验，m→∞ → 支持均值、Q→Σ。
- 对角对数密度与 `scipy.stats.multivariate_normal` 一致至 1e-9；全协方差 Cholesky 路径同样一致。
- T3 后验风险对 reference / support / query 三支均通过 `torch.autograd.gradcheck`。
- 打乱 query 标签：所有头所有 (m, draw) 的 logits 逐位相同，仅指标变化。
- 无类别信息特征：所有头准确率 <0.45（4 类）→ 不会重现 P2E A1 的 100% 泄漏。

## 4. 风险与未覆盖

1. **Shen 无任何本机 ckpt，WiSig 无 10 类 ckpt** → 严格 Stage A 必须先做 6 次 T0 拟合（需授权）。
2. `train.py` 的 T0 使用新 episode schedule，而不是历史 `balanced_schedule`；与历史 CE 数字不可直接等同。步数按包暴露匹配，属于预登记推算，尚无 horizon pilot。
3. H4 K 的估计：WiSig fit 10 RX、Shen fit 仅 4 RX → `a_K = 3/11`，Shen 上 K 主要靠池化，类相关性的检验力弱。这会降低 H4 与 H4-pooled 的可区分度，需在报告中写明。
4. Shen 支持/查询只能按 HDF5 行号前后半切分，时间独立性未证。
5. OSU 迁移的 `2K_pool` 假设未验证；前端（窗口 + 相干筛选）与 Shen 包级 CIS 是否可比仅在 8192 点 / 1 MS/s 层面成立。
6. H5b、H5c 是**冻结头移植**，不是 Simple CNAPS / MetaQDA 复现；RF 同任务方法（IoT-J 2025 Gaussian prototype 等）尚未取得全文、未实现。
7. `evaluator.paired` 的 t 分位为查表近似（n≤14 精确到两位小数，其余用 2.0）；RX 数少时区间只作描述。
8. 第三套独立多 RX 数据仍未落实（DECISION_LOG D1）。
