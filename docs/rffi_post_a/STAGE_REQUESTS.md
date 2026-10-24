# D1 / R0 / score-replay 授权请求（STAGE_REQUESTS）

**本文件是三份相互独立的请求，不是授权。** 当前 `configs/rffi_post_a/registration.json` 的 reservation 为
`active=false`、`allowed_job_ids=[]`；`requests_not_active` 中的 D1/R0 草案不会被任何脚本读作许可。
Stage A 的 reservation 保持关闭，其剩余配额不使用。

## A. 既有逐 query 分数 replay（D0 补充，零拟合、零 GPU）

| 项 | 范围 |
|---|---|
| 文件 | `/workspace/gai/branches/rffi-budget-calibration-v1/reports/rffi_calibration/E4_frozen_heads/A_*/gpu_head_fix_v2/predictions_rx*.npz`，共 12 个（WiSig 8 + Shen 4），合计约 372 MB；只用 `allow_pickle=False` 读取 |
| 不含 | E4_A0 的 24 个旧 8 类文件、任何信号/特征/权重 |
| 产出 | score→accuracy/macro-F1/NLL/Brier/ECE 独立重算并与 rows.csv 对账；H4/H1/H4-pooled 的混淆矩阵、预测类占比、正确/错误置信度分布 |
| 用途限制 | 类占比失衡只作线索，不用于调先验或选头；query 标签只在指标层使用 |
| 成本 | CPU，预计 <5 分钟 |

## B. D1：固定冻结特征诊断（零拟合）

**作业清单（静态）**：六个 Stage A T0 checkpoint × 原 12 个 held-RX 目标任务，见 `configs/rffi_post_a/diagnostic_matrix.json`。

| task | checkpoint SHA256（前 12 位） | 源 fit RX | 目标 held RX |
|---|---|---|---|
| A_wisig_w0_T0_s20261013 | f83205c68d99 | 10 RX，日期 0/1，位置 0:120 | 1, 7, 16, 30 |
| A_wisig_w1_T0_s20261013 | 2c7bd9eab12a | 10 RX，日期 0/1，位置 0:120 | 6, 26, 2, 5 |
| A_shen_A1_T0_s20261013 | 742722a8d262 | rtl_1, rtl_3, rtl_6, rtl_9（每 TX 前 160 行） | b210_1 |
| A_shen_A2_T0_s20261013 | 395d92dd261f | b210_1, rtl_3, rtl_6, rtl_9 | rtl_1 |
| A_shen_B1_T0_s20261013 | 0205ac6594d8 | pluto_1, rtl_2, rtl_5, rtl_7 | b200_mini_1 |
| A_shen_B2_T0_s20261013 | bfb6c858cc8a | b200_mini_1, rtl_2, rtl_5, rtl_7 | pluto_1 |

**固定矩阵**：20 个 slot（K/Σ 2×2 共 4、中心/密度 6、坐标/相关性 10；其中 D_CC=M_TQ=H4、M_SP=H1、C_PF_L=H2L 为锚，
去重后 19 个独立计算）+ 8 个历史头 replay（H1、H4、H2L、H4-pooled、H4-AK、H3、H5a、H5b）；预算 0/1/2/5/10；
原 20 个 support draw；原 query；均匀类先验；无超参数搜索。另含参数面板、源 RX 留一稳定性、行序前后半、
WiSig 日期分解、行序滞后相关、中心误差（指标层）、H4/H1 改判方向。

**执行顺序**：`cache_features.py --device cpu`（先核对 checkpoint SHA、由元数据重建 manifest 并比对 Stage A 前缀、
角色检查，全部通过后才反序列化权重和读源/目标缓存）→ `diagnose.py`（纯 CPU）→ `verify_results.py --d1`
（独立 score replay）→ 无论结果如何，关闭 reservation。

**成本依据与申请**：Stage A 对同样的特征做提取 + 22 个头 × 20 draw 的完整评估，每任务 GPU 预留 14–62 s
（`stage_A_execution/gpu_head_fix_v2/evaluation_execution.json`）。D1 的源/目标样本量与之相同（WiSig 每任务
24 000 源 + 16 000 目标；Shen 6 400 源 + 2 000 目标），小型 CNN 在 CPU 上提取即可。

- 建议：**D1 纯 CPU、0 GPUh**；预计 CPU 墙钟 30–60 分钟（全部 slot × 81 次 adapt × 12 任务 + 稳定性块）；
  磁盘：缓存 float64 特征约 WiSig 2×40 000×80×8 B ≈ 51 MB，Shen 4×8 400×128×8 B ≈ 34 MB，加 D1 分数文件约 300 MB。
- 若你希望用 GPU 提取：另设上限 **0.5 A100 GPUh**，提取完立即释放卡再做 CPU 分析。
- 权限：仅源 fit RX 的 fit 位置和 12 个原 held-RX 目标；不含 development、final、日期 3、equalized、OSU。

激活方式（批准后由我写入）：把 `requests_not_active.D1` 复制为 `reservation` 并设 `active=true`。

## C. R0：WiSig 三臂可校准表征小试验（6 次新拟合）

**作业清单（静态）**：`configs/rffi_post_a/pilot_matrix.json`

| job | 折 | 臂 | seed |
|---|---|---|---|
| R0_wisig_w0_C_MATCH_s20261024 | w0 | 配对 CE | 20261024 |
| R0_wisig_w0_P_MIX_s20261024 | w0 | CE + 接收机混合 ProtoNet | 20261024 |
| R0_wisig_w0_P_RX_s20261024 | w0 | CE + 接收机成组 ProtoNet | 20261024 |
| R0_wisig_w1_C_MATCH_s20261024 | w1 | 配对 CE | 20261024 |
| R0_wisig_w1_P_MIX_s20261024 | w1 | CE + 接收机混合 ProtoNet | 20261024 |
| R0_wisig_w1_P_RX_s20261024 | w1 | CE + 接收机成组 ProtoNet | 20261024 |

配方（冻结）：OfficialCNN 10 类、80-D、Adam 5e-4/eps 1e-7、clip 5、原 regularization、训练预算 {1,5}、
其他 RX 每类日期 0/1 各 8 条 reference、query 每类 8 条、episode 权重 1、温度 1、4800 步、每 32 步评估、
主 checkpoint `last.pt`、次要 `native_best.pt`。同折三臂初始化 SHA 与 all_ids 链 SHA 必须一致（runner 校验）。
评估：Stage A 原 manifest，主头 H1e，固定次要头 H1、H2L、H5a、H5b、H5c[k0=1]、H0；判定规则见 pilot_matrix
`decision_rule`，由 `verify_results.py` 独立实现。

**成本依据与申请**：Stage A WiSig T0 2400 步训练循环 30.6–31.6 s、预留 36.1–38.7 s；A3 WiSig 评估 23.6–51.6 s。
R0 步数加倍、batch 规模相同、另有 ProtoNet loss 与每 160 步一次梯度探针，静态预测每个拟合 ≤300 s（含加载），
12 次评估（6 拟合 × 2 checkpoint）每次 ≤120 s。预测合计约 **0.9 GPUh 上界**（6×300 + 12×120 = 3 240 s），
申请上限 **1.5 A100 GPUh**、恰好 6 个拟合、不重试、不换 seed、不加第七次拟合、不延长步数。
这是静态推算，没有做 GPU 计时；若你要求先计时，需另行批准一个不保存 checkpoint 的计时任务。

激活方式：`requests_not_active.R0` → `reservation`，`active=true`；运行 `run_pilot.py --gpus 0 1 2 3`；结束后自动关闭 reservation。

## 三项之间的关系

A 只补充 D0；D1 回答“当前冻结特征上 H4 为什么退化”；R0 回答“训练能否改善简单原型、接收机成组是否必要”。
任何一项的结果都不自动触发另一项；R0 不以 D1“救回 H4”为前提。
