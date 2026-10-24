# 决策日志（DECISION_LOG）

## D1 2026-10-24 OSU 角色：`TRANSFER_STRESS_ONLY`

证据（结构，不是分数）：
1. 每个方向只有 **1 个源物理 RX** → K（跨 RX 类中心协方差）在 OSU 内不可识别；`estimate_between_rx_covariance` 对单 RX 抛 `SingleReceiverError`（测试 `test_single_receiver_is_rejected_not_silently_zeroed`）。
2. 每 (TX,RX) 只有一份 20 s 连续录制；窗口是 8192 点网格窗 + 周期相干筛选，**不是检测出的包**；同一录制内的窗口不是独立 capture。
3. 两 RX 同日同场景（2022-03-08），无法证明支持/查询之间无共享事件；只能用固定原始样本块 + guard 隔离。
4. 25 TX 与 Shen 10 TX 无标签对应；native 10 类头对 OSU 为 N/A（`NativeHead` 显式拒绝）。

因此：OSU 只运行登记好的 `TRANSFER_POOLED_PRIOR` 变体——Shen 拟合的冻结 trunk + Shen 的类无关 `K_pool, Σ_pool`，OSU 源 RX fit 块提供 25 个新身份中心（`A = 2K_pool + Σ_pool/n_src`），目标 RX support 块做 m-shot，query 块计分，双方向。这是“冻结表示外部迁移”，**不是 OSU 内完整多 RX 机制**。

旧 G1 跨 RX 约 11.8% / 13.6% 不作为判定理由。若目标评分开始后 OSU 结果差，保留在正式外部表中，不替换数据。

未决：第三套**独立多 RX 主验证**仍未满足。替代候选（WiFi_RFFI、MobRFFI）本轮外网核验被环境拒绝，需用户决定是否授权核验/最小子集下载。

## D2 2026-10-24 WiSig 主协议 10 类

旧 8 类（TX 0–7）ckpt 只用于 Stage A0 历史兼容诊断，不与 10 类主表合并。

## D3 2026-10-24 支持/查询规则

WiSig 按日期（0 支持 / 1 查询）；Shen 按行号前后半 + guard；OSU 按原始样本块。规则在任何真实数据分数前写入 `registration.json`。

## D4 2026-10-24 Stage A 检查点

本机没有 Shen 任何 ckpt，也没有 WiSig 10 类 ckpt。严格 encoder-held Stage A 需要新的 6 次 T0 拟合（见 STAGE_A_EXECUTION_PLAN）。未授权前不拟合。

## D5 2026-10-24 Stage A Execution: `STOP_DECOMPOSITION`

The user's "全交给你去运行" delegated the minimal six T0 fits under 1.5
A100 GPUh, A0/A3, and external metadata/full-text checks. Optional T1 and
Stage B were not activated. Six fits completed, no failed fits/retries.

WiSig 1-shot H4 minus the strongest registered simple head:
w0 **-5.19375 pp** (H1), w1 **-10.925 pp** (strict cross-fitted H2*).
Shen H4 is below fixed H2L on all four selection RX at both 1 and 5 shots.
Shen H2*/H5c* are unavailable: each checkpoint has only one encoder-held RX,
and another task's encoder fitted that RX. Do not pool to claim strict tuning.

Twelve initial evaluations failed before rows/metrics (CPU/CUDA H0 mismatch).
The device-only fix was separately tested; all twelve NEW evaluations passed
independent score/metric/confusion/aggregate verification. Failed logs/costs
are retained. A final-ledger variable-shadowing error was recovered from
persisted complete progress; report serialization was corrected without
numeric reruns. Total conservative reservation **0.2539801585 A100 GPUh**,
including initial failures and 30 GPU seconds of GPU-test allowance.

Stop the current decomposition-necessity claim. Preserve simple calibration,
all negative results, horizon limitations and historical evidence. No Stage B,
optional T1, OSU numerical test, final/dev numeric access, or push.
See `reports/rffi_calibration/stage_A/REPORT.md` and `decision.json`.

## D6 2026-10-24 External Source Qualification Remains Incomplete

WiFi_RFFI author repository reachable, DataPort proxy tunnel 403; data license
and physical-RX coverage unverified. MobRFFI code documents 3/4 RX and repeated
dates/rounds, but the published S3 bucket returns NoSuchBucket; free processed
access and license unverified. Matching-title IoT-J source code is readable,
but model imports are missing, no full text obtained, IEEE record inaccessible.
No candidate downloaded or selected; third independent multi-RX evidence is
still unsatisfied. See `EXTERNAL_SOURCE_CHECK_20261024.md`.
