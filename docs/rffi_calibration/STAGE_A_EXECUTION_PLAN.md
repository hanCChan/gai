# Stage A 执行计划与授权请求（STAGE_A_EXECUTION_PLAN）

**本文件是请求，不是授权。** 当前 `registration.json` 预约为 `active=false`；所有脚本在未激活时直接退出、不读数据。

Status update: preserve this as the original preregistered request. The user's
2026-10-24 delegation subsequently authorized the minimal six T0 fits.
Stage A is now completed and stopped, with a CLOSED reservation. Actual
bounded launchers are `run_stage_a.py` and `run_stage_a_evaluation.py`;
the original direct-training commands below are archival and are now blocked
without a running queue claim. See the execution amendment and final report.

## 1. 先由用户在本机运行的零风险检查（无数据数值访问、无 GPU）

```bash
cd /workspace/gai/branches/rffi-budget-calibration-v1
PY=/data/miniforge/envs/env-3.12/bin/python
$PY -m pytest tests/rffi_calibration -q                                   # 已验证：56 passed
$PY scripts/rffi_calibration/frozen_heads.py --synthetic --out /tmp/rffi_stageA_dry   # 合成干跑，打印 H0–H5 表
$PY scripts/rffi_calibration/preflight.py --out reports/rffi_calibration/E0_preflight  # 只读元数据/标签计数
```

`preflight.py` 会读取 source RX 缓存中的**标签与位置数组**（不读信号）以核对覆盖；封存 RX 只做文件存在性检查。

## 2. 请求授权的最小 Stage A

| 子阶段 | 内容 | 拟合数 | 预测资源 | 产出 |
|---|---|---|---|---|
| A0 | 6 个历史 8 类 WiSig ccalign 搜索 ckpt（w0/w1 × λ{0.3,1,3}），对各折 selection RX 做 H0–H5 | **0** | GPU 推理 + CPU，约 10–20 min | 8 类历史兼容诊断；不进门槛 |
| A1 | WiSig 10 类 T0：w0、w1 各 1（seed 20261013，2400 步） | 2 | ≈0.5 GPUh（预测 900 s/个） | 10 类 encoder-held ckpt |
| A2 | Shen T0：A/b210_1、A/rtl_1、B/b200_mini_1、B/pluto_1 各 1（220 步） | 4 | ≈0.45 GPUh（预测 400 s/个） | 4 个 encoder-held ckpt |
| A3 | 对 A1/A2 每个 ckpt 运行 `frozen_heads.py`（20 draws × m∈{0,1,2,5,10}） | 0 | CPU 为主，约 30–60 min 总计 | 门槛表、配对差、逐 RX/TX 记录 |

合计 **6 次拟合，预约上限建议 1.5 A100 GPUh**（预测 ~0.95 GPUh + 余量），4 卡并行时墙钟约 15–25 min。不含 T1/T2/T3，不含 OSU、development、final、日期 3。

可选（需另行勾选）：Shen T1（ccalign λ=3，P2E LORO 所选）4 次拟合，+0.45 GPUh —— 让 Stage A 也在历史最强 encoder 上检验头。

## 3. 激活方式（用户批准后由我修改并提交，或用户自行修改）

```json
"reservation": {"stage": "A", "active": true,
  "allowed_job_ids": ["A_wisig_w0_T0_s20261013","A_wisig_w1_T0_s20261013",
                      "A_shen_A1_T0_s20261013","A_shen_A2_T0_s20261013",
                      "A_shen_B1_T0_s20261013","A_shen_B2_T0_s20261013"],
  "max_fits": 6, "max_reserved_gpu_hours": 1.5,
  "allow_feature_extraction": true, "allow_final_numeric_access": false, "allow_failed_fit_retry": false}
```

执行命令（A1/A2 每卡一个）：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/rffi_calibration/train.py --job A_wisig_w0_T0_s20261013
CUDA_VISIBLE_DEVICES=1 $PY scripts/rffi_calibration/train.py --job A_wisig_w1_T0_s20261013
CUDA_VISIBLE_DEVICES=2 $PY scripts/rffi_calibration/train.py --job A_shen_A1_T0_s20261013   # 之后 A2
CUDA_VISIBLE_DEVICES=3 $PY scripts/rffi_calibration/train.py --job A_shen_B1_T0_s20261013   # 之后 B2
# A3
for j in outputs/rffi_calibration/A_*; do
  $PY scripts/rffi_calibration/frozen_heads.py --ckpt $j --out reports/rffi_calibration/E4_frozen_heads/$(basename $j); done
# A0
for d in /workspace/rl1/gai_retrain/outputs/receiver_p2_wisig/search/w?_ccalign_lam*; do f=$(basename $d | cut -c1-2)
  $PY scripts/rffi_calibration/frozen_heads.py --legacy-wisig8 $d --fold $f --out reports/rffi_calibration/E4_A0/$(basename $d); done
```

失败不重试、不换种子；失败记录保留。

## 4. 判定（预先写定）

- 门槛只在 encoder-held selection RX 上、用交叉拟合的 H2*/H5c* 计算（EXPERIMENT_PROTOCOL §8）。
- 结果三选一：**GO**（H4 达门槛）→ 申请 Stage B（T0–T3 配对 pilot，再 48 次正式）；**MODIFY**（H4 只在部分预算/数据集占优）→ 报告并请用户决定；**STOP_DECOMPOSITION**（H4 不优于简单头）→ 保留简单校准 + 诊断路线，不加新 loss。
- 任何结果都不触发自动进入下一阶段。

## 5. 仍需用户决定

1. 是否批准 §2 的 6 次拟合与 1.5 GPUh（以及可选 Shen T1 4 次）。
2. 第三套独立多 RX 数据：是否授权核验 WiFi_RFFI / MobRFFI 的入口与许可（本轮外网核验被环境拒绝）。
3. 是否需要我先取得 IoT-J 2025 Gaussian prototype RFFI 全文以实现同任务 RF 对照（需外网）。
