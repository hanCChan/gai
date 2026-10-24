# 数据就绪状态（DATASET_READINESS）

日期 2026-10-24，本机 4×A100-40GB。来源：文件树/大小、JSON 配置、代码阅读、2026-10-23 下载校验记录。**本轮未读取任何信号数值。**

Execution update (same date, after explicit user delegation): the table below
is the pre-execution inventory. Six new T0 checkpoints now exist locally;
source-only Stage A numerical access was performed, with 68 matching input
hashes and no dev/final/OSU numeric access. The final decision is
STOP_DECOMPOSITION. External candidates were checked but remain unqualified;
see `EXTERNAL_SOURCE_CHECK_20261024.md`. Reservation is closed.

| 项目 | WiSig ManyRx | Shen LoRa | OSU LoRa |
|---|---|---|---|
| 体制 / 输入 | WiFi，非均衡 IQ `float32[B,1,256,2]` | LoRa，Strict CIS `float32[B,1,52,126]` | LoRa，cf32 1 MS/s 连续录制 → G1 Strict CIS |
| TX / 类别 | 10（主协议 10 类；旧 8 类仅历史兼容） | 10（原始 31..40 → 0..9） | 25（Device k → k−1；与 Shen 标签无关） |
| 物理 RX | 32（14 source / 10 development / 8 final） | 10 source 已缓存；4 dev + 6 reserved 未缓存 | 2（RX1, RX2；USRP B210） |
| 原始资产 | zip 1,310,655,283 B，SHA `b0eca705…` 与 10-13 登记一致（官方字节等价未证） | 52 根 h5 + Test/ + Train/（30 GB），用户上传 zip CRC 通过 | 50×160,000,000 B `.dat`，全部 SHA256 与 `OSU_download_independent_verification.json` 一致（10-23） |
| 已有缓存 | `RETRAIN_ROOT/outputs/wisig_w01/cache`：72 npz = 24 非 final RX × 日期 {0,1,2}；**final 8 RX 与日期 3 从未缓存** | `RETRAIN_ROOT/outputs/shen_p2a8/g1_cache_v1`：10 source RX × 2000 行（每 TX 200） | 无（G1 cache 未在本机重建） |
| 样本单位 | `cache_position`（单元内位置，不是时间戳） | `packet`（HDF5 行号；行序未证为时间序；根文件无日期/会话） | `window`（8192 点网格窗 + 周期相干筛选，**不是包**） |
| 支持/查询隔离 | 不同采集日期（date 0 支持，date 1 查询） | 同一 (RX,TX) 内按行号前/后半 + guard 2；依赖性限制已披露 | 固定原始样本块：fit [0,.40)、support [.45,.60)、query [.65,1.0)，guard 8192 |
| 编码器检查点（本机） | 24 个历史 8 类搜索 ckpt（ccalign/rxlogit/rxnull/rxnull_rand × λ × w0/w1），**无 CE、无 10 类** | **无**（R0/R1、P2E formal 均不在本机） | 不适用（冻结 Shen trunk 迁移） |
| 不可访问 | final 8 RX、日期 3、equalized | dev/reserved RX（需先审计访问史再建缓存）、Test/drift 文件未登记 | — |
| 当前可执行 | Stage A0：8 类历史 ckpt 的 encoder-held 诊断（0 次拟合）；Stage A 10 类需 2 次 T0 拟合 | Stage A 需 4 次 T0 拟合 | 仅 TRANSFER_STRESS_ONLY（见 DECISION_LOG） |

## WiSig 细节

- 源内两折（`RETRAIN_ROOT/configs/receiver_p2/wisig_folds.json`，10-23 登记，早于任何校准分数）：
  w0 fit `[2,5,6,10,12,13,22,26,28,31]` / selection `[1,7,16,30]`；w1 fit `[7,30,1,10,12,13,16,28,31,22]` / selection `[6,26,2,5]`。
- 历史 8 类搜索 ckpt 血缘（`train_wisig_p2.py`）：梯度 = fit RX、日期 0–1、位置 0:120；epoch 选择 = **fit RX** 位置 120:160 的接收机等权 CE。selection RX 不进梯度、不进选 epoch → 对 selection RX 是 encoder-held。但只含 TX 0–7。
- 覆盖：登记 `hardware_and_coverage` 显示 14 个 source RX 的最小单元 200；10 个 development RX 中 9 个存在缺类（`min_development_cell` 0），只有 1 个 ≥156。**development RX 不能进入 10 类完整表**，只能进部分覆盖表——这一点在任何分数之前确定。
- 日期 2 已在旧开发通路出现，不是盲测。

## Shen 细节

- 两组外层五 RX：A `b210_1,rtl_1,rtl_3,rtl_6,rtl_9`，B `b200_mini_1,pluto_1,rtl_2,rtl_5,rtl_7`；均已多次用于研发 → 只算配对机制证据。
- Stage A 源内 selection 任务（各组前两个 RX，按登记顺序，分数前固定）：A/b210_1、A/rtl_1、B/b200_mini_1、B/pluto_1；每个任务用组内其余 4 RX 拟合。
- scaler：仅拟合包（每 TX 前 160 行）上的全局 mean/std，随 ckpt 保存。

## OSU 细节

- sigmf：`cf32`、`"1 MS/s"`、`sample_count` 20,000,000、915 MHz、2022-03-08；每 (TX,RX) 只有一份连续录制。
- 块计划后每 TX 每 RX 约 fit 975 / support 364 / query 852 个网格窗（相干筛选前；精确数由 preflight 输出）。
- 已知元数据矛盾（RF 描述）不用于修改带宽/频率假设。

## 第三数据集替代候选（未下载）

按任务书顺序：WiFi_RFFI（IEEE DataPort，HTTP 403 未核）、MobRFFI（S3 下载费用提示、MATLAB 预处理，未核）、SiamRFFI（单 RX，仅场景压力）。本轮外网请求被环境权限拒绝，**未做候选核验，未下载**。OSU 判定为 TRANSFER_STRESS_ONLY 而非 REPLACE_PRIMARY 的理由见 DECISION_LOG；若用户要求第三套“独立多 RX 主验证”，需单独授权替代数据核验。
