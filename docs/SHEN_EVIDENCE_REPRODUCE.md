# Shen P0/P1 可复现运行

工作目录 `/home/keyan/code-y`，独立本地分支 `experiment/shen-p0-p1`。`source-snapshot` 保存所有修复前的源代码，对应旁边 gai 的 ba5d4d415ad00c3aa640acfcae335039a83074e9；当前 GitHub main 无法在线核验。旧报告、旧权重及 HDF5 未覆盖。

```bash
cd /home/keyan/code-y
PY=/home/keyan/venvs/shen-stage1/bin/python
$PY -m unittest discover -s tests -v
$PY scripts/audit_shen_evidence.py --output outputs/shen_evidence/p0_audit_new
$PY scripts/check_shen_representations.py --out-dir outputs/shen_evidence/cis_check_new
$PY scripts/launch_shen_evidence.py --stage smoke --output outputs/shen_evidence/smoke_new
$PY scripts/launch_shen_evidence.py --stage p1 --smoke-dir outputs/shen_evidence/smoke_new --output outputs/shen_evidence/p1_new
$PY scripts/report_shen_evidence.py --runs outputs/shen_evidence/p1_new --output outputs/shen_evidence/p1_new/report
```

实际首轮 smoke 的有效目录为 `outputs/shen_evidence/p0_20261012/four_gpu_smoke_v2`；首个 `four_gpu_smoke` 目录保留失败记录，原因是 PyTorch 的 adaptive average pooling 在启用严格 CUDA 确定性时没有 backward 实现，随后改为形状等价的固定 AvgPool1d(8)。正式 P1 用修复后的固定池化。所有新运行目录必须不存在，以免覆盖历史结果；失败保留日志，不静默重试。

矩阵 `configs/shen_evidence/matrix.json` 每卡顺序处理 N210-1、RTL-6；每台独立训练。E01/E02/E03 采用完全相同的 1D CNN 结构及参数量，E04 是作者 non_adv_net 架构的 PyTorch 移植，参数更多。CIS 数值变换保留原先等价验证的 SciPy float64 算法，HDF5 对 CIS 读取 float64，最终输入 float32。该闭集 CE/Adam 配方不声称复现作者所有训练/增强和论文性能。

所有模型使用 seed=20261012，Adam lr=1e-3，无 weight decay，batch=64，最多30 epoch，Day2 全集 Macro-F1 早停 patience=7、min_delta=1e-4。初筛均一个学习率，无模型专属调参；后续搜索需相同预算且登记全部候选。预登记复验 seeds=[20261012,20261013,20261014]，正式 seeds 再补 [20261015,20261016]。

Full IQ 在完整包上 RMS 归一化。严格 IB 先 FFT 保留 |f|≤62.5kHz、IFFT，再独立 RMS。OOB 先保留 |f|>75kHz，仅依据所保留频谱归一化，输出 real/imag 两通道且屏蔽处为零；不借用带内能量作尺度。保护区 (62.5,75] kHz 不进入两支。CIS 使用原包，具有不同信息边界，不能视为 strict IB。Fs=1MHz/BW=125kHz 均为文献假设，本地文件参数未独立证实。以上频率结论为条件结论；不重复 CFO 补偿。

每次训练记录 config/协议 SHA、源码 commit、指标版本、hardware、history、选中 epoch、参数量、训练秒数、吞吐、checkpoint。`predictions.csv` 中 sample_id=相对 HDF5 路径:row，十个类的概率保留，`errors.csv` 单独列错误。model 吞吐只计已变换/驻 GPU 输入的 forward；pipeline 吞吐含 HDF5 读取、预处理和 transfer。每次同步计时，因此不是异步最大吞吐 benchmark。模型验证和重新加载 checkpoint 完全使用 Day2；没有最终测试入口。

HDF5 worker 只持有一个只读文件句柄及 2MB 缓存；绝不整份读 IQ 到内存。随机行 512 包测得 1.06秒/0.32秒 是顺序暖缓存比较，不能当成四卡完整性能提升倍数。

P2/P3 的配置不会被本启动器执行；细节见 `SHEN_P2_P3_PLAN.md`。Session/实际日期/原包身份和既有 OOB 诊断接收机重叠仍未核验。不能用本轮两台分别训练的结果宣称零样本跨接收机性能。


## 本机完整首轮与自动复验

本次额外登记了统一 lr=3e-4 候选，目录 `outputs/shen_evidence/p1_lr0003`；原 lr=1e-3 目录是 `outputs/shen_evidence/p1_seed20261012`。`scripts/finish_shen_p1.py` 等待这两份矩阵全部成功，再仅根据 seed20261012 的开发 Macro-F1 逐输入/接收机冻结学习率，保存 `selection.json`。当两台 OOB-only 均超过10%机会水平时，执行 seed20261013/20261014 的全四输入复验。`p1_final_balanced/pipeline_status.json` 记录阶段；最终报告位于 `p1_final_balanced/report/REPORT.md`，初筛失败则显式记为 failed，不继续复验。

```bash
$PY scripts/launch_shen_evidence.py --stage p1 --matrix configs/shen_evidence/lr_search_matrix.json --wait-for outputs/shen_evidence/p1_seed20261012 --smoke-dir outputs/shen_evidence/p0_20261012/four_gpu_smoke_v2 --output outputs/shen_evidence/p1_lr0003
$PY scripts/finish_shen_p1.py --output outputs/shen_evidence/p1_final_balanced
```

这些命令对应本机已启动的目录，重新执行应选择新的输出目录或复制项目后重新设置协调器的首筛路径；不能覆盖已有运行。报告同时保留两个学习率所有候选，仅所选配方进入三seed均值。多次尝试同一个seed不能计为多个seed。`calibration.csv` 为开发集未校准模型的ECE/NLL/Brier，`paired_case_examples.csv` 记录每对方法的前20个差异例。


复验阶段按首筛实测耗时分配到四张同型号V100，避免CIS全部排在单卡而其他卡闲置。每个输入/接收机的配方在新seed开始前冻结；GPU分配不改变输入、seed或训练预算，成本计划保存为 `gpu_cost_plan.json`。协调器已作为独立后台进程运行，日志是 `outputs/shen_evidence/p1_final_balanced.log`，PID记录在 `p1_coordinator_pid.json`。原等待协调器目录 `p1_final` 标记为superseded，仅替换了等待进程，训练未中断。
