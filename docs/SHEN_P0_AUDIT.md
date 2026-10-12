# P0 准入报告（2026-10-12，中国标准时间）

P0 通过，可以启动已授权的 P1。工作目录 `/home/keyan/code-y`，独立分支 `experiment/shen-p0-p1`；修复前源代码保留在 `source-snapshot`。旧权重/旧报告及 HDF5 保留，指标版本变更为 `macro_fixed_10_classes_v2`，不回填旧结果。

## 源码和环境

原 code-y 不是 Git 仓库，已建立仅跟踪源码与配置的本地 Git 历史。修复前 src/scripts/external 与 `/home/keyan/gai` 的 ba5d4d415ad00c3aa640acfcae335039a83074e9 快照一致；Git HTTPS 无凭证，GitHub App 调用返回 Unknown tool，因此未独立核实远端当前 main。完整比较、base commit、源码 hash 和环境见 `outputs/shen_evidence/p0_20261012/audit.json`。

四张 Tesla V100-SXM2-32GB 均可用；驱动535.129.03，驱动支持CUDA12.2，PyTorch2.5.1+cu121实际运行CUDA12.1/cuDNN90100。Python3.11.17，numpy1.26.4/h5py3.11.0/scipy1.13.1/sklearn1.5.2。

## 修复与输入界限

1. Macro-F1 改为固定十类的非加权均值；旧 finetune 两条验证路径及新训练器都汇总全验证集后一次计算。不逐 batch 平均 F1。六项回归测试包含不平衡类反例。
2. `fft_oob` 使用真正 OOB 掩码，独立于完整 fft。新增严格 IB-IQ 和 OOB-only 输入，在掩码后归一化，不利用被排除频段的尺度。
3. concat OOB dropout 同步清除拼接张量的 OOB 尾部；linear/cnn_stem 两种 embedder 均经过特征变化测试。关闭显式分支仍不移除旧完整 IQ/FFT 中的 OOB，报告未将 no_oob 称为 IB-only。
4. 新 Shen 入口仅接入受保护的 `ShenDevelopmentPackets`；不用 OSU/SigMF finetune 加载 Shen。原始数据只读；CFO 不再次补偿。
5. 作者 CIS 数值实现保持原算法，合成输入等价最大绝对误差0；新 CIS CNN 对应本地作者 non_adv_net 结构。其1,344万参数多于其他三种输入共享CNN的65.9万参数，报告明确区分架构与表示。
6. 全局 FFT/OOB token 是频率轴，IQ patch 是时间轴；旧混合轴索引/位置嵌入保留为历史对照，禁止作物理对齐解释。
7. 只读 HDF5 缓存按PID管理、pickle不带句柄、worker互不共享；随机行512包顺序暖缓存测量1.063秒降到0.319秒。新增两worker/序列化/原文件不改写回归测试。

## 数据与保护

本轮重新校验六个允许文件的SHA256，匹配既有协议。两台各8000包Day1 train、2000包Day2 validation、2000包Day1 diagnostic，共24000包。标签31–40映射0–9；仅Day2选模，同日诊断不混入验证。完整 manifest、24000个包的划分及 SHA、逐包频段能量表见 P0 输出目录。

没有读取Day3–4信号；没有重新读取普通多接收机的最终六台信号/标签/统计量。真实日期、session、作者发布原包身份未证实；Fs/BW本地适用性未证实。所有物理频率/OOB结论均以文献参数为条件。

IB范围 |f|≤62.5kHz，共1025bins。默认OOB |f|>75kHz，共6963bins，204bins保护区被两侧排除。按0/6.25/12.5/25kHz保护带生成完整能量敏感性表。12.5kHz条件下Day2 OOB占比中位数N210约0.0891%、RTL约0.1143%；25kHz时分别0.0299%与0.0558%。非零能量不证明PA指纹、独立设备信息或互补性；归一化会放大弱带外信号与噪声，接收机滤波、有限窗口泄漏均可能混杂。

## 四卡 smoke

有效记录目录 `outputs/shen_evidence/p0_20261012/four_gpu_smoke_v2`。每卡对应一种表示，每台接收机从Day1训练集取覆盖十类的10包；八组均有限loss、所有参数梯度有限、21–31步后训练准确率100%，loss下降超过50%，checkpoint重载输出逐位一致。shape分别为[B,2,8192]及CIS[B,1,52,126]。这个过拟合只验证训练管线，不是验证集识别性能。

首轮 `four_gpu_smoke` 保留三张卡的失败：严格CUDA确定性拒绝adaptive_avg_pool backward。已经在修复后用形状等价固定AvgPool1d(8)重新完成四卡检查。未删除失败实验。

## P1 准入后的运行规则

八个配置及矩阵在 `configs/shen_evidence/`。每卡顺序跑N210和RTL，两台各自训练。seed20261012，lr1e-3，Adam，最多30epoch，Day2全验证集Macro-F1早停patience7。初筛统一一个学习率预算；复验/超参数搜索另预登记。

P2/P3配置已准备，自动启动为false，详见 `SHEN_P2_P3_PLAN.md`。三seed复验只在首筛通过后登记；不能把本轮包装为未见接收机实验。


后续访问保护复核：额外禁止允许文件名通过 symlink 指向保留文件，七项数据/模型回归全部通过，原六项测试记录保留；另有一项临时目录中的协调器集成测试，验证先冻结学习率再生成两个新seed，且只启动P1。该测试不接触研究数据或GPU。完整P1自动协调器为 `scripts/finish_shen_p1.py`，最终状态和报告位于 `outputs/shen_evidence/p1_final_balanced`。
