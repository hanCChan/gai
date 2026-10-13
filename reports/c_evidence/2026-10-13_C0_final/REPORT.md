# C0/C1/C2 多窗口证据可靠性审计：本轮决策

**决策：STOP_INVALID_GROUPING。C0完成，C1/C2未启动；训练与新增推理均为0。**

按用户最后答复，保留“真实录制顺序中的相关错误与序贯决策”问题，不改做
无序离线包融合。Shen已有完整预测资产，但真实时间顺序与session不可核验，
不能把HDF5行号包装成连续观测。OSU原跨日期实验另有资产缺失，因此也无法
逐窗口/逐capture复现原稿Table II。当前不判断相关性机制成立或基线已充分。

## 本轮范围

独立worktree `/home/keyan/code-y-c-audit`，分支 `experiment/c-evidence-audit`。
原P0/G1/G2-0/VIRP/EISR未改。P0同步路线保持停止。
执行原稿核验、原跨日期资产检查、Shen十源HDF5元数据及既有预测缓存头部
审计、文献差异核验和计划更新。不读取IQ值，不计算既有预测的新K性能，不
加载模型，不训练、不下载大型数据，也没有开发/锁定RX或日期信号访问。

## 原稿与Table II

用户补充的 `OOB_Guided_Cross_Attentive_RF_HSTU_Hybrid_Modeling_for_Robust_LoRa_Device_Authentication__15_.pdf`
已经核验题名、平均logits定义、Day1–3/Day4/Day5协议和第8页Table II。
此前P0报告中“原稿本地未找到”是当时的资产状态；本轮补齐来源，不改写旧
封存报告。原稿与拒稿信在仓库外私有目录，公开材料仅包含SHA和事实摘要。

从本地 `metrics_all.csv` 的逐seed汇总重新计算得到：

|模型|历史seed数|窗口Accuracy|文件Accuracy|未取整聚合差值|本机正式权重|
|---|---:|---:|---:|---:|---:|
|CNN-IQ|5|43.5352%|54.1667%|+10.6315 pp|1/5|
|RF-HSTU无OOB|3|39.4206%|66.6667%|+27.2461 pp|0/3|
|原RF-HSTU完整模型|5|41.4876%|75.0000%|+33.5124 pp|5/5|

这些均值与原稿取整值一致，证实反转存在于历史报告。**这不是逐窗口或逐文件
复现实验**：正式 `step1_phase7_clean/outputs` 本机不存在，没有该协议逐窗
预测、完整logits或逐capture结果。无法由汇总值恢复哪一个文件被纠错。

另核实旧汇总的标准差采用 `ddof=0`，如完整模型文件Accuracy标准差为
5.2705pp，与原稿5.3一致；样本标准差 `ddof=1` 为5.8926pp。本轮两种定义
都明确导出，不悄悄更改旧稿数字。无OOB只有3seed，不把它说成5seed。
逐seed新汇总用stdlib statistics独立复核，最大差异1.11e-16。

## OSU资产：STOP_ASSET_INSUFFICIENT

原正式manifest有120个capture：72训练、24开发、24历史测试。本机这120个
`.dat` 全部缺失，元数据保留。五个完整模型与CNN seed0的权重hash均与旧
资产清单一致；历史 `has_ckpt=True` 不代表当前权重存在。
早期Day1→Day2及不同协议的预测不能代替Table II。仅存的正式cross-receiver
CSV也不能替代跨日期主实验，而且只有top1/最大概率，没有完整类别分数。
原服务器路径 `/data1/hcc/llm4RF` 在当前机器不存在。

旧loader的确定窗口是全capture均匀稀疏网格，提供`sample_offset`。
即使恢复原预测，前K是这个网格的时间前缀，不是相邻连续窗口；延迟必须
按实际offset/Fs计算。原稿的连续窗口也未验证为独立LoRa包。
Day4已用于旧模型选择，Day5已用于旧稿/研究决策，不改称独立盲测。

## Shen替代资产：已有完整分数，但真实分组不成立

按用户“可以先用Shen”指示，核验G1原十台源训练RX，两折名单保持不变。
所有十个HDF5只读元数据；每个仅有data/label/CFO/SNR字段，文件及字段属性
均为空，没有timestamp、packet time、capture ID、session或采样偏移。
作者公开材料说明包级采集和预处理，但尚未提供把本地行号映射至可靠
连续时间/session的证据。因此是“顺序不可核验”，不是证明包顺序已被打乱。

找到EISR E0归档的 `CV0/CV1 × Full/Strict CIS` 共20个源留出NPZ：
每文件logits/probabilities=(2000,10)，labels/rows=(2000,)，记录固定epoch20、
checkpoint SHA、源拟合scaler和RX来源。完整类别分数具备，**无需新增推理**。
当前只读取NPZ数组头部及文件hash，未评价其中预测值。
沿用G1历史每RX×TX200包的选定行；不声称源HDF5/同标签构成独立session。

|字段/来源|十源训练文件|结论|
|---|---|---|
|包级I/Q、标签、CFO/SNR|有|可用于包级识别，不能单独确定会话|
|真实时间戳/采集顺序记录|未找到|不按行号计算连续错误长度或实时停止|
|独立session/capture对应|未找到|不能报告capture级独立实验数量|
|已有完整预测向量|20个NPZ|未来合法任务可复用，本轮不据此绕过分组关口|
|K=256所需不同已预测包|每RX×TX历史200包|不能复采样凑256；本轮未扩样或推理|

按用户选择“保留原问题，严格按数据关口停止”，适用STOP_INVALID_GROUPING。
不改成K≤128的无序融合，不生成貌似真实时间的K曲线、相关性图或早停收益。

## C1/C2及论文价值判断

本轮无法计算原问题的错误持续、margin相关、可靠实时K曲线或强融合比较。
这些结果缺失已逐项记录，不用合成序列或无序包实验填充。
停止理由是分组/资产条件不足，不是STOP_NO_MECHANISM或STOP_BASELINE_SUFFICIENT。
不能据此判断相关性没有影响，也没有依据提出GO_NEW_METHOD_PROPOSAL。

文献进一步确认多包平均、预算饱和、自适应权重已有先行研究；MPAF目前仅
核验作者摘要/会议记录，全文未取得。详情见[文献矩阵](../../../docs/c_evidence/LITERATURE_MATRIX.md)。
标准融合是否解释原稿反转仍然未知；当前不适合启动新算法训练。

下一步只有补齐真实capture/session及时间序列映射、对应源域预测/冻结模型
数据后，才重新核验C0。仅增加设备数量、GPU或包数不能自动修复顺序证据。
如果未来允许恢复OSU，优先取得原服务器逐窗完整分数和偏移；其次才考虑
恢复原IQ+已有权重并报告冻结推理成本。当前没有恢复下载/推理授权或启动。

## 交付与GitHub

- [更新执行计划](../../../docs/c_evidence/EXECUTION_PLAN.md)
- [逐capture/日期/TX资产状态](OSU_per_capture_assets.csv)
- [逐模型/seed资产状态](OSU_per_seed_assets.csv)
- [历史Table II重汇总](Table_II_historical_reaggregation.csv)
- [Shen元数据证据](Shen_source_headers.json)与[完整分数资产头部](Shen_prediction_asset_headers.json)
- [输入/源码SHA](INPUT_MANIFEST.json)、[访问记录](access_log.json)、[决策](decision.json)
- [独立汇总复核](independent_verification.json)

P0已经补传到[GitHub P0报告](https://github.com/hanCChan/gai/blob/experiment/p0-sync-mechanism/reports/p0_sync/2026-10-13_A0_A1_final/REPORT.md)，
本地/远端文件树均为`f180ce18c8b0faf65ead0457b1d03c81a7063c45`，完整47个新增/修改blob
SHA一致并回读了决策。P0本地924dc21与远端29074af是不同提交身份、相同文件树。
本轮C计划、源码、结果和证据将发布至`experiment/c-evidence-audit`，上传和远端
回读核验完成后才交付链接。后续每轮发布要求已写入AGENTS.md。

复现C0（只读元数据/历史汇总，需同样本地资产路径，输出目录必须为空）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/keyan/venvs/shen-stage1/bin/python scripts/audit_c_evidence_assets.py \
  --output outputs/c_evidence/C0_reproduction_new
```

**本轮已经按数据关口停止，没有进入C1/C2或自动启动任何训练。**
