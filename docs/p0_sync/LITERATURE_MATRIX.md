# P0 文献差异与强基线矩阵

核对时间：2026-10-13（Asia/Shanghai）。只使用作者仓库、原论文和出版社页面。
这是文献边界核验；未执行强方法新训练，也没有“已排除全部先行工作”的结论。

|材料|本轮取得的证据|已覆盖的机制|与 P0 的差异/限制|本轮实验状态|
|---|---|---|---|---|
|[Shen 等，Receiver-Agnostic](https://arxiv.org/html/2207.02999)|作者原文全文，III-B、V-A；本地作者 loader/CIS|同步与前导提取、CFO 补偿、归一化，CIS、接收机对抗训练及协同推理|论文流程不证明当前 HDF5 上游操作；缺包外记录和同步真值|仅代码溯源；未启动对抗训练或目标推理|
|[Shen 等，Spectrogram/CFO](https://arxiv.org/html/2101.01668)|作者原文全文，III-B、IV-A|瞬时频率均值粗 CFO、相邻重复符号相角细 CFO，另将 CFO 用作辅助身份信息|旧 frequency.py 只有 modulo 频率代理，缺粗估计及起点检测；不能把补偿越多当作身份越好|数值核验模糊性；未补偿真实 IQ|
|[Tapparel/Burg，Two-Pass Synchronization](https://arxiv.org/html/2502.08485v1)|作者原文全文，III-A–C|分数 CFO/STO、利用 up/down chirp 区分整数 CFO/STO、SFO 及两次估计|需要相应完整帧结构；本轮仅观测重复 up chirp，不能宣称完整复现或超越|未适配、未做性能比较|
|[Ahmed 等，2024 LoRa 综述](https://orbi.umons.ac.be/bitstream/20.500.12907/49474/1/sensors-24-04411.pdf)|作者机构库全文通过浏览工具可读，§4.2.1–2；本地下载超时已记录|已总结 Schmidl–Cox、预期/实测瞬时频率相关及 CFO 预处理|综述是检索入口；新颖性需回到具体原论文；本轮锯齿相关诊断不作为新算法|资料核对|
|[Shen 等，Scalable/Channel-Robust](https://arxiv.org/html/2107.02867)|作者原文全文 III-C、IV；[作者 60 TX 说明](https://github.com/gxhen/LoRa_RFFI/blob/main/Openset_RFFI_TIFS/readme.md)及 loader|同步/裁前导/CFO/RMS、CIS、深度度量学习、注册及 rogue detection|60 TX 的任务能力已存在；其发布前导也不能保证包含同步失败所需包外信息|仅结构与成本审查；零 HDF5 下载、零训练|
|[Toe 等，PIR](https://www.sciltp.com/journals/pc/articles/2607004785)|出版社正式摘要和 DOI，2026-08-10；PDF 链接失败、本地页面下载超时|摘要明确前导对齐、CIS 中减同处理理想参考|全文/代码未取得，不能宣称完整机制审查或复现；“对齐+CIS+理想残差”已不能当新贡献|未执行；证据层级明确为摘要|
|本地 G1 Full/Strict CIS 前端|`src/shen_p2a/representations.py` 及作者 `external/receiverAgnosticRFFI/dataset_preparation.py`|RMS、128 点 boxcar STFT、hop64、无边界填充、复谱加 1e-12、相邻帧谱商 log10；历史 cropped Full CIS 为 52×126|这是历史固定前端，不是同步检测器。60 TX 作者代码默认 win256/overlap128、无相同 epsilon；两数据集不能直接互换前端|只读代码，不修改或新增模型推理|

两阶段 CFO、标准同步和 CIS 均有先行工作。PIR 的具体全文章节、参数及代码仍是资料缺口。
本轮在可辨认性关口停止，因此没有取得“标准方法解决/不能解决当前自然失效”的实验结果。
没有将综述列出的算法、锯齿模板诊断或强基线组合登记为第二篇的新贡献。

公开材料的本地检索路径和 SHA256 见
`outputs/p0_sync/20261013_a0_literature/{manifest,extra_manifest,shen60_structure_manifest}.json`。
未取得的材料按失败原因保留，不用其他论文、摘要或相似标题补齐。
