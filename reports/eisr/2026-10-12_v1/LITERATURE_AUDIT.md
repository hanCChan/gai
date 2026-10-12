# EISR-v1：E0 竞争方法与协议核对

日期：2026-10-12，北京时间。仅核对原文/作者代码/出版社信息，不训练竞争模型、不读取它们的目标数据，不把摘要或 README 当完整创新性证明。

|工作|已核对一手材料|对本轮机制和复现的影响|
|---|---|---|
|ASKNet|[作者仓库](https://github.com/BeechburgPieStar/ASKNet-RFF)、[LoRa 入口](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/main/Lora/run_CIS.py)、[ResCISNet](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/main/Lora/backbones/ResCISNet.py)|代码实际将 IQ 经过可学习 FFT 截断、酉变换、IFFT，再算 CIS。不能把自动选带或普通 CIS 改进当新贡献。LoRa 入口支持 n210_1/rtl_6 不同日，源内随机 train/val 划分，并按 val loss 调度 LR、提前停止、保存模型。它不是本轮五 RX/四加一 RX 协议，后续公平比较需单独注册，不能直接搬来称同协议。|
|FAN/IB-RFF|[作者代码](https://github.com/BeechburgPieStar/IB-RFF)、[main.py](https://github.com/BeechburgPieStar/IB-RFF/blob/main/main.py)|FAN 与 HSIC 信息瓶颈已有频率冗余抑制；公开入口为 WiSig ManyRx/ManySig，按接收机分组后源内 train_test_split，损失包含分类与 hidden-input HSIC。本轮没有复现它，不称已经超越。|
|TQCD/DRCI|[出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S1874490726001473)|出版社摘要/介绍已明确 temporal quotient、基于距离的置信度整合和轻量蒸馏；不能把谱商或一般置信度补偿当首次提出。全文未取得，尚不能排除其已涉及类别损失/信息恢复，精确距离整合公式也未完全核验。|
|MR-CIS/EAFE|[出版社公开记录](https://www.sciencedirect.com/science/article/pii/S2405959526000809)|MR-CIS、多尺度、CBAM 和角度间隔损失已组合使用；不是当前 EISR 的贡献空间。本轮只核对公开摘要及方法介绍，不称完整复现。|
|Ma|[作者原文](https://arxiv.org/html/2512.12070v1)|对比预训练、Siamese 和增强已有。EISR 只有两个既有分类器的预测，不把同 TX 约束或双专家组合当新算法。|
|DRIFT|[作者 v2 原文](https://arxiv.org/html/2510.09405v2)|TX/RX 解耦、GRL 和中心正则已有，不称普通特征分离为贡献。|

本轮最窄待证问题是：在固定 Full/Strict 专家下，Full 可挽救的分歧是否能只由其他源 RX 的预测输出识别，并在新留出 RX 上减少设备损害。互补性本身不证明风险可预测；线性预测成功也不自动证明信息被 Strict 因果擦除。

两个专家分别训练、对应 scaler 不同且可能源历史开发不同，因此逐包错误差异不等于只删除带外频谱的纯因果实验。Hard oracle 是二选一 Accuracy 上界；Soft oracle 限定标量概率凸组合，并不产生 Macro-F1 上界。

严格 E0 以本地四个 final.pt、原始训练记录和 source scaler 为准，远端分支是否齐全不能代替本地 provenance。先查实20轮模型，再生成新的 CPU 推理预测；旧最佳验证 checkpoint 的预测仅属于历史结果，不混入新筛选。

E2 即使通过也只授权后续讨论；需要适用强基线、完整论文差异核对和独立确认数据，才能评估贡献。未取得的全文和当前未做的基线复现保留为明确证据缺口。
