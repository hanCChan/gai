# 文献与贡献边界

2026-10-12核对一手来源，最新用户方案优先。

[Shen TMC原文](https://arxiv.org/pdf/2207.02999)已有多RX对抗训练与接收机漂移研究；当前CNN+GRL不能单独主张创新。原文RF设置与本地文件溯源分开，详见M0审计。

[Ma 2025原文](https://arxiv.org/abs/2512.12070)使用对比预训练、Siamese训练及增强；当前src/shen_next是代表性组件，没有完整预训练和原作者协议，不等于完整复现。

[TIFS 2026官方介绍](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security/2026/09/information)及[作者IB-RFF代码](https://github.com/BeechburgPieStar/IB-RFF)已结合FAN频率处理与HSIC信息瓶颈的两阶段训练，不依赖RX标签，在WiSig/LoRa测试。官方介绍位于2026年9月栏目；精确online日期需IEEE元数据确认，不把二手8月日期当独立核实。G3要正面对比该方法，不能将注意力改名为稳定性学习就宣称新颖。

[ICCCN2026 RIFLE](https://ieeexplore.ieee.org/abstract/document/11662855/)研究不完全TX-RX覆盖下的GRL；当前Shen完整覆盖可以做机制基线，但不能声称解决稀疏覆盖。

[OSU官方来源](https://research.engr.oregonstate.edu/hamdaoui/datasets)及本地SigMF确认915MHz、1MS/s与两RX相同发射机采集。25类与Shen31–40类没有分类器标签对应。作为不同采集团队复现，仍需披露旧项目使用史与同步窗口限制。

[WiSig官方说明](https://cores.ee.ucla.edu/wisig/overview/)报告174TX、41RX、一个月内四次capture（不应说四次跨月），常用前导256samples含原始和均衡版本。当前阶段优先Shen+OSU，WiSig可作后续G4；不能搬用8192 LoRa FFT与固定IB定义。不同体制应检验相同学习目标，允许不同前端和不同频带。

[RIEI的一手IEEE记录](https://ieeexplore.ieee.org/document/10742098/)与[作者预印本](https://arxiv.org/abs/2411.03636)已确认：期刊卷期是2025年3月，online是2024年11月，研究可分离条件、发射机/接收机特征解耦及FedRIEI。尚未完成完整方法/原作者协议复现。2024综述与其引用的早期OOB原文尚未精读，本轮不把转述当作已复现证据。

进一步阅读[IB-RFF作者main.py](https://github.com/BeechburgPieStar/IB-RFF/blob/main/main.py)、FAN.py与load_data.py发现：当前公开入口选择ManyRx/ManySig，seq_len固定256，用CE/NLL加hidden-input HSIC，源数据内部随机验证、学习率调度及两阶段加载。README提及LoRa，不能据此宣称本地已取得可直接运行的8192点LoRa适配器。G3复现须明确输入长度、LoRa前端、类别数和训练预算的改动，并保留原作者版比较；本轮未训练FAN或HSIC。

目前只有失效线索，无新通用方法。G1复现冲突→G2源域指标预测未见环境→G3超越强基线且减少负迁移→G4跨数据集/条件成立。每一步必须有结果才改变结论，不预先写论文标题或创新声明。
