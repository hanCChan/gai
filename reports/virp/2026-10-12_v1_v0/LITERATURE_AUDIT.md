# VIRP-v1：V0 文献与机制核对

核对日期：2026-10-12（北京时间）。本文件是原理及已有工作边界记录，不证明创新空白，也不作为放宽预登记门槛的理由。

|工作|核对的一手材料|可确认的机制与本轮边界|
|---|---|---|
|Shen：Receiver-Agnostic and Collaborative RFFI|[原论文](https://arxiv.org/pdf/2207.02999)，本地作者 CIS 实现|LoRa 使用 CIS、128 点窗/64 点重叠，并有接收机对抗训练。幅频扰动必须检验能否穿过谱商。|
|Ma：Channel-Robust and Receiver-Independent RFFI|[2025 原文全文](https://arxiv.org/html/2512.12070v1)|§IV-B 已使用 TDL、多径、AWGN；SNR 为 10–40 dB，并结合 Siamese/对比训练。B1 使用这个噪声范围，仅称普通增强基线，不称完整复现 Ma。|
|DRIFT|[作者 v2 全文，2026-05-26](https://arxiv.org/html/2510.09405v2)|当前版本将特征分成 TX/RX 两部分，使用 GRL、接收机中心正则和分离约束；WiFi 输入为 2×256。不能把普通特征解耦称创新，也不将旧摘要的风格表述误当当前版本的信号层增强。|
|FAN + IB|[IEEE 论文记录](https://ieeexplore.ieee.org/abstract/document/11653216)、[作者代码](https://github.com/BeechburgPieStar/IB-RFF)|IEEE 记录为 TIFS 21:8008–8021，online 2026-08-12，DOI 10.1109/TIFS.2026.3723081。FAN 与 IB 已处理频率冗余/过度对齐；作者公开代码使用 HSIC 路径。当前未完整获得 IEEE 正文，也未完成 LoRa 适配复现，不声称已经超越。|
|Adaptive Semantic Augmentation|[作者仓库及说明](https://github.com/Edith-xx/Cross-Domain-RFFI)、[作者模块](https://raw.githubusercontent.com/Edith-xx/Cross-Domain-RFFI/main/MixStyle_model.py)|作者材料说明 2D-DWT、多尺度提特征及中间特征 mean/std 混合、AdaIN。本文全文尚未取得，代码核对不冒充全文差异论证。VIRP v1 是 IQ 层受限谱形修改，是否更有用仍需实验证明。|
|Group DRO|[Sagawa 等原文](https://arxiv.org/pdf/1911.08731)|预定义组的最坏风险、正则化及避免过拟合已被研究。TX 条件风险和平滑最大损失本身无新数学贡献；B4 必须控制组覆盖、预算与正则化。|
|物理扰动已有工作|[Rajendran/Sun 官方论文记录](https://ieeexplore.ieee.org/document/9732439/)|RF impairment model 与 domain generalization 已有研究。仅核实论文记录，未取得全文；不能声称物理增强首次提出或宣称现有工作没有身份保持机制。|

本地源码可直接确认 CIS 是 `log10(abs(STFT[t+1]/STFT[t])²)`。若局部 STFT 近似满足静态乘性响应，响应约去；全包滤波不与有限窗 STFT 精确可交换，必须区分有效表征变化与窗边界/低分母残余。这一判断是数学推导与代码核对，是否形成有用扰动由冻结 V0 判定。

纯零相位频谱乘法没有证明因果硬件可实现性，因此只使用 frequency-response-inspired spectral augmentation 名称。采集谱差异包含信道、接收链路及采集因素；未验证的本地 RF 元数据、严格前导与采集会话不由论文设置补齐。

如果 V0 未通过，则不启动分类训练，也没有“超过 ASA/DRIFT/FAN/DRO”的证据。全文比较中未取得的材料保留为明确缺口；本轮停止不会转成同名方法的模块拼接。
