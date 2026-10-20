# LITERATURE_D — D 阶段证据对应的文献核对（2026-10-20）

范围：只读核对。对象是 D 阶段证据（WiSig ManyRx 未均衡 L-STF、14 source RX、10 TX；Shen LoRa Strict CIS）以及三个候选方向：
(a) 物理可实现的 LTI FIR 谱随机化，幅度/协方差由 source RX 与 date 变化校准，并加身份保持约束；
(b) 源域反事实谱敏感度，作为无目标数据的 model/seed selection 信号；
(c) 原始视图与扰动视图之间的 KL consistency。

阅读方式：逐篇读本地全文 `/workspace/rl1/papers/txt/`（大多是 arXiv 版本，期刊终版可能有差异）。引文保留英文原文，只把连字（ﬁ→fi）和断行规范化。“§/Eq.”指所读版本中的章节/公式。凡属推断，都标成 **[推断]**；凭记忆而本地未核实的，标成 **[未核实]**。

---

## 0. 结论先行（直白版）

- **(c) 已有人做，不能算贡献。** 跨接收机 RFFI 里，KL consistency 已经出现在 CRODA-ST（VAT，对抗扰动下的 KL）中；Ma 等人在两台真实 RX 的同 TX 配对之间、加信道增强后的视图之间做了 Siamese NT-Xent；MASF 在不同 domain 之间做 symmetrized-KL 的类关系对齐。只有当 (c) 用的扰动分布 (a) 本身是新的，(c) 才有意义。
- **(a) 只有一部分是新的。** “随机 LTI FIR 增强”本身就是信道增强：TDL 多径就是随机 FIR。Shen TMC、Ma 都做了，按引用文献的描述 Soltani 2020 也做了。**用实测 source RX/date 响应校准的接收机响应增强**，在读过的论文里没人做；WiSig 原文 2022 年把它列为 open problem。**输入层的身份保持约束**也没人做。但我们自己的 D 证据削弱了 (a) 的“校准”前提：random 与 real 的破坏力一样（ratio 0.96）；按单位变化量算，real RX 方向反而最无害；主 RX 轴又承载了 35% 的 TX 能量。在 Shen Strict-CIS 上，(a) 基本是空操作，因为 CIS 用相邻 STFT 帧相除，静态 LTI 响应近似约掉 **[推断，与 RX share≈2% 一致]**。
- **(b) 在 RFFI 文献里没出现过**：读过的论文都用源域 validation loss/acc 选模型，或者直接报目标测试曲线。但它在 ML 里属于一个已知思路（stress test 预测 OOD）。D'Amour 等人正好给出反证：同一 pipeline 内，seed 之间的 stress-test 表现与 iid 表现、不同 stress test 之间的相关都弱。我们的结果是 random 与 real 方向同样能预测，所以这个信号其实是**通用谱敏感度/平坦度**，“反事实接收机”的叙事得不到自己数据的支持；它还被 3 个 seed 簇混杂。
- **真正新的东西在诊断层面**：
  1. 在冻结模型上做 matched-norm 干预，比较 real RX 响应与 random/tilt/TX 方向；
  2. 在输入空间量化“RX 变化子空间与 TX 身份子空间重合”。这一点直接与 SD-RXA、RIEI、Ma、Division 方法的显式假设冲突（见 §4）；
  3. 同一 RX 跨 date 漂移与换 RX 的单位破坏力相当；
  4. Shen 上未见 RX 的失败是近邻 TX 对的单向塌缩，且 SWAD 加重、MLDG/GroupDRO 减轻。

  方法论文若只靠 (a)+(c)，审稿人会把它看成“换了参数分布的信道增强 + 一致性正则”。

---

## 1. 读取清单与版本

| ID | 论文 | 读取版本 | 正式出版信息（来源） |
|---|---|---|---|
| A1 | Shen et al., Receiver-Agnostic & Collaborative RFFI | arXiv 2207.02999 **v1**（2022-07） | IEEE TMC 23(7):7618–7634, 2024（见 Ma 参考文献[38]、C2b [32]）。终版可能有新增实验，本文只依据 v1 |
| C2f | Zhao et al., GAN-RXA | arXiv 2303.14312 v2（2024-02） | IEEE TCCN 10(2):403–416, 2024（Ma 参考文献[33]） |
| C3a | Ma et al., Channel-Robust & Receiver-Independent RFFI | arXiv 2512.12070 v1 | 正文只写 “Manuscript received xxx”。**期刊（用户称 TIFS）本地无法核实** |
| C3b | Xie, Peng, Zhang, Spectrum Averaging + High-Order Difference | author copy | 文件名标 TIFS 2026，正文页眉是通用模板，未核实卷期 |
| A2 | Zhang et al., RIEI (cross-receiver DG) | arXiv 2411.03636 v1 | IEEE IoTJ 12(5):5207–5218, 2025（Ma [40]） |
| C2b | Yang et al., Division-based Receiver-agnostic RFF (WiFi) | arXiv 2511.15458 v3（2026-02，投稿中） | — |
| C2a | He et al., Cross-Receiver RFFI under Varying Channels | arXiv 2603.08402 v1 | — |
| C2e | Yang et al., Mitigating Receiver Impact via DA | arXiv 2404.08566 v1 | IEEE IoTJ 11(13):24024–24034, 2024（RIEI [19]） |
| C2c | Yang et al., Source-data-free cross-receiver (MS-SHOT) | arXiv 2512.16648 v1 | — |
| A6 | Hanna et al., WiSig | arXiv 2112.15363 | IEEE Access 10:22808–22818, 2022 |
| B21 | Li et al., CIDG（AAAI 2018，**kernel 版**） | arXiv 1807.08479 | 注意不是 ECCV 2018 的对抗版 CIDDG |
| B18 | Dou et al., MASF | arXiv 1910.13580 | NeurIPS 2019 |
| B08 | Zhou et al., MixStyle | arXiv 2104.02008 | ICLR 2021 |
| C1 | Tran et al., ConstStyle | ICCV 2025 open access | ICCV 2025 |
| B03 | Cha et al., SWAD | arXiv 2102.08604 | NeurIPS 2021 |
| B17 | D'Amour et al., Underspecification | arXiv 2011.03395 | JMLR 2022 |
| 附带核对 | C2d CRODA-ST（arXiv 2607.02567 v4）、A5 DRIFT（arXiv 2510.09405）、B02 DomainBed、B15b EoA | 本地全文，只检索相关段落 | — |

**本地没有、也没有取到的两篇：**
- Soltani et al., "More Is Better: Data Augmentation for Channel-Resilient RF Fingerprinting", IEEE Commun. Mag. 58(10):66–72, 2020。
- 2026 IEEE TCCN "Multi-Scale Complex-Valued Learning and Balanced Domain Alignment"（WiSig 子集）。

本会话尝试联网检索（Semantic Scholar API、IEEE Xplore，经 star-proxy），两次都被运行环境的权限分类器拒绝，**没有拿到任何摘要**。下文不对这两篇作内容性陈述。

关于 Soltani 2020，只能引用本地论文对它的二手描述：
- WiSig §VI-B："we can use channel augmentation techniques, whether handcrafted or data-driven similar to those proposed in [16]"；
- Shen A1 §III-C 把它（[24]）列在 "The training signals are replicated and passed through a channel simulator to emulate multipath and Doppler effects" 一类之下；
- C2b §I："data augmentation methods [16], [22]–[24] can enhance classification accuracy, although they typically incur increased training overhead"。

所以本文只能确认它属于**信道增强**。它有没有涉及接收机效应、test-time augmentation，**均未核实**。TCCN 2026 一文只知道标题，标题暗示是“对齐类”方法，内容一律不推断，投稿前必须读全文。

---

## 2. 逐篇提取

### A1 Shen et al.（TMC；读 arXiv v1）

- **扰动/对齐**：
  - 信道增强（§III-C、§IV-C）："the exponential power delay profile (PDP) is employed to characterize the multipath effect and the Jakes model is used to describe the Doppler spectrum. The root mean square delay spread, maximum Doppler frequency, Rician K-factor and SNR are uniformly distributed in [5 ns, 300 ns], [0 Hz, 10 Hz], [0, 10], [0 dB, 50 dB], respectively."
  - 输入用 CIS（§IV-D）。
  - 接收机方面不做增强，而是用 GRL 做 RX-adversarial（§III-E，Eq. 14：θ←θ−μ(∂L_tx/∂θ − λ∂L_rx/∂θ)）。
- **物理性**：信道是物理模型（TDL/PDP + Jakes + AWGN）；接收机没有物理模型。
- **目标 RX 数据**：主方法不用。可选 fine-tuning 用**有标签**目标数据（§III-G："collect very few labelled signals y_j using the under-performing receiver j"；§V-C-4 为每对 DUT-SDR 20 包）。
- **数据/RX 数**：10 个 LoRa DUT（5 LoPy4 + 5 mbed SX1261），20 台 SDR（9 RTL、2 Pluto、2 B200、2 B200mini、2 B210、3 N210）。训练 RX 数 I=1–5，分 homogeneous（RTL-1..5）与 heterogeneous（RTL-1、PLUTO-1、B200-1、B200 mini-1、B210-1）。办公楼测试用 3 台 N210、6 个位置。
- **敏感度/选择**：没有敏感度分析。§V-A-4："10% of the training data was split out for validation ... The validation loss was monitored"，即源域选择。
- **身份/纠缠**：目标是彻底去除 RX 信息（§III-E-4："we expect the feature vector to not contain any receiver-related information"），没有讨论 RX-TX 纠缠。
- **与 D 相关的事实**：
  - 同 RX 漂移（§V-B-1）："the performance of RTL-6 degrades seriously by more than 40% on Day 2, 3, and 4"。这与我们“date 漂移的单位破坏力 ≈ 换 RX”方向一致，但他们没有做幅度归一化比较。
  - 训练 RX 数在 I≥3 后增益很小（Fig. 10）。

### C2f GAN-RXA（TCCN 2024）

- **扰动/对齐**：没有任何数据增强。名字里虽有 GAN，**但不生成样本**。它是“判别器 + 让 RX 后验逼近先验”的对抗对齐（§III-D，Eq. 8 的 L_disc = mean{exp[(ŷ−z)²]−1}，Eq. 9 的 L = L_disc + γL_tx，Algorithm 1 交替训练）。另一个变体 SD-RXA（Eq. 5–7）最大化 RX/TX 特征协方差距离。
- **物理性**：无。预处理用 L-LTF MMSE 均衡（§III-A、§IV）："we use equalized data for all experiments"。
- **目标 RX 数据**：不用。两阶段：lab 多 RX 训练 FE，field 用单 RX 训分类器，再部署到新 RX。§III-B 要求 "T_lab ∩ T_field = ∅, R_lab ∩ R_field = ∅"。
- **数据/RX 数**：WiSig 子集，50 个 Atheros TX、28 个 USRP RX（3 B210、9 X310、16 N210）、4 天；**自取 400 样本**（"the entire L-LTF and L-STF ... 400 × 2 IQ samples"），不是 256 点 compact subset。|R_lab|=25，|T_lab|=40，|R_field|=1，|T_field|=10，测试 |R̃_field|=2。Table V closed-set：Naive 43.6%，Basic-RXA 63.1%，SD-RXA 66.6%，GAN-RXA 68.1%。
- **敏感度/选择**："20% signals randomly sampled from each training dataset are reserved for validation"，γ 按 "cross-validation" 选（推测在源域上做）。
- **身份/纠缠**：这是读过的论文里**唯一明说存在 trade-off 的**："if L_disc updates f_tx too aggressively, f_tx loses the capability to distinguish between transmitters ... a small γ will cause the FE f_tx to fail in learning the TX features, and a large γ will cause f_tx to learn RX-dependent features"（§III-D）。但 SD-RXA 的假设恰好与我们的发现 3 相反："we assume the feature distributions of transmitters and receivers are uncorrelated"（§III-C）。

### C3a Ma et al.（用户称 TIFS；读 arXiv v1）

- **扰动/对齐**：
  - 信道 + AWGN 增强（§IV-B）："the tapped delay line (TDL) channel has been chosen ... the RMS delay spread ranges from [5,300]ns, Doppler frequency spans [0,5]Hz, and SNR is [10,40]dB"，在线随机，每个样本独立增强两次。
  - Siamese 对比训练（§III-B）："We select packets from the same DUT but different receivers as a pair to send into the Siamese network"，损失为 NT-Xent（τ=0.05）+ CE（Eq. 9–11）。
  - 无监督对比预训练（§V），用公开数据集 [29]。
- **物理性**：信道是物理模型。接收机建模为 log-spectrogram 加性项（Eq. 6：log|X(t,f)| ≈ log|S(G)| + log|S(h)| + log|S(F_k)|），但**不对 G 做增强**，只靠两台真实训练 RX 配对。
- **目标 RX 数据**：不用。预训练数据（[29] 中的 Pluto、B200 mini、B200、B210 子集）与测试 RX 是否物理上同一台，论文没说 **[不确定]**。
- **数据/RX 数**：
  - Public 1：Shen TIFS22，10 LoPy4，单 N210，只测信道；
  - Public 2：Elmaghbub，B210；
  - Public 3：Shen TMC，用 B200-1、B210-1 训练，测试 20 台 SDR，另加 N210-1/2/3 在 6 个位置；
  - 自采：10 DUT，用 B210-a、N210-c 训练，测试 RTL-a/b、B210-a/b、N210-a/b，有动态信道。
- **敏感度/选择**：没有敏感度分析；监控源域 validation loss（§VI-C）；每个设置跑 4 轮。
- **身份/纠缠**：没有讨论，而且明确声称不丢信息："spectrogram ... decouples transmitter impairments, channel effects, and receiver impairments without resulting in information loss"（贡献 1）；§IV-D-2 写道 "(log|S(G1)|+log|S(h1)|) and (log|S(G2)|+log|S(h2)|), representing variations introduced by receivers and channels, will be mitigated. This ensures that the model learns the RFFs solely"。这隐含 log|S(F_k)| 与 log|S(G)| 不共线，正是我们的发现 3 所否定的。

### C3b Xie, Peng, Zhang（TIFS 2026 author copy）

- **扰动/对齐**：没有学习型对齐。信号处理两步：
  1. 多天线谱平均（Eq. 8–9）；
  2. log CSI 幅度的 d 阶差分（Eq. 28–30）。

  §V-C 明说："the high-order difference operation is equivalent to a high-pass filter process"；Appendix C 给出 |G_d(θ)| = 2^d|sin(θ/2)|^d。训练时为测 SNR 鲁棒性加了 AWGN 增强。
- **物理性**：信道建模为多径（Eq. 2、31），并用 Weierstrass 多项式逼近论证 "∆^d L_H(ω_k) ≈ 0"（Eq. 36）。接收机**不在研究范围**：只有一台 4 天线 X310。
- **目标 RX 数据**：不涉及（单 RX）。
- **数据**：12 部 COTS Wi-Fi 设备、1 台 4 天线 USRP X310、5 个位置、3 种场景。
- **敏感度/选择**：没有。训练/验证按时间顺序 8:1 切分。
- **身份/纠缠**：假设 RFF 位于子载波轴的高频分量："the received signal spectrum contains high-frequency components (i.e., low correlation across subcarriers) dominated by RFF features"（§V-C）。**[推断]** 任何在子载波轴上平滑（低阶多项式）的 TX 成分都会和信道/RX 倾斜一起被删掉。我们的主 RX 轴是低带/高带功率不对称（平滑），且承载 35% 的 TX 能量，所以这一类 front-end 正好可以拿来检验“身份保持”主张。另外，L-STF 只有 12 个音，d 阶差分后几乎没剩多少自由度。

### A2 RIEI（IoTJ 2025）

- **扰动/对齐**：特征解耦。总损失为 L = L_CE + λ1 L_MI − λ2 L_IE（Eq. 19）。其中 L_MI 是 emitter/receiver 特征的余弦（Eq. 18），L_IE 是交叉喂给对方分类器时的输出熵（Eq. 15–17），采用交替训练（Eq. 20–21）。没有数据增强。只在测试端加窄带跳频干扰/宽带噪声做鲁棒性测试（§V-B-4）。
- **物理性**：无。
- **目标 RX 数据**：不用。Remark 1："since we cannot access D_t, the best we can do is to make separable condition valid over D_s."
- **数据/RX 数**：WiSig 子集 "six transmitters (Tx) and six receivers (Rx)"，2 RX 训练、1 RX 测试；FedRIEI 为 4→1。HackRF：4 TX、3 RX。
- **敏感度/选择**："we take into account the mean and standard deviation of the classification accuracy for the last 5 epochs"。没有给出选择规则，看上去是报告目标测试曲线。
- **身份/纠缠**：理论假设与我们的发现相冲突。Assumption 1（引自 Robey et al. MBDG）：P(X^ν) = G#(P(X)×P(ν))；Definition 1 要求 ν→X→Z_X 且 Z_X ⊥ Z_ν。**[推断]** 当输入层 RX 变化方向与 TX 判别方向重合时，“Z_X 不含 RX 信息”只能靠丢弃该方向上的 TX 信息来实现。

### C2b Division-based receiver-agnostic RFF（WiFi）

- **扰动/对齐**：确定性除法不变量。
  - RD：未知设备的 L-STF/L-LTF 除以参考设备的同一字段（Eq. 10–13，平坦衰落）。
  - HL：HT-LTF 除以 L-LTF（Eq. 17–21，频选衰落）。
  - DV 基线：L-STF 除以 L-LTF（引 [15]、[16]）："this method can extract only 12-dimensional RFF features, as the short preamble ... occupies just 12 subcarriers"。
- **物理性**：频域乘性 LTI 模型 Y = H·F_R·F_T·X（Eq. 16）。
- **目标 RX 数据**：HL 不用。RD 要在新 RX 上采一次参考设备信号（"a new receiver, Rx2, first gathers the model signal from the reference device"），这是无 DUT 标签的校准。
- **数据/RX 数**：14 部 WiFi 设备、5 台 USRP（N210、B210、3×B205mini），有线、静态 LOS/NLOS、移动、走廊等场景。
- **敏感度/选择**："The model with the lowest loss value on the validation set is saved"（源域）。另提出参考设备选择指标 η_LF（低频能量比，Eq. 22），与跨 RX 准确率 "p-value ... 0.0034"。这是唯一一个源侧、无目标标签的 proxy，但它选的是参考设备，不是模型。
- **身份/纠缠**：把关键假设写明了（Eq. 19–20），"the RFFs across HT-LTF and L-LTF parts are different of the device to be classified, while the RFFs across HT-LTF and L-LTF parts are similar of the receivers"。也就是说，两段字段共有的 TX 成分（例如 TX LTI 滤波）会和 RX 一起被除掉。论文自己报告了后果："Dev12 and Dev14 are frequently misclassified as each other. This is because the RFF features of these two devices in both the L-LTF and HT-LTF parts are similar, leading the HL method to extract noise-like RFF features"（§V-C-1）。这是读过的论文里“去除不变量导致身份丢失”最直接的实例。
- **另外**：Set3（4 台训练 RX）时 GAN-RXA 超过 HL（Fig. 23）。

### C2a Cross-receiver under varying channels（仿真）

- **扰动/对齐**：信道方面，LTF 相邻有效子载波相除，得到 DSQ（Eq. 17）。接收机方面训练一个校准网络 TCNN，学习 G0(G1^{-1}(·))（§III-D）。
- **物理性**：全仿真。接收机模型只有 IQ imbalance + LNA 三阶非线性（Eq. 9–10，Table I），**没有任何线性频率响应**。
- **目标 RX 数据**：需要。"we employ both the source receiver and target receiver to capture the transmitted signals from a transmitter for building the calibration dataset"：一台高端发射机线缆直连，靠相位偏移配对。不需要 DUT 标签，但需要接触目标 RX。
- **数据**："twelve simulated WiFi transmitters and three receivers"，4 种 TDL 信道。
- **敏感度/选择**：source 数据集按 5:1:2 切分；无敏感度分析。
- **身份/纠缠**：未讨论。DSQ 相当于子载波轴一阶比值，会同时去掉平滑 TX 成分 **[推断]**。

### C2e Yang et al. DA（IoTJ 2024）

- **扰动/对齐**：UDA。Donsker-Varadhan 形式的 KL 特征对齐（Eq. 5–6，min-max）+ curriculum pseudo-labeling + class weighting（Eq. 7–11）。无增强。
- **目标 RX 数据**：**无标签**目标数据。"adapts the trained model to unlabeled signals from a new receiver"。
- **数据/RX 数**："six transmitters (Tx) and 12 receivers (Rx) ... over four days"（ManySig），已均衡；单源→单目标共 4 个任务，外加 d01→d23 跨天。Source only 在 14-7→3-19 上只有 30.25%。
- **敏感度/选择**：超参数敏感度在目标任务准确率上画（"14-7 → 3-19"）。无源域选择规则。
- **身份/纠缠**：未讨论。

### C2c MS-SHOT（source-data-free）

- **扰动/对齐**：SFDA。动量中心软伪标签 + 核范数多样性 + 类先验 ℓ1（Eq. 13–14）。这里的 "consistency regularization" 指类分布一致，**不是增强视图一致**。源训练时加 AWGN 只为测 SNR。
- **目标 RX 数据**：无标签目标数据，且不访问源数据。
- **数据/RX 数**：WiSig ManySig 6 TX × 12 RX（"1,000 equalized signals"）；扩展实验 80 TX、26 RX（20 源 → 6 目标）；HackRF 4 TX、3 RX。
- **敏感度/选择**：报告 "accuracy over the last 5 epochs"，即目标曲线。
- **身份/纠缠**：未讨论。

### A6 WiSig（IEEE Access 2022）

- **数据**：174 TX、41 RX、4 次 capture。compact subsets（Table II）：ManySig (6 TX, 12 RX, 1000 sig)、ManyTx (150, 18, 50)、**ManyRx (10, 32, 200)**、SingleDay (28, 10, 800)；有未均衡与均衡两版。均衡流程（§IV-B）：L-STF 自相关定位起点 → 估计并校正 CFO → "the channel is then estimated using the L-LTF and the signal is equalized using MMSE" → 重新加回 CFO → 取前 256 点。
- **接收机泛化**（§VI-A，ManyRx、均衡、单天）："using only one Rx for training results in a drop in accuracy from 99% to less than 33%"；"the improvement starts to saturate at about 20 Rx"。
- **明确列为 open problem**："One solution is to develop Rx fingerprint augmentation techniques. Once trained, such an approach would use signals from one Rx to synthesize signals from other virtual receivers similar to the approach used in [15]"（[15] 为 Karunaratne et al., Generative Outlier Augmentation, GLOBECOM 2021）。
- **跨天**（§VI-B）："there is still a small drop in accuracy for different days even with equalization, which might be because some aspects of the Tx fingerprint or Rx fingerprint may vary slightly over time"。
- **[推断，需按帧定时核验]** 256 点 compact 窗在 25 Msps 下覆盖 L-STF（200 点）+ GI2（40 点），只带 L-LTF 第一个长符号（80 点）的一小段，所以不能对 L-LTF 做整符号 FFT。DV/HL 一类除法不变量不能直接用于 256 点 release，要从 raw WiSig 重新切。GAN-RXA 自取 400 点正是这个原因。反过来说，如果均衡器按子载波除以 L-LTF 估计，**均衡版 STF 音调近似就是 DV 特征 F_T,S/F_T,L**（实现细节论文未写，**不确定**）。

### B21 CIDG（Li et al., AAAI 2018，kernel 版）

- 目标是使 P(h(X)|Y) 在 domain 间不变，而不是对齐边缘分布："We propose to learn a feature representation which has domain-invariant class conditional distributions P(h(X)|Y)"。两个正则项："The first term measures the variance of each class-conditional distribution across all source domains ... The second term is the variance of class prior-normalized marginal distribution"。
- 不用目标数据，无增强，无敏感度分析。
- 与 D 的关系 **[推断]**：RX 变化与 TX 轴重合时，边缘分布对齐（DANN/GAN-RXA/GRL）可以靠“类间错位”达成；类条件对齐不允许这样。这是对 Shen 上“近邻 TX 对单向塌缩”最对口的 DG 基线之一。

### B18 MASF（NeurIPS 2019）

- 每步随机划分 meta-train/meta-test domain（Alg. 1："Randomly split source domains D into disjoint meta-train D_tr and meta-test D_te"）。global class alignment："align the soft class confusion matrix between two domains ... by minimising their symmetrized Kullback–Leibler (KL) divergence"（Eq. 4）。另有 local sample clustering（contrastive/triplet）。
- 与 (c) 的关系：它是跨真实 domain 的 KL 类关系一致，不是“原视图对扰动视图”。但审稿人会把“保持近邻 TX 关系”的主张拿来与 MASF 比。

### B08 MixStyle（ICLR 2021）

- 特征统计量随机插值：γ_mix = λσ(x)+(1−λ)σ(x̃)，β_mix 同理，MixStyle(x) = γ_mix·(x−μ)/σ + β_mix（Eq. 5–7），"λ∼Beta(α,α) ... we set α to 0.1"，激活概率 0.5，"At test time, no MixStyle is applied."
- 数据驱动的随机扰动，非物理；不用目标数据；无身份约束。1D IQ CNN 浅层的通道统计部分编码谱形 **[推断]**，所以它是 (a) 最直接的“非物理随机”对照。

### C1 ConstStyle（ICCV 2025）

- 每个 seen domain 的 style 统计（特征 mean/std）建模为 N(ε_S, Σ_S)，unified domain 取这些分布的 barycenter（迭代求 Σ_B）。"During training, all samples are mapped onto this unified domain ... During testing, unseen domain samples are projected similarly before predictions."
- 推理时只对单个测试样本做 partial alignment，以免信息损失："balances the alignment of the testing sample style with the trained style ... while preserving the essential information of the testing data"。
- 与 (a) 的关系：“用 source domain 协方差校准扰动/对齐”在特征层已经有了（Gaussian + covariance），而且 ConstStyle 明确考虑了“对齐会破坏信息”。我们在输入层、物理参数化、并按 TX 轴保护，是区别所在，但需要实验证明有区别。

### B03 SWAD（NeurIPS 2021）

- 鲁棒风险 Ê^γ_D(θ) = max_{‖Δ‖≤γ} Ê_D(θ+Δ)，即参数空间平坦度。dense weight averaging，"we search the start and end iterations for averaging by considering the validation loss"（源域 validation）。
- 与 (b) 的关系：参数平坦度是 OOD 的一种源域 proxy，我们的输入谱敏感度是另一种。审稿人会要求比较两者，尤其因为我们在 Shen 上观察到 SWAD 加重 Pluto 塌缩。

### B17 Underspecification（JMLR 2022）

- 与 (b) 直接相关的反证："within this ensemble, there is weak sample correlation between performance on the iid test set and performance on each benchmark stress test, and performance between tasks (all 95% CI's for Pearson correlation using n = 50 and n = 30 contain zero, see Figure 5)"（ImageNet-C，seed ensemble）；NLP 部分："Predictor performance across stress tests are typically weakly correlated"。另有 "even pipelines that are identical up to their random seed can produce predictors that encode distinct shortcuts"。
- 含义：我们 18 个模型上 ρ≈−0.96 的相关如果主要来自 3 个 seed 簇之间（即 recipe 差异），就不能证明能在 pipeline 内部做 seed 选择。必须报告簇内相关和选择 regret。

### 附带核对（与 (b)(c) 直接相关的局部段落）

- **C2d CRODA-ST**（arXiv 2607.02567 v4）：跨接收机 open-set。用 **VAT**（KL，对抗输入扰动）+ GRL 接收机判别器，并使用**有标签**目标已知设备（"limited labeled target-domain data"）。作者自己承认扰动不是物理的："the optimization-derived r_adv represents a worst-case local direction rather than a calibrated physical-noise model"。"no target-unknown sample is used for ... model selection"。数据为 WiSig ManyTx + 合成 LoRa。**这就是“(c) 在 RFFI 跨接收机已有”的直接证据**；该引文也可以拿来动机化 (a)。
- **A5 DRIFT**：WiSig ManySig，5 RX 训练 → 未见 RX；GRL + RX center + MSE 分离；无增强。
- **B02 DomainBed**："domain generalization algorithms without a model selection strategy should be regarded as incomplete"，给出 training-domain validation、leave-one-domain-out CV、oracle 三种选择准则。
- **B15b EoA**："stochasticity in optimization (e.g. seed) plays a big role"，且 OOD 不稳定 "causes the rank correlation between in-domain validation accuracy and out-domain test accuracy to be weak"。

---

## 3. 对照表

| 论文 | 扰动/增强或对齐 | 物理信道模型 | 接收机响应建模/增强 | 目标 RX 数据 | 数据集 / RX 数 | 敏感度测量或无目标选择 | 身份保持 / RX–TX 纠缠 |
|---|---|---|---|---|---|---|---|
| Shen A1 | TDL(exp-PDP)+Jakes+AWGN 增强；CIS 输入；GRL RX 对抗 | 是 | 否（GRL 去除） | 无；可选有标签 FT（20 包/DUT-RX） | LoRa 10 DUT，20 SDR，训练 ≤5 RX | 无；源域 val loss | 无；目标是“完全不含 RX 信息” |
| GAN-RXA | 判别器使 RX 后验趋于先验（Eq. 8–9）；SD-RXA 协方差距离；不生成样本 | 否（L-LTF MMSE 均衡） | 否 | 无 | WiSig 50 TX/28 RX（lab 25 RX），400 点，均衡 | 无；20% 源 val，γ 交叉验证 | **承认 γ trade-off**；SD-RXA 假设 RX/TX 特征“uncorrelated” |
| Ma C3a | TDL+Jakes+AWGN；同 DUT 跨两 RX 的 Siamese NT-Xent；对比预训练 | 是 | log-spectrogram 加性 G 项，不增强 | 无（预训练 RX 与测试 RX 是否重叠未说明） | 3 公开 LoRa + 自采；训练 2 RX，测试至多 20 RX | 无；源域 val loss | 声称“without information loss”，隐含不共线 |
| Xie C3b | 多天线谱平均 + log 域 d 阶差分（高通） | 是（多径，多项式逼近） | 不涉及（单 RX） | 不涉及 | 12 Wi-Fi，1×4 天线 X310 | 无 | 假设 RFF 在高频；平滑 TX 成分会被删 **[推断]** |
| RIEI A2 | 解耦：CE + MI(cos) − IE(entropy)；无增强 | 否 | 否 | 无 | WiSig 6 TX/6 RX（2→1）；HackRF 4/3 | 无；报最后 5 epoch | 假设 P(X^ν)=G#(P(X)×P(ν))，Z_X⊥Z_ν |
| Division C2b | L-STF/L-LTF、HT-LTF/L-LTF、参考设备除法 | 是（乘性 LTI） | 通过除法消去 | HL 无；RD 需目标 RX 采参考设备 | 14 WiFi，5 USRP | η_LF 选参考设备（p=0.0034） | 假设 F_R 字段间相同而 F_T 不同；**Dev12/14 身份塌缩** |
| C2a | DSQ（相邻子载波除法）+ TCNN 校准 | 是（TDL 仿真） | 仿真 RX 只有 IQ imbalance + LNA 非线性，无频响 | **需要**目标 RX 校准数据（无 DUT 标签） | 仿真 12 TX/3 RX | 无 | 无 |
| C2e | UDA：DV-KL 对齐 + 伪标签 | 否 | 否 | **无标签** | WiSig ManySig 6 TX/12 RX，均衡 | 超参数敏感度看目标 | 无 |
| C2c | SFDA：MS-SHOT | 否 | 否 | **无标签**，且无源数据 | ManySig 6/12；80 TX/26 RX；HackRF | 报最后 5 epoch | 无 |
| C2d（附） | VAT-KL + GRL + 结构锚定 | 否（承认非物理） | 否 | **有标签**目标已知设备 | WiSig ManyTx；合成 LoRa | 源域校准阈值，未用 target-unknown 选择 | 无 |
| WiSig A6 | 无方法；MMSE 均衡 | — | **列为 open problem** | — | 174 TX/41 RX；ManyRx 10 TX/32 RX | — | — |
| CIDG | 类条件分布不变（kernel scatter） | 否 | — | 无 | 视觉 | 无 | 类条件对齐可避免类间错位 **[推断]** |
| MASF | episodic meta + class-confusion 对称 KL + metric | 否 | — | 无 | 视觉/医学 | 无 | 保持类关系（跨真实 domain） |
| MixStyle | 特征 mean/std 随机插值 | 否 | — | 无 | 视觉 | 无 | 无 |
| ConstStyle | Gaussian style + barycenter，训练与测试映射 | 否 | — | 无（只用单个测试样本） | 视觉 | 无 | partial alignment 防信息损失 |
| SWAD | dense weight averaging（平坦） | 否 | — | 无 | DomainBed | 源域 val 决定平均区间 | 无 |
| D'Amour | stress test 检测 underspecification | — | — | — | 视觉/NLP/医学/EHR | **seed 内 stress test 相关弱** | — |

---

## 4. 我们的 D 证据与文献的对应

1. **matched-norm 干预（real ≈ random；smooth tilt 更伤；real RX 方向单位破坏力最小；TX 方向最大）**：读过的论文里没有任何一篇在冻结 RFFI 模型上把真实接收机响应与等幅随机谱方向对照。最接近的只有：WiSig、Shen 的“换 RX 掉点”现象描述，以及 C2a 的仿真 RX（但没有频响）。**[推断]** “real RX 方向单位破坏最小”很可能源于 14 RX 多接收机训练已经覆盖了真实 RX 方向（WiSig Fig. 10 显示约 20 RX 饱和）。这对 (a) 很不利：按真实 RX 协方差校准的增强，很大程度上在重复训练集里已有的变化。
2. **source-only 敏感度预测未见 RX（ρ≈−0.96，random ≈ real，3 个 seed 簇混杂）**：RFFI 文献里没有对应工作。ML 里 D'Amour 给出 seed 内弱相关的反证，EoA 说明 OOD 沿训练轨迹不稳定。所以“预测”成立的范围要先限定在“簇间/recipe 间”。
3. **主 RX 轴（96% 方差、低/高带不对称）承载 35% 的 TX 能量（随机期望 8%）**：与下列显式假设冲突：SD-RXA 的 "uncorrelated"、RIEI 的 Assumption 1 与 Z_X⊥Z_ν、Ma 的 "without information loss"、C2b 的 F_R 字段间相同而 F_T 不同、Xie 的“RFF 在高频”。文献里能找到的支持性证据只有两条：GAN-RXA 的 γ trade-off，以及 C2b 的 Dev12/Dev14 塌缩，都是零散证据。**这是本文最硬的新发现。**
4. **Shen Strict-CIS：输入 RX share ≈2%，特征 RX share <1%，线性擦除只会更差；失败是近邻 TX 对单向塌缩**：与 CIS 消去静态 LTI 的数学性质一致（本地 VIRP V0 已核对 CIS 为 `log10(|STFT[t+1]/STFT[t]|²)`）。由此推断，Shen 上剩下的 RX 效应主要是非 LTI 的（C2a 列举了 IQ imbalance、LNA 非线性等），**(a) 在 Shen 上基本无作用**。对口的方法是类条件/最坏组/元学习（CIDG、GroupDRO、MLDG、MASF），这与我们观察到的 MLDG/GroupDRO 有效、SWAD 有害一致。
5. **date 漂移单位破坏力 ≈ 换 RX**：方向上与 Shen §V-B-1（RTL 跨天掉超过 40%）、WiSig §VI-B（均衡后跨天仍有下降）一致。两者都没做幅度归一化比较，我们的归一化比较是新的。

---

## 5. 判定：(a)(b)(c) 各自已做 / 未做

### (a) 物理 LTI FIR 谱随机化 + source RX/date 校准 + 身份保持

- **已做的部分**：
  - 随机 LTI FIR 作为增强：TDL 多径本身就是随机 FIR（Shen §IV-C、Ma §IV-B；按引用文献描述的 Soltani 2020）。
  - 用 source domain 协方差建模扰动/对齐：在特征层由 ConstStyle 完成（Gaussian + barycenter）；MixStyle 是非参数随机版。
  - 确定性的“接收机 LTI 不变量”：C2b 的 DV/HL/RD、C2a 的 DSQ、Xie 的高阶差分、Shen 的 CIS、WiSig/GAN-RXA 的 L-LTF 均衡。
- **读过的论文里没做的部分**：
  - **用实测 source RX 与 date 的音调响应差来校准幅度/协方差的接收机响应增强**（WiSig 2022 将其列为 open problem）；
  - **输入层的 TX 身份轴保护**（不在与 TX 判别方向重合的 RX 轴上强加不变性）。
- **风险**：
  1. 本地未读的潜在先行工作见 §7，尤其 Rajendran & Sun TIFS 2022、Cai et al. TIFS 2025、TCCN 2026、FAN+IB TIFS 2026，以及视觉领域的 RandConv；
  2. **我们自己的数据不支持“校准优于随机”**（ratio 0.96，real RX 方向单位破坏最小）；
  3. Shen 上无作用，只在 WiSig 未均衡输入上可能有用，单数据集主张站不稳；
  4. “物理可实现”要说清楚：zero-phase FIR 是非因果的，加上整体延时等价于线性相位因果 FIR，延时由同步吸收；min-phase 是另一种可实现的选择；
  5. 真实 RX 还有非 LTI 成分（CFO 残差、IQ imbalance、非线性），LTI 增强只覆盖其中一部分。

### (b) 源域反事实谱敏感度作为无目标模型/seed 选择

- **已做**：RFFI 里没有；读过的论文都用源域 val loss/acc（Shen、GAN-RXA、Ma、C2b、Xie），或者报告目标曲线（RIEI、C2c、C2e）。ML 里，“源侧 stress test 预测 OOD”是已知思路，D'Amour 给出 seed 内弱相关的反证；DomainBed/EoA 给出必须比较的选择准则。
- **真正未做**：在跨接收机 RFFI 里，一个物理参数化、无需目标数据、能预测未见 RX 准确率的信号。**但** random 与 real 同样能预测，说明“反事实接收机”这一点没有增量，它就是一个通用谱敏感度/平坦度指标。要成立，必须同时满足：
  1. 簇内（同 recipe、不同 seed）秩相关显著；
  2. 选择 regret 小于源域 val acc 与 leave-one-RX-out CV；
  3. 优于简单 proxy：Gaussian 噪声敏感度、输入梯度范数、参数 sharpness。

### (c) 原视图与扰动视图的 KL consistency

- **已做**：C2d（VAT-KL，跨接收机 RFFI）、Ma（跨真实 RX 的 Siamese NT-Xent 加信道增强视图）、MASF（跨 domain 对称 KL）；视觉领域的 AugMix-JSD、RandConv-KL、FixMatch 等 **[未核实细节，凭记忆]**。
- **未做**：只有 (c)+(a) 的扰动分布组合才可能新。损失本身不是贡献。

**直白的建议**：
- 论文主线放在诊断发现上：RX 鲁棒性等价于通用谱形敏感度；RX 变化与 TX 身份纠缠；漂移 ≈ 换机。
- (a) 只作为由诊断直接推出、必须过强基线的最小方法，并去掉“按接收机校准”这一卖点，除非 B3 消融证明校准有用。
- (b) 只在通过簇内检验后才写，否则降级为“诊断相关性”。

---

## 6. 审稿人会要求的最小基线（按优先级）

统一要求：每个方法 ≥3 seeds，按 DomainBed 的 training-domain validation 选模型并同时报告 leave-one-RX-out CV；Shen 与 WiSig 两个数据集都要报。

**第 1 档（缺了会直接被拒）**
1. **ERM 多 RX 合并训练**（即现有 R0/R1）。
2. **信道增强（Soltani/Shen/Ma 风格）**：TDL(exp-PDP) + Jakes + AWGN。参数按已发表范围：Shen 为 τ_rms [5,300] ns、f_D [0,10] Hz、K [0,10]、SNR [0,50] dB；Ma 为 [5,300] ns、[0,5] Hz、[10,40] dB。理由：TDL 就是随机 FIR，它是 (a) 最直接的竞争者。
3. **等幅谱增强消融（决定 (a) 生死）**，五个版本在相同 ‖Δ‖ 下比较：
   - 各向同性随机 per-tone log-gain；
   - 平滑 tilt/低阶多项式；
   - 用 source RX 协方差采样；
   - 真实 source RX 互换（transplant b[r′]−b[r]）；
   - date 漂移互换。

   再加上 “协方差采样 + 投影去除 RX∩TX 共享轴” 这一 (a) 的身份保持版本。
4. **接收机对抗**：Shen GRL（DANN 型）、GAN-RXA（Eq. 8–9，γ=1，两阶段训练）。两者使用的 RX 标签信息与 (a) 相同。
5. **类条件/最坏组/元学习**：GroupDRO（按 RX 分组）、MLDG、MASF、CIDG。我们已经看到 MLDG/GroupDRO 减轻 Pluto 塌缩，这一组是 Shen 上的真正对手。
6. **平坦度**：SWAD（可加 SAM）。我们已观察到它在 Shen 上有害，必须照实报告。

**第 2 档（强烈建议）**
7. **特征统计增强**：MixStyle（α=0.1，p=0.5，浅层）、ConstStyle（含测试时 partial alignment）。若能取得代码，再加 RFFI 版 MixStyle（Cai et al. ASA）。
8. **确定性不变量 front-end**，直接检验身份保持主张：
   - Shen 上已经是 CIS；
   - WiSig 上做 12 音 log 功率的相邻音比值或 Xie 的 d 阶差分（d=1–3）；
   - 若决定解封，再加均衡版 WiSig（GAN-RXA、C2e、C2c 都用均衡数据，这是审稿人最自然的问题，是否解封由用户决定）；
   - DV/HL 在 256 点 compact 窗上不可行（L-LTF 不完整，见 §2 A6 **[推断]**），需要 raw WiSig；如果不做，要在文中写明原因。

   如果这些 front-end 与 (a) 一样好，“身份保持”主张就不成立；如果更差，正好佐证发现 3。
9. **一致性损失与扰动分布分离**：VAT-KL 与 AugMix 式 JSD 用通用扰动，(c) 用 (a) 的扰动，三者同预算比较。

**(b) 专用**
10. 选择准则对比：源域 val acc、leave-one-RX-out CV、Gaussian 噪声敏感度、输入梯度范数、参数 sharpness；EoA/集成作为不需要选择的替代方案。报告：
    - 全体与**簇内** Spearman；
    - top-1 选择 regret；
    - 随机选择的 regret 分布。

**只作上界/背景、不能当对手**：有标签目标 fine-tune（Shen §III-G）、UDA（C2e）、SFDA（C2c）、需要目标 RX 校准的 C2a/RD。

---

## 7. 未读或未核实的潜在先行工作（投稿前必须补读）

| 工作 | 为什么危险 | 状态 |
|---|---|---|
| Soltani et al., Commun. Mag. 2020 "More Is Better" | 信道增强基准，可能涉及更多增强类型 | 未取得全文/摘要；只有二手描述（信道增强） |
| TCCN 2026 "Multi-Scale Complex-Valued Learning and Balanced Domain Alignment"（WiSig 子集） | 同数据集上的对齐方法，可能是直接竞争者 | 未取得；不作内容陈述 |
| Rajendran & Sun, "RF impairment model-based IoT physical-layer identification for enhanced domain generalization", TIFS 17:1285–1299, 2022 | 用物理 impairment 模型做 DG，可能已有“物理参数化增强”；被 Shen、GAN-RXA、Ma 引用 | 只核对了引用条目，未读全文（本地 VIRP V0 审计亦如此） |
| Cai et al., "Toward robust RFFI via adaptive semantic augmentation", TIFS 20:1037–1048, 2025 | RFFI 版 MixStyle（2D-DWT + mean/std 混合，据作者仓库） | 只核对了作者代码，未读全文 |
| FAN + IB-RFF, TIFS 21:8008–8021, 2026 | WiSig ManyRx/ManySig 上的频率处理 + HSIC 信息瓶颈 | 本地早期审计只读了记录与代码 |
| Wang et al., "Avoiding shortcuts ... single-source DG", TWC 24(4), 2025 | random overlay augmentation 做信道 DG | 只见于 DRIFT 的二手描述 |
| Zha et al., SPL 2023；Li et al., JSAC 43(6), 2025；Merchant & Nousain, Globecom-W 2019；Karunaratne et al., GLOBECOM 2021 | 跨接收机对比/DA、接收机变换网络、生成式增强 | 未读 |
| RandConv（Xu et al., ICLR 2021）、Pro-RandConv（CVPR 2023）、DSU（ICLR 2022）、AugMix（ICLR 2020） | 视觉领域“随机卷积（=随机 LTI 滤波）增强 + KL 一致性”、按 batch 方差采样特征统计扰动。若 RandConv 确实含 KL 一致项，(a)+(c) 的组合在视觉里就已有原型 | **凭记忆，未核实** |
| ICCCN 2026 RIFLE | 不完全 TX–RX 覆盖下的 GRL | 只见于本地早期审计 |

**核对缺口**：Shen TMC、GAN-RXA、RIEI、C2e 只读了 arXiv 版本，期刊终版的新增实验没有核对；Ma 与 Xie 的期刊信息无法从本地文本确认。
