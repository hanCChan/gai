# D0 文献差异矩阵（2026-10-13）

结论：已有工作覆盖判别性与不变性的权衡、对比几何、类条件对齐和最差分组风险。D0 只能支持继续做受控基线验证；不能宣称“首次发现过度对齐损害身份”，也不能将一般 margin loss 或 RX×类别 Group DRO 改名为新算法。未取得全文的条目保留 UNKNOWN，不能据此排除先行工作。

核查范围：Shen、Ma、RIEI 的作者 arXiv HTML 全文可访问，核对目标、训练/评价协议及图表目录；IB 核对 IEEE SPS 摘要与作者代码；RIFLE、联邦工作核对出版社摘要；ICC 工作取得 IEEE 在 Crossref 登记的题名/作者/DOI，无摘要、全文。来源免费可访问不等于本次已完整复现。代码只读，不执行。代码固定 commit 和 SHA 见 [author_code_provenance.json](author_code_provenance.json)。本矩阵的“未见”仅限查阅版本，不表示作者没有隐式处理问题。

| 论文／可核查材料 | 具体目标 | 类别对 margin/confusion 分析 | 未见 RX 最差类别 | pairwise / class-conditional 鲁棒优化 | 目标 RX 数据 | 公开代码 | 对候选问题的关系 |
|---|---|---|---|---|---|---|---|
| [Shen：Towards Receiver-Agnostic and Collaborative RFFI](https://arxiv.org/html/2207.02999)，TMC；查阅作者 v1，非出版社最终版 | CIS，TX CE + RX 对抗 CE/GRL；多 RX 软融合 | III-E 式10–15是 CE/GRL；V–VI 的图8–16侧重 RX/场景 accuracy；查阅版本未见逐对 margin 实验 | 报差 RX 准确率；未见 min-TX Recall 指标 | CE 已隐含类别竞争；没有显式最差 RX×类别对目标 | 核心 DG 不用；另有需要新 RX 少量有标签包的可选 fine-tune | [作者仓库](https://github.com/gxhen/receiverAgnosticRFFI) 已核验 | 直接 receiver-invariance 基线；不能将 D0 的纯 CIS CNN 当作复现 GRL |
| [Ma：Towards Channel-Robust and Receiver-Independent RFFI](https://arxiv.org/html/2512.12070v1)，TIFS 2025，作者全文 | 无监督 NT-Xent 预训练；跨 RX 同 TX 正对，CE+NT-Xent Siamese；信道增强 | IV-D 显式样本对几何；VII–VIII图9–17主要 RX/场景/训练包预算表现；未见逐类别对 logit margin表 | 查阅版本未见未见 RX 的 min-TX 指标 | 已做跨 RX 同类 pairwise 一致性；未见 receiver-wise 最差类别对 minimax | 训练的两 RX 使用 TX 标签；评价其他 RX 的分类不需目标适配；预训练来源另列 | 本次未找到经作者确认的配套仓库；不称“无代码” | 最接近的跨 RX 对比基线之一；普通“保护同类结构”已被覆盖 |
| [RIEI：Domain Generalization for Cross-Receiver RFFI](https://arxiv.org/html/2411.03636)，IoT-J，作者 v1全文 | Separable Condition；TX/RX CE、互独立项、交叉分类信息熵，交替优化；FedRIEI | 图6已有按 emitter 着色的 t-SNE，显示类别重叠变化；表II–VII为总体识别性能；未见逐对 logit margin | 理论处理未知 RX 分类误差；未见最差 TX 保证或该指标 | 判别性+解耦是明确目标；不是“只对齐而不管类别” | 定义 source-only DG，无需训练时目标 RX 数据 | 作者全文未提供配套实现链接，定向检索未验证到 | 直接先行理论/方法；新增解耦或泛化界本身不足以创新 |
| [IB-RFF：Information Bottleneck-Driven Cross-Receiver RF Fingerprinting for Physical-Layer Security](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security/2026/09/information-bottleneck-driven-cross-receiver-rf-fingerprinting-physical-layer-security)，TIFS 2026；摘要+代码，全文未取得 | FAN 频率模块；两阶段分类+HSIC(hidden,input) IB 正则；不需 RX 标签 | 摘要已明确 over-alignment 可能伤害判别性；main.py 无逐对鲁棒 loss；全文图表 UNKNOWN | 全文实验 UNKNOWN；公开 main.py 输出 accuracy，不足推断论文未报逐类结果 | 判别性压缩与抑制接收机冗余；已覆盖候选的宽泛动机；显式 pairwise 最差目标未在所查代码出现 | main.py 源 RX train/validation，测试 RX 单独划分；核心训练不使用目标 RX | [作者 MIT 仓库](https://github.com/BeechburgPieStar/IB-RFF)，固定5a03a04… | 最重要直接竞争者，必须同预算对照+原实现核验；不能先宣布其剩余缺口 |
| [RIFLE：Receiver-Invariant RFFI under Partially Observed IoT Deployments](https://doi.org/10.1109/ICCCN69946.2026.11662855)，ICCCN 2026；出版社摘要，全文未取得 | 非完整 TX–RX 覆盖导致身份/RX统计耦合；残差特征+GRL RX 判别器 | 摘要无足够细节，全文 UNKNOWN | UNKNOWN | 解决部分覆盖与身份/RX耦合；是否包含最差类或对损失 UNKNOWN | 摘要为 cross-RX generalization；精确目标可见性需全文确认 | 本次未确认作者实现 | 用户候选不能把部分覆盖、标签耦合当新发现 |
| [Song 等：Receiver-Agnostic RFFI via Supervised Contrastive Learning and Parametric Wasserstein Barycenters](https://doi.org/10.1109/ICC59461.2026.11586949)，ICC 2026 | IEEE 登记题名确认 SupCon+参数化 Wasserstein barycenters；公式/是否类条件 barycenter UNKNOWN | UNKNOWN，不能从题名猜测 | UNKNOWN | 题名已构成对比+分布几何的直接竞争，具体鲁棒优化 UNKNOWN | UNKNOWN | 定向题名/作者检索未确认公开仓库 | 在全文补齐前，新颖性无法排清。元数据见 [icc_publisher_metadata.json](icc_publisher_metadata.json) |
| [Receiver-Agnostic RFFI via Federated Learning](https://ieeexplore.ieee.org/document/11373634/)，TMLCN 2026；出版社摘要 | 联邦对抗训练；标签偏斜分析与 Label Loss Driven 客户端选择 | 全文 UNKNOWN | UNKNOWN | 已关注标签非均衡与 RX 客户端贡献 | 摘要为未见 RX 泛化；适配细节 UNKNOWN | 未确认 | 普通 RX 选择、均衡和联邦标签偏斜已被覆盖；不启动候选B |
| [CSCNet：Cross-Receiver RFFI: A Source-Free Adaptation Approach](https://www.mdpi.com/1424-8220/25/14/4451)，Sensors 2025；出版摘要及可索引目标段落，全文获取遇限流 | 新 RX 上伪标签 CE、熵最小化、对比；仅保留源模型 | 对比按伪类别几何处理；完整 confusion/图表 UNKNOWN | UNKNOWN | 类别相关目标已存在；目标域自适应设置 | **需要无标签目标 RX 数据** | 未确认 | 可对比部署需求，不能与完全 source-only 方法混作公平主表 |
| [DRIFT](https://arxiv.org/html/2510.09405v2)，2026作者预印本 | TX/RX CE，GRL、RX中心一致性、两部分特征分离 | 显式表示结构约束；未确认逐对最差指标 | 未确认 | 类别判别、RX冗余解耦已覆盖；查阅目标未见最差有向类对 minimax | source-only DG | 本次未确认作者代码 | 直接解耦竞争；预印本身份明确，不当已发表创新排除证据 |
| [SupCon](https://arxiv.org/abs/2004.11362)，NeurIPS 2020 | 所有同类 positives 拉近、异类 negatives 推开 | 本身显式 pairwise；无本任务逐 RX confusion | 非 RFFI 未见 RX 实验 | 类条件几何；原始目标不是 RX×pair worst risk | 只用训练标签 | [作者代码](https://github.com/HobbitLong/SupContrast) | 任意“加监督对比”不新颖，必须公平基线 |
| [ArcFace](https://arxiv.org/abs/1801.07698)，CVPR 2019 | 归一化特征/权重，加性角度 margin softmax | 显式类间 margin，但不同于 D0 原始 logits 尺度 | 非 RFFI | 显式边界约束，非 receiver-aware 最差对 | 训练标签 | [官方项目](https://github.com/deepinsight/insightface) | margin 或归一化技巧已有，不能重命名 |
| [Group DRO](https://arxiv.org/abs/1911.08731)，ICLR 2020 | 已知训练 groups 的最大平均损失，结合正则改善 worst-group 泛化 | groups 可设 RX 或 RX×TX；不等于显式有向错误边 | 原论文直接 worst-group；未知 RX 的类别保护不自动保证 | RX×TX Group DRO 已覆盖最差域类风险的朴素提法 | 源数据有 group ID | [作者代码](https://github.com/kohpangwei/group_DRO) | 强公平反证；把 RX×TX 重加权称新算法不成立 |
| [CDAN](https://arxiv.org/abs/1705.10667)，NeurIPS 2018 | 特征与类别预测联合条件化对抗域对齐；熵条件加权 | 已显式用类别结构做分布对齐 | 非本任务最差 TX验证 | class-conditional DA 已存在 | **需要无标签目标数据** | [作者项目](https://github.com/thuml/CDAN) | 概念先行；source-only 和目标可见性须区分 |

## 可行的新问题与反证

可继续核查的窄问题仅是：在相同训练预算、结构和标签可见性下，是否还有跨 seed/独立 RX 可迁移的有向错误边风险，无法由已有对比、margin、Group DRO 与 IB 解释或压低。D0 的“风险排序更准”不是“对这些边加权训练一定更好”。CE 本身可写为 log(1+Σ exp(−m_ij))，因此仅显式写出 margin 项不能构成新目标。最差 RX×TX 分组也已有 Group DRO 直接可用。

我们**没有证明**任何强基线解决不了；也没有取得 ICC、IB-RFF、RIFLE 全文以排清细节。现阶段反证最强的是“宽泛动机已有、简单 loss 已有”；正证最强的是归档源池内可预测的残余混淆。两者共同支持一次有预算的基线验证建议，而非新算法 GO。

## 获取与检索记录

核查日：2026-10-13 UTC。首选作者 arXiv、IEEE Xplore/SPS、作者 GitHub、UCLA。ICC 起初关键词检索无结果，改用 fingerprinting+barycenter 定位完整题名，再用 DOI 的 [IEEE登记元数据](https://api.crossref.org/works/10.1109/ICC59461.2026.11586949) 确认。IEEE全文端返回418/机器人验证；RIFLE/IB仅摘要可核对，不使用付费聚合站代替全文。RIEI/Ma 定向题名+github检索未确认实现；这不是不存在实现的证明。主矩阵没有将二手摘要用于技术结论。
