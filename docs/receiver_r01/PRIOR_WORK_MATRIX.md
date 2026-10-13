# 强先行工作与候选贡献差异

2026-10-13 核查论文主文、作者代码及公开实现。以下是本轮定位，不是宣称已经证明的新贡献。

| 先行工作 | 已有机制 | 本轮处置 | R2 必须新增的证据 |
|---|---|---|---|
| [MLDG, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11596) / [作者代码](https://github.com/HAHA-DL/MLDG) | 源域内模拟 train/test 域变化，优化虚拟更新后的留域分类损失 | 复现一步 SGD 内层、完整二阶元梯度；RX/日期域只是应用协议 | 多留出 RX 的更新风险反馈能超过标准元损失；不能以留 RX 命名创新 |
| [SWAD, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf) / [作者代码](https://github.com/khanrc/swad) | 稠密参数平均及源验证 LossValley 防过拟合，追求平坦解 | 作者 LossValley/AveragedModel 按 MIT 许可收入本分支，仅改 import；每步平均、每 epoch 分段 | 更新风险控制须超过既有优化稳定性收益，并公平计成本 |
| [Fishr, ICML 2022](https://proceedings.mlr.press/v162/rame22a.html) | 域间逐样本损失梯度方差匹配、EMA；论文讨论梯度协方差/Fisher/Hessian 与局部域损失面 | 查阅主文与作者/DomainBed 实现；本轮鲁棒基线预固定 Group DRO，Fishr 未拟合 | 不能把方差正则或统计梯度对齐重新命名；后续需要同预算 Fishr 实跑，不能声称已胜过 |
| [PCGrad, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) / [作者代码](https://github.com/tianheyu927/PCGrad) | 负梯度内积时投影冲突梯度，以局部一阶信号处理任务冲突 | 两个随机 RX 组任务的标准对称投影对照，未新增接受规则 | 非线性有限更新后的多环境风险，应提供超过一阶梯度手术的可测增量 |
| [Group DRO, ICLR 2020](https://arxiv.org/abs/1911.08731) / [作者代码](https://github.com/kohpangwei/group_DRO) | 预定义最差组训练损失，正则和早停对最差组泛化重要 | 继承 RX×TX 指数权重更新及共用源内模型选择；WiSig组含日期 | 不能把组损失重加权或尾部训练损失当更新可靠性原创 |
| [DomainBed, ICLR 2021](https://openreview.net/pdf?id=lQdXeXDoWtI) / [代码](https://github.com/facebookresearch/DomainBed) | 公平划分训练、模型选择和测试协议，域泛化比较易受搜索/选择条件影响 | 有限两 LR、共同训练 horizon、三配对 seed、只源选参，报告实际额外计算 | 新机制须同数据/骨干/选择预算，不能从内部 RX 结果倒推最佳系数 |
| [RIEI](https://arxiv.org/html/2411.03636v1) | 跨 RX DG、可分离条件及 TX/RX 特征分解；含联邦扩展 | 阅读建模和方法部分；其问题设定本身已存在 | 目标设定不是新贡献；本轮不声称超越 RIEI，后续发表需按适用输入复现 |
| [DRIFT v2, 2026-05-26](https://arxiv.org/html/2510.09405v2) | 先等化，再 TX/RX 双表征、GRL、接收机一致性和特征分离 | 核查模型设计/协议；本轮未等化封存对照不偷读等化版 | 训练更新控制与解耦损失有可能不同，仍需跨方法公平实验；不得用跨 RX 名称充当新意 |

MLDG 二阶目标使用原始元学习思想，外层 objective 除以2是冻结的尺度归一化。DomainBed 当前实现公开使用一阶近似，内层 Adam；本轮一步 SGD/全二阶不是逐代码复现 DomainBed。标准方法在 RFFI 上的实测收益只是基线证据。

训练中局部负内积仅是局部一阶风险提示。多 RX 上损失变化不能自动证明未知 RX 迁移不稳定的因果机制，尤其两包/类探针噪声及 dropout 训练和 eval 的差异。本轮只作来源明确的诊断，R2 应先做独立包重测和冻结模型轨迹的预测检验。需要 CVaR/类条件鲁棒优化、受约束优化和其它梯度约束文献对照；目前数学新颖性未确证，禁止提前称原创。
