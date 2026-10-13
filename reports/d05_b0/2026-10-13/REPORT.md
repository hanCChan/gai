# D0.5 + B0 最终交付与Stop/Go

已完成用户授权：D0.5、B0限额8次训练、B1实施审核矩阵、作者GRL/FAN源码与适配成本审计；**完成后停止，B1/B2/新Loss未执行**。

1. **新增损害比普通类难度可预测吗？** 同CNN对内有：signed Spearman0.416/0.343，超Recall/类难度/factorized对照。但fold0明显不如绝对混淆，跨CNN正向检索低于均匀。不能把Δ预测器优越性当研究主张。[D0.5完整报告](d05/REPORT.md)。
2. **严重退化跨seed重现吗？** 有有限复现：三seed平均收益均正，9个单元≥2/3seed下降≥10pp，跨7RX、两折，无collapse；仅2单元三seed都严重，灾难幅度有随机性。[B0完整报告](b0/REPORT.md)。
3. **简单/强方法能否解决？** 仍UNKNOWN。B0全是历史CE；D0.5预测baseline不是新的神经训练baseline。[B1审核计划](../../../docs/d05_b0/B1_REVIEW_PLAN.md)给出16个内部搜索+16正式模型，32作业、480轮，估0.7–2GPUh/建议总上限2；尚无许可、无训练结果。
4. **足够支撑独立算法论文吗？** 尚不足。局部风险值得受控强基线反证，但source Δ跨拟合模型检索失败是重要限制；SupCon/DRO/GRL/IB等直接竞争未排清。暂不实现新pairwise loss。[作者方法核查](../../../docs/d05_b0/AUTHOR_IMPLEMENTATION_AUDIT.md)区分同架构机制与论文级方法，FAN/LoRa输入和完整成本待审计/实测。

投入：8次实际训练，8次零步工程启动失败原样保留；新增最终分类输出80,000；累计reserved GPUh 0.22810。只读十源RX CIS缓存与归档分数，数据下载0GB，开发/锁定/OSU/Day3/4/原始IQ访问0。原D0和所有历史负结果未改。

独立复核：D0.5 240 profile、B0 60 model×RX/600 Recall/6000 confusion以及8个真实checkpoint/3140 Adam步；全部通过。B1仅有可执行配置生成和约束检查，四目标训练runner未在本轮添加，不能称B1已跑通。

公开证据：[evidence.zip](evidence.zip)，包含所有完整派生CSV、图表、逐epoch日志/配置、初始失败记录及每成员SHA。大型向量/margin放ZIP，小汇总直接GitHub可读；原始NPZ/权重不发布。代码、更新计划、注册、决策和来源SHA同分支公开。全树/文件/ZIP回读校验作为发表最终步骤。
