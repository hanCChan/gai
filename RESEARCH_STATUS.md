# 第二篇论文研究入口

2026-10-13 UTC：D0.5 signed新增损害审计与B0三seed复验已完成，独立分支experiment/d05-b0-class-risk。
8次实际拟合完成（之前8次零优化的工程启动失败保留），累计0.22810 reserved GPUh。
三个seedStrict平均收益均正；9个RX×TX严重单元≥2seed重复，跨7RX两折，无collapse。
D0.5同CNN内增量风险有信息，但折0不如绝对混淆，跨CNN检索低于uniform；不开发特定Δ新Loss。

- [本轮最终交付与决策](reports/d05_b0/2026-10-13/REPORT.md)
- [D0.5增量风险报告](reports/d05_b0/2026-10-13/d05/REPORT.md)
- [B0完整三seed结果/模型优化诊断](reports/d05_b0/2026-10-13/b0/REPORT.md)
- [更新执行计划](docs/d05_b0/EXECUTION_PLAN.md)
- [B1具体审核计划/预算，未执行](docs/d05_b0/B1_REVIEW_PLAN.md)
- [作者级GRL/IB输入与复现约束](docs/d05_b0/AUTHOR_IMPLEMENTATION_AUDIT.md)
- [全部派生结果/真实日志/初始失败证据ZIP](reports/d05_b0/2026-10-13/evidence.zip)

B1申请32个训练作业（16内部搜索+16正式模型）、上限2GPUh；尚未获准、没有自动运行。
B2/新算法不执行；OSU、开发/锁定、Day3/4、原始IQ、新数据下载均不涉及。
以下为原D0与更早历史入口；其报告与科学判定不变。

2026-10-13 UTC：最新方案是以文献缺口为中心的 D0 类别判别结构审计，已完成后停止。
源池平均Accuracy 68.37%→82.99%，8个有向混淆方向跨两折局部受损；内部留一RX风险预测有信息。
决定 LIMITED_SUPPORT_CONDITIONAL_BASELINE_POC_ONLY：跨seed/独立硬件与强基线剩余缺口未知，不能宣布新算法创新。

- [D0执行计划](docs/d0_class_pair/EXECUTION_PLAN.md)
- [D0完整报告](reports/d0_class_pair/2026-10-13_D0/REPORT.md)
- [论文差异矩阵](docs/d0_class_pair/LITERATURE_MATRIX.md)
- [冻结评价规则](docs/d0_class_pair/PREREGISTRATION.md)
- [全部派生数值与图表ZIP](reports/d0_class_pair/2026-10-13_D0/evidence.zip)
- [有条件的基线验证预算草案，未批准执行](docs/d0_class_pair/NEXT_BASELINE_POC_DRAFT.md)

本轮新训练/推理/数据下载均为0；不读OSU、开发/历史锁定RX、原始IQ。
P0、C0与此前G2/VIRP/EISR停止结论保留；没有重启C1/C2或并行开展候选B/C。
历史C0因时间顺序/session无法核验停止，用户没有批准改成无序融合。

历史入口：

- [最新C0报告](reports/c_evidence/2026-10-13_C0_final/REPORT.md)
- [C0证据索引](reports/c_evidence/2026-10-13_C0_final/EVIDENCE_INDEX.md)
- [更新后的C0/C1/C2计划](docs/c_evidence/EXECUTION_PLAN.md)
- [本轮审计代码](scripts/audit_c_evidence_assets.py)
- [文献差异](docs/c_evidence/LITERATURE_MATRIX.md)
- [P0已发布报告](https://github.com/hanCChan/gai/blob/experiment/p0-sync-mechanism/reports/p0_sync/2026-10-13_A0_A1_final/REPORT.md)
- [P0本地/远端一致性记录](docs/c_evidence/P0_PUBLICATION_VERIFIED.json)

每轮成果发布并回读核验的要求见AGENTS.md。私有原稿/拒稿信不在GitHub，
仅公开事实摘要及来源SHA；源数据和权重只读留在原目录。
