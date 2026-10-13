# 第二篇论文当前计划：D0 已执行后停止

用户新方案原文：[USER_PLAN.md](USER_PLAN.md)。此前同步/时序路线停止不变；OSU已从第二篇必要实验移除。只研究类别对判别结构候选，不并行开展B/C，不命名新算法。

| 步骤 | 状态 | 交付/边界 |
|---|---|---|
| D0-1 文献与差异矩阵 | 完成，有明确获取限制 | [矩阵](LITERATURE_MATRIX.md)；ICC/IB/RIFLE全文未知，不作创新排清声明 |
| D0-2 归档资产核验与预冻结 | 完成 | 20份预测、4个checkpoint配置、200包/RX×TX、Full/Strict索引严格一致；[预登记](PREREGISTRATION.md) |
| D0-3 CPU所有类别对诊断 | 完成 | 90有向/45无序，两个CNN折，硬件族控制，LORO及类/去向对照 |
| D0-4 独立复算与决定 | 完成：有限正证 | 14,400个标量、240个risk profile复核通过；[报告](../../reports/d0_class_pair/2026-10-13_D0/REPORT.md) |
| WiSig元数据核查 | 完成 | [说明](WISIG_FEASIBILITY.md)，数据下载0 GB |
| 后续基线验证/算法PoC | 未执行 | [有条件预算草案](NEXT_BASELINE_POC_DRAFT.md)，新训练另行批准 |
| GitHub独立分支发表与回读 | 本轮最终校验步骤 | experiment/d0-class-pair-audit，全树及源码/报告/结果SHA核对；本地commit不等于上传 |

当前决定为 LIMITED_SUPPORT_CONDITIONAL_BASELINE_POC_ONLY：问题值得一轮强基线验证，算法新颖性与跨seed/独立数据集剩余缺口未证实。D0授权的新训练0次、推理0次、OSU/开发/锁定/原始IQ读取0，不重开G2/VIRP/EISR/P0/C0。归档来源只从只读worktree取分数/索引/配置和权重字节哈希；权重不加载、分数不上传。运行后不反复更改指标。

复算方式：在独立输出目录运行 `python scripts/audit_d0_class_pair.py --output reports/d0_class_pair/your_new_run`，需要 registration 所列本地预测资产。仓库无数据/权重。报告ZIP解压后可直接用发布CSV、图表和SHA检查结果；`verify_d0_class_pair.py`依赖上述本地资产，明确不导入metric模块。环境见execution.json。公开文献核查需网络，与模型推理无关。
