# 第二篇论文 P0 同步机制审计

2026-10-13（北京时间），本轮已完成并停止。

**决策：STOP_NO_OBSERVABLE_SYNC。** 2,000 源包有稳定重复上 chirp 和
条件性符号相位；真实同步不确定性及独立时间参照未建立。模型影响和
标准方法是否解决均尚未检验，未进入 B 或任何 GPU/训练实验。

- [最终决策报告](../../reports/p0_sync/2026-10-13_A0_A1_final/REPORT.md)
- [逐包、RX/TX、硬件类型、两折与数值复核证据](../../reports/p0_sync/2026-10-13_A0_A1_final/EVIDENCE_INDEX.md)
- [完整数值证据 ZIP](../../reports/p0_sync/2026-10-13_A0_A1_final/P0_sync_evidence.zip)
- [结果前登记](PREREGISTRATION.md)、[精确索引](../../configs/p0_sync/sample_manifest.json)
- [文献差异矩阵](LITERATURE_MATRIX.md)、[资料溯源及60TX成本](DATA_AND_REVIEW_PROVENANCE.md)
- [环境版本](ENVIRONMENT.json)、[Git起始状态](GIT_BASELINE.json)

数据只读绝对路径引用，历史工作目录未改。私有原文不在 Git/ZIP。
主审计、独立复核、报告导出分别是 scripts/audit_p0_sync.py、
scripts/verify_p0_sync.py、scripts/report_p0_sync.py。复現使用新输出目录。
合成验证及白名单功能检查通过，100个RX×TX单元独立数值复核通过。
本轮结论是数据/证据适用性不足，不是已证明同步影响不存在。

本地报告首次打包版本（缺少完整复现路径）留在
outputs/p0_sync/report_draft_packaging_v1，仅为导出草稿；最终ZIP已逐项验证
全部封存文件齐全且SHA一致。研究规则和原始实验结果未因此改变。
