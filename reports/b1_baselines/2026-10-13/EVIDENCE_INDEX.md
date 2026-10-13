# B1完整证据索引

- [报告](REPORT.md)、[决策](decision.json)、[32作业/实际GPU占用](execution.json)、[独立核验](independent_verification.json)。
- [冻结注册](../../../configs/b1_baselines/registration.json)、[优化前代码seal](../../../configs/b1_baselines/pre_execution_seal.json)、[40输入/六共享schedule预检](preflight.json)、[学习率选择](selection.frozen.json)。
- [全部RX](per_RX.csv)、[全部Recall](per_RX_TX.csv)、[配对Recall](paired_TX_changes.csv)、[CE比较](CE_comparison.csv)、[逐seed/fold/硬件族](method_summaries.csv)、[跨seed重复](CE_relative_cross_seed_repetition.csv)、[最差单元](minimum_recall_cells.csv)。
- [作业参数/模型SHA](all_job_status.csv)、[480轮优化曲线](training_history.csv)、[16内部搜索](source_search_metrics.csv)、[输入/预测来源SHA](input_provenance.csv)、[GRL额外容量/实际时间](training_cost.csv)。
- [完整ZIP](evidence.zip)：上述原始字节、7200有向对及margin、5400对变化、8000混淆计数、32作业各自真实config/status/history/metrics/process.log及三幅图的PNG/PDF/SVG；[成员SHA清单](ARTIFACT_MANIFEST.json)逐文件核验。
- 训练/数值检查/分析/独立复核/打包源码在 `scripts/*b1*.py` 与 `src/b1_baselines/`。ZIP不包含原IQ、权重或预测NPZ；这些留在 `outputs/b1_baselines/`，其SHA可回查。

报告链接从独立实验分支或固定commit访问。公开树与成员SHA上传后回读核验；本地发布receipt保存在忽略的outputs/publication中。
