# G1 逐RX、逐seed、逐TX与源任务有效性审计

只分析已保存checkpoint/预测，不修改训练。全部失败保留，条件分组单列。source_task_learned仅为工程筛查（最后epoch源训练准确率高于随机+0.05且无确认全零塌缩），不能证明模型优化完全充分或诊断早期根因。类数为25，随机水平0.04。

|方向/开发RX|Seed|ΔAccuracy pp|ΔMacro-F1|两个源任务均学会|
|---|---:|---:|---:|---|
|osu_RX1→osu_RX2|20261012|+2.20|+0.0114|True|
|osu_RX1→osu_RX2|20261013|+0.37|+0.0022|True|
|osu_RX1→osu_RX2|20261014|+2.38|+0.0133|True|
|osu_RX2→osu_RX1|20261012|+3.04|+0.0341|True|
|osu_RX2→osu_RX1|20261013|+3.16|+0.0151|True|
|osu_RX2→osu_RX1|20261014|+2.36|+0.0144|True|

最严重类别回退及误判去向见worst_CIS_TX_regressions.csv/new_error_destinations.csv。两个源任务学会时的配对仍可能受scaler、epoch、BN和优化轨迹影响，不能当作纯删除OOB因果。11/12之类计数不是独立环境显著性检验。源训练probe仅检查每类首个样本，不能替代完整训练曲线或全源评估。
