# G1 逐RX、逐seed、逐TX与源任务有效性审计

只分析已保存checkpoint/预测，不修改训练。全部失败保留，条件分组单列。source_task_learned仅为工程筛查（最后epoch源训练准确率高于随机+0.05且无确认全零塌缩），不能证明模型优化完全充分或诊断早期根因。类数为10，随机水平0.1。

|方向/开发RX|Seed|ΔAccuracy pp|ΔMacro-F1|两个源任务均学会|
|---|---:|---:|---:|---|
|10_training_RX→shen_b200_1_test|20261012|+25.60|+0.2497|True|
|10_training_RX→shen_b200_1_test|20261013|+5.95|+0.0447|True|
|10_training_RX→shen_b200_1_test|20261014|+10.00|+0.0734|True|
|10_training_RX→shen_n210_1_test|20261012|+22.55|+0.2741|True|
|10_training_RX→shen_n210_1_test|20261013|+5.95|+0.0618|True|
|10_training_RX→shen_n210_1_test|20261014|+11.80|+0.1280|True|
|10_training_RX→shen_rtl_4_test|20261012|+28.45|+0.3211|True|
|10_training_RX→shen_rtl_4_test|20261013|+17.40|+0.1774|True|
|10_training_RX→shen_rtl_4_test|20261014|+21.55|+0.2173|True|
|10_training_RX→shen_rtl_8_test|20261012|+13.25|+0.1444|True|
|10_training_RX→shen_rtl_8_test|20261013|-6.55|-0.0789|True|
|10_training_RX→shen_rtl_8_test|20261014|+0.45|+0.0046|True|

最严重类别回退及误判去向见worst_CIS_TX_regressions.csv/new_error_destinations.csv。两个源任务学会时的配对仍可能受scaler、epoch、BN和优化轨迹影响，不能当作纯删除OOB因果。11/12之类计数不是独立环境显著性检验。源训练probe仅检查每类首个样本，不能替代完整训练曲线或全源评估。
