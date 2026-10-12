# G1 联合复核与执行决策

|Dataset /方向|Full CIS Acc|Strict CIS Acc|配对ΔF1|正向seed|源任务均学会|
|---|---:|---:|---:|---:|---:|
|shen 10_training_RX→shen_b200_1_test|69.75%|83.60%|+0.1226|3/3|3/3|
|shen 10_training_RX→shen_n210_1_test|84.15%|97.58%|+0.1546|3/3|3/3|
|shen 10_training_RX→shen_rtl_4_test|62.63%|85.10%|+0.2386|3/3|3/3|
|shen 10_training_RX→shen_rtl_8_test|91.53%|93.92%|+0.0233|2/3|3/3|
|osu osu_RX1→osu_RX2|10.11%|11.76%|+0.0090|3/3|3/3|
|osu osu_RX2→osu_RX1|10.75%|13.61%|+0.0212|3/3|3/3|

操作关卡：positive_external_replication_allows_exploratory_G2_0。G2-0允许=True；G2-1/G3未启动。关卡是后续资源/研究决策，不是显著性检验或普适物理机制证明。

最严重CIS类别回退：{'dataset': 'shen', 'domain': 'shen_b200_1_test', 'seed': '20261012', 'raw_label': '36', 'baseline_recall': '0.93', 'comparison_recall': '0.01', 'new_errors': '184', 'both_source_tasks_learned': 'True'}。完整误判去向见各数据集new_error_destinations.csv，不能只看平均Accuracy或TX34。

所有失败seed作为主结果保留；source_success_conditioned分组只是解释，不替换主结果。Shen十类随机10%，OSU二十五类随机4%。源任务筛查与全零head证据不同，若无法拟合源任务不能将低分单独归因为频谱过滤。Shen Full/Strict CIS源CV分别选16/19epoch且scaler不同，比较的是训练流程而非纯OOB因果；OSU固定20epoch，scaler仍按源输入各自拟合。

OSU窗口数详见OSU_actual_per_TX_windows.csv，周期窗不是严格同步packet。两个方向共享同一对RX，不是两个独立硬件环境样本；每方向只有一个源RX，V_RX缺失。Shen root实际日期/session未知；OSU来自独立采集团队，但研究史留出独立性不能夸大。

训练曲线/失败日志/权重保留在outputs/shen_p2a8/g1_osu_v1/runs和logs。固定频带的效益可因方向、TX和优化不同，尚不能命名为新算法。若允许G2-0，必须先冻结九源拟合、一留出评价，以及源分类准确率负对照；不能使用全十源训练的CNN制造留一泄漏。源域风险是否可预测尚未由G1证明。
