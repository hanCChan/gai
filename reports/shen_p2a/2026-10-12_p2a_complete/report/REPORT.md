# P2-A 实验结果与决策



共108次正式训练：52候选+冻结28组的56个额外seed。Day2为开发集；所有候选、三seed汇总、真正十类Macro-F1、逐包配对、每设备、梯度和成本均保存。



|Receiver|Group|Method|Acc mean ± sample SD|Macro-F1 mean|Params|Train s|Forward packets/s|Peak VRAM MiB|
|---|---|---|---:|---:|---:|---:|---:|---:|
|n210_1|A1_matched_representation|cis_light_train_scalar|99.22 ± 0.35%|0.992177|658314|100.5|9163|398|
|n210_1|A1_matched_representation|spec_light_train_scalar|98.73 ± 0.25%|0.987343|658314|105.8|10459|398|
|n210_1|A2_capacity_raw_cis|cis_light_raw|99.25 ± 0.23%|0.992495|658314|113.5|11323|398|
|n210_1|A4_frequency_rings|far_freq_independent|56.02 ± 1.21%|0.554316|659370|94.0|11316|255|
|n210_1|A4_frequency_rings|ib_freq_independent|68.67 ± 4.93%|0.655509|659370|164.1|11582|255|
|n210_1|A4_frequency_rings|mid_freq_independent|47.53 ± 2.34%|0.464693|659370|127.0|8788|255|
|n210_1|A4_frequency_rings|near_freq_independent|86.30 ± 1.95%|0.861641|659370|110.6|9514|255|
|n210_1|A4_frequency_rings|oob_freq_independent|72.63 ± 1.08%|0.706057|659370|82.9|9816|255|
|n210_1|A5_information_complementarity|capacity_ib_ib_concat|91.07 ± 6.06%|0.909514|1318730|184.1|6098|435|
|n210_1|A5_information_complementarity|capacity_ib_ib_gate|92.37 ± 1.48%|0.922949|1317707|257.2|5220|435|
|n210_1|A5_information_complementarity|ib_oob_concat|93.17 ± 0.52%|0.931406|1318730|160.9|7059|435|
|n210_1|A5_information_complementarity|ib_oob_gate|93.20 ± 1.67%|0.931960|1317707|193.6|5263|435|
|n210_1|A5_information_complementarity|single_ib|92.92 ± 2.05%|0.927479|659370|178.7|9648|255|
|rtl_6|A1_matched_representation|cis_light_train_scalar|73.52 ± 3.40%|0.674086|658314|74.7|7651|398|
|rtl_6|A1_matched_representation|spec_light_train_scalar|67.43 ± 1.03%|0.590525|658314|67.2|13097|398|
|rtl_6|A2_capacity_raw_cis|cis_light_raw|75.08 ± 2.98%|0.698219|658314|55.0|10317|398|
|rtl_6|A4_frequency_rings|far_freq_independent|24.22 ± 0.58%|0.192730|659370|71.4|9716|255|
|rtl_6|A4_frequency_rings|ib_freq_independent|68.52 ± 1.70%|0.673237|659370|111.5|8948|255|
|rtl_6|A4_frequency_rings|mid_freq_independent|28.92 ± 2.16%|0.232965|659370|61.0|11290|255|
|rtl_6|A4_frequency_rings|near_freq_independent|89.27 ± 2.25%|0.883240|659370|78.8|11291|255|
|rtl_6|A4_frequency_rings|oob_freq_independent|87.22 ± 5.95%|0.864596|659370|74.7|9398|255|
|rtl_6|A5_information_complementarity|capacity_ib_ib_concat|93.83 ± 2.67%|0.937760|1318730|204.7|6767|435|
|rtl_6|A5_information_complementarity|capacity_ib_ib_gate|93.92 ± 2.54%|0.939070|1317707|225.0|5635|435|
|rtl_6|A5_information_complementarity|ib_oob_concat|82.50 ± 1.39%|0.794494|1318730|161.3|9242|435|
|rtl_6|A5_information_complementarity|ib_oob_gate|76.50 ± 4.62%|0.721294|1317707|130.0|7562|435|
|rtl_6|A5_information_complementarity|single_ib|88.50 ± 9.01%|0.884151|659370|118.0|11031|255|
|rtl_6|A6_stability|full_stability|94.82 ± 2.38%|0.947277|659370|133.1|12156|255|
|rtl_6|A6_stability|ib_stability|88.80 ± 9.27%|0.886750|659370|132.8|11356|255|



## 七个研究问题



1. RTL设备34：P1三个seed的0/200失败已核对；重新读取缓存进行作者模型预测，检查与历史预测逐元素一致。分布、低能分母、原始CFO/SNR、训练拟合聚类和特征图见独立diagnostics_v1。可观测相关性不能证明硬件机制或标签错误。

2. Spec/CIS：A1控制几何、轻量卷积前端和训练预算；A2原始CIS控制容量。查看配对三seed和设备34召回；train_scalar结果不能与原始作者CIS混称纯容量实验。

3. OOB环带：五个频域RI频带均做三seed；保护带/相对能量/窗为首轮探索，全部候选保留。频带分类器统一坐标和CNN，但频带宽度不同。其与P1时域IB基线的差异不能解释为信息量顺序。原包Hann前窗只能检验有限记录泄漏敏感性，不建立真实硬件归因。

4. Concat/Gate：纠错与新错误同时报告，按三seed相对SingleIB及同结构IB+IB控制判断。[{"receiver": "n210_1", "method": "ib_oob_concat", "pairs": 6, "all_three_seeds_accuracy_and_f1_positive_vs_single_and_capacity": false}, {"receiver": "n210_1", "method": "ib_oob_gate", "pairs": 6, "all_three_seeds_accuracy_and_f1_positive_vs_single_and_capacity": false}, {"receiver": "rtl_6", "method": "ib_oob_concat", "pairs": 6, "all_three_seeds_accuracy_and_f1_positive_vs_single_and_capacity": false}, {"receiver": "rtl_6", "method": "ib_oob_gate", "pairs": 6, "all_three_seeds_accuracy_and_f1_positive_vs_single_and_capacity": false}]

5. Full/IB：A6同等2LR×2WD预算，30epoch和反事实patience7同时保存。早停后的救援收益属于训练政策改变；不同表示结果仍不足分离所有优化和信息影响。

6. HSTU/CrossAttn：预登记工程门槛未满足，暂不支持开发复杂融合。 P2-B/P3均未启动。

7. P3冻结：旧论文全部20台已研究使用，其中6台原盲测已公布结果，不能称研究从未使用。10/4/6仅区分本轮训练和选择角色；按型号前缀B200-1与N210-1均无同型号训练接收机，RTL4/8同RTL组。B200-mini不等于B200；本地实体/版本仍未核实。协议配置仍是候选，需要单独确认执行。



## 成本解释



Spec/CIS的pipeline吞吐基于预计算映射，不包含STFT构建。feature_construction_cost.json另行测量2000包HDF5读取与同时构建两种表示；P1在线STFT和本轮缓存输入的pipeline吞吐不能直接视作模型速度差。model_packets_per_second只测前向；训练吞吐含实际I/O和梯度记录。部分后续输入采用逐包bit一致的float32 IQ缓存，运行config记录input_storage；这不改变模型或批顺序。



## 限制



Fs/BW/SF与CFO/SNR单位未核实；P1未记录梯度；rank16容量控制改变线性头的秩约束；本轮未读Day3/4或六锁定接收机，未下载OSU。3seed不是接收机总体独立重复，也不是最终测试。P3 GRL/Ma代表性基线和P2-B代码只准备，未真实数据训练。

## 三seed的直接读数



n210_1 相同容量 Spec/CIS：cis_light_train_scalar Acc=99.22±0.35%，F1=0.9922，TX34召回=99.00%；spec_light_train_scalar Acc=98.73±0.25%，F1=0.9873，TX34召回=94.83%

n210_1 五频带：far_freq_independent=56.02±1.21%；ib_freq_independent=68.67±4.93%；mid_freq_independent=47.53±2.34%；near_freq_independent=86.30±1.95%；oob_freq_independent=72.63±1.08%

n210_1 ib_oob_concat相对single_ib逐seed净收益：20261012: +2.35pp（纠错144，新增错误97）；20261013: -0.70pp（纠错111，新增错误125）；20261014: -0.90pp（纠错82，新增错误100）

n210_1 ib_oob_concat相对capacity_ib_ib_concat逐seed净收益：20261012: +8.90pp（纠错277，新增错误99）；20261013: -1.25pp（纠错94，新增错误119）；20261014: -1.35pp（纠错87，新增错误114）

n210_1 ib_oob_gate相对single_ib逐seed净收益：20261012: +3.80pp（纠错164，新增错误88）；20261013: -2.15pp（纠错111，新增错误154）；20261014: -0.80pp（纠错86，新增错误102）

n210_1 ib_oob_gate相对capacity_ib_ib_gate逐seed净收益：20261012: +0.80pp（纠错107，新增错误91）；20261013: +0.55pp（纠错150，新增错误139）；20261014: +1.15pp（纠错124，新增错误101）

rtl_6 相同容量 Spec/CIS：cis_light_train_scalar Acc=73.52±3.40%，F1=0.6741，TX34召回=0.00%；spec_light_train_scalar Acc=67.43±1.03%，F1=0.5905，TX34召回=0.00%

rtl_6 五频带：far_freq_independent=24.22±0.58%；ib_freq_independent=68.52±1.70%；mid_freq_independent=28.92±2.16%；near_freq_independent=89.27±2.25%；oob_freq_independent=87.22±5.95%

rtl_6 ib_oob_concat相对single_ib逐seed净收益：20261012: -10.00pp（纠错118，新增错误318）；20261013: +5.10pp（纠错386，新增错误284）；20261014: -13.10pp（纠错92，新增错误354）

rtl_6 ib_oob_concat相对capacity_ib_ib_concat逐seed净收益：20261012: -12.05pp（纠错64，新增错误305）；20261013: -12.10pp（纠错84，新增错误326）；20261014: -9.85pp（纠错147，新增错误344）

rtl_6 ib_oob_gate相对single_ib逐seed净收益：20261012: -22.20pp（纠错113，新增错误557）；20261013: +1.60pp（纠错379，新增错误347）；20261014: -15.40pp（纠错100，新增错误408）

rtl_6 ib_oob_gate相对capacity_ib_ib_gate逐seed净收益：20261012: -19.85pp（纠错107，新增错误504）；20261013: -15.10pp（纠错83，新增错误385）；20261014: -17.30pp（纠错58，新增错误404）

RTL同预算优化后的Full−IB逐seed：20261012: +0.05pp，F1差+0.0012；20261013: +19.30pp，F1差+0.1943；20261014: -1.30pp，F1差-0.0139

Guard/窗/尺度敏感性仅首轮，需作为探索读取：oob_guard_0: 84.30%；oob_guard_25000: 87.95%；oob_ib_relative_projection_hann: 70.10%；oob_ib_relative_rect: 84.75%；oob_independent_projection_hann: 66.10%
