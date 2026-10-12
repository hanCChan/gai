# G1 osu 首批跨RX强基线

固定四种强基线与三seed。全部预测、完整混淆、逐类别margin与纠错均独立重算。Shen用源RX两折选epoch后冻结，OSU用20固定epoch；没有用目标RX标签选择模型或scaler。

|Source|Evaluation domain|Baseline|Accuracy mean ± seed SD|Macro-F1|
|---|---|---|---:|---:|
|osu_RX1|osu_RX1|full_iq|84.12 ± 12.18%|0.8085|
|osu_RX1|osu_RX1|strict_iq|85.22 ± 11.26%|0.8350|
|osu_RX1|osu_RX1|full_cis|100.00 ± 0.00%|1.0000|
|osu_RX1|osu_RX1|strict_cis|100.00 ± 0.00%|1.0000|
|osu_RX1|osu_RX2|full_iq|6.30 ± 1.94%|0.0350|
|osu_RX1|osu_RX2|strict_iq|9.11 ± 1.29%|0.0443|
|osu_RX1|osu_RX2|full_cis|10.11 ± 0.46%|0.0709|
|osu_RX1|osu_RX2|strict_cis|11.76 ± 0.72%|0.0799|
|osu_RX2|osu_RX1|full_iq|7.40 ± 2.21%|0.0449|
|osu_RX2|osu_RX1|strict_iq|5.36 ± 1.63%|0.0339|
|osu_RX2|osu_RX1|full_cis|10.75 ± 0.83%|0.0515|
|osu_RX2|osu_RX1|strict_cis|13.61 ± 0.72%|0.0727|
|osu_RX2|osu_RX2|full_iq|86.41 ± 9.31%|0.8426|
|osu_RX2|osu_RX2|strict_iq|85.94 ± 9.49%|0.8504|
|osu_RX2|osu_RX2|full_cis|100.00 ± 0.00%|1.0000|
|osu_RX2|osu_RX2|strict_cis|100.00 ± 0.00%|1.0000|

root Shen采集日期/session未知，不能称同日cross-RX；Day1/2 drift另列。B200/N210为训练未含同型号的外推，RTL4/8是同型号实例转移，不能只看四RX均值。OSU是独立采集团队；该subset是否曾参与旧方法选择仍需执行历史核实，当前只作开发复现，不称研究史全新盲测。连续周期窗口不等于严格同步前导，所有结果均为开发机制证据。3seed只描述训练随机性，两OSU接收机不等于接收机总体。

G1需要联合Shen与OSU趋势判定。本单数据集报告不宣布通用频带规律；G2源域预测关系、G3新方法/GRL/对比/FAN比较、G4跨体制外部验证尚未完成。未启动旧P2-B/P3。
