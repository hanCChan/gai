# G1 shen 首批跨RX强基线

固定四种强基线与三seed。全部预测、完整混淆、逐类别margin与纠错均独立重算。Shen用源RX两折选epoch后冻结，OSU用20固定epoch；没有用目标RX标签选择模型或scaler。

|Source|Evaluation domain|Baseline|Accuracy mean ± seed SD|Macro-F1|
|---|---|---|---:|---:|
|10_training_RX|shen_b200_1_test|full_iq|62.33 ± 23.54%|0.5647|
|10_training_RX|shen_b200_1_test|strict_iq|31.78 ± 37.73%|0.2493|
|10_training_RX|shen_b200_1_test|full_cis|69.75 ± 8.20%|0.6815|
|10_training_RX|shen_b200_1_test|strict_cis|83.60 ± 3.60%|0.8041|
|10_training_RX|shen_b200_mini_1_test|full_iq|65.65 ± 22.38%|0.5950|
|10_training_RX|shen_b200_mini_1_test|strict_iq|34.78 ± 42.93%|0.2815|
|10_training_RX|shen_b200_mini_1_test|full_cis|83.40 ± 11.47%|0.8239|
|10_training_RX|shen_b200_mini_1_test|strict_cis|95.28 ± 2.42%|0.9519|
|10_training_RX|shen_b210_1_test|full_iq|65.65 ± 24.39%|0.5996|
|10_training_RX|shen_b210_1_test|strict_iq|35.22 ± 43.68%|0.2878|
|10_training_RX|shen_b210_1_test|full_cis|89.77 ± 9.91%|0.8910|
|10_training_RX|shen_b210_1_test|strict_cis|96.30 ± 3.82%|0.9598|
|10_training_RX|shen_n210_1_day1|full_iq|64.87 ± 22.96%|0.5919|
|10_training_RX|shen_n210_1_day1|strict_iq|35.80 ± 44.69%|0.2956|
|10_training_RX|shen_n210_1_day1|full_cis|84.55 ± 9.77%|0.8228|
|10_training_RX|shen_n210_1_day1|strict_cis|96.40 ± 3.13%|0.9619|
|10_training_RX|shen_n210_1_day2|full_iq|63.63 ± 27.74%|0.5842|
|10_training_RX|shen_n210_1_day2|strict_iq|38.15 ± 48.76%|0.3264|
|10_training_RX|shen_n210_1_day2|full_cis|87.20 ± 7.20%|0.8528|
|10_training_RX|shen_n210_1_day2|strict_cis|97.30 ± 1.78%|0.9725|
|10_training_RX|shen_n210_1_test|full_iq|66.13 ± 29.03%|0.6092|
|10_training_RX|shen_n210_1_test|strict_iq|39.77 ± 51.56%|0.3431|
|10_training_RX|shen_n210_1_test|full_cis|84.15 ± 7.01%|0.8202|
|10_training_RX|shen_n210_1_test|strict_cis|97.58 ± 2.34%|0.9748|
|10_training_RX|shen_pluto_1_test|full_iq|62.22 ± 25.69%|0.5721|
|10_training_RX|shen_pluto_1_test|strict_iq|35.93 ± 44.92%|0.2971|
|10_training_RX|shen_pluto_1_test|full_cis|84.85 ± 12.04%|0.8387|
|10_training_RX|shen_pluto_1_test|strict_cis|93.50 ± 7.99%|0.9283|
|10_training_RX|shen_rtl_1_test|full_iq|64.73 ± 27.81%|0.6090|
|10_training_RX|shen_rtl_1_test|strict_iq|36.07 ± 45.15%|0.3051|
|10_training_RX|shen_rtl_1_test|full_cis|83.88 ± 5.66%|0.8331|
|10_training_RX|shen_rtl_1_test|strict_cis|90.08 ± 5.04%|0.8970|
|10_training_RX|shen_rtl_2_test|full_iq|65.20 ± 28.11%|0.6095|
|10_training_RX|shen_rtl_2_test|strict_iq|37.83 ± 48.21%|0.3236|
|10_training_RX|shen_rtl_2_test|full_cis|73.07 ± 3.79%|0.7241|
|10_training_RX|shen_rtl_2_test|strict_cis|85.93 ± 4.50%|0.8550|
|10_training_RX|shen_rtl_3_test|full_iq|65.15 ± 29.94%|0.6039|
|10_training_RX|shen_rtl_3_test|strict_iq|38.35 ± 49.10%|0.3288|
|10_training_RX|shen_rtl_3_test|full_cis|76.55 ± 4.03%|0.7557|
|10_training_RX|shen_rtl_3_test|strict_cis|87.03 ± 5.06%|0.8662|
|10_training_RX|shen_rtl_4_test|full_iq|62.85 ± 27.84%|0.5818|
|10_training_RX|shen_rtl_4_test|strict_iq|35.85 ± 44.77%|0.3025|
|10_training_RX|shen_rtl_4_test|full_cis|62.63 ± 1.96%|0.6070|
|10_training_RX|shen_rtl_4_test|strict_cis|85.10 ± 5.09%|0.8456|
|10_training_RX|shen_rtl_5_test|full_iq|63.30 ± 28.53%|0.5921|
|10_training_RX|shen_rtl_5_test|strict_iq|37.10 ± 46.94%|0.3163|
|10_training_RX|shen_rtl_5_test|full_cis|72.17 ± 5.18%|0.7109|
|10_training_RX|shen_rtl_5_test|strict_cis|83.55 ± 4.90%|0.8306|
|10_training_RX|shen_rtl_6_day1|full_iq|66.48 ± 22.97%|0.6177|
|10_training_RX|shen_rtl_6_day1|strict_iq|33.95 ± 41.48%|0.2729|
|10_training_RX|shen_rtl_6_day1|full_cis|76.92 ± 3.33%|0.7545|
|10_training_RX|shen_rtl_6_day1|strict_cis|85.72 ± 9.61%|0.8451|
|10_training_RX|shen_rtl_6_day2|full_iq|66.42 ± 27.64%|0.6178|
|10_training_RX|shen_rtl_6_day2|strict_iq|38.98 ± 50.20%|0.3352|
|10_training_RX|shen_rtl_6_day2|full_cis|87.18 ± 3.27%|0.8649|
|10_training_RX|shen_rtl_6_day2|strict_cis|93.12 ± 6.24%|0.9292|
|10_training_RX|shen_rtl_6_test|full_iq|66.18 ± 28.26%|0.6163|
|10_training_RX|shen_rtl_6_test|strict_iq|38.50 ± 49.36%|0.3301|
|10_training_RX|shen_rtl_6_test|full_cis|94.32 ± 3.48%|0.9425|
|10_training_RX|shen_rtl_6_test|strict_cis|95.43 ± 4.15%|0.9509|
|10_training_RX|shen_rtl_7_test|full_iq|68.27 ± 24.13%|0.6339|
|10_training_RX|shen_rtl_7_test|strict_iq|38.82 ± 49.91%|0.3335|
|10_training_RX|shen_rtl_7_test|full_cis|93.12 ± 6.55%|0.9289|
|10_training_RX|shen_rtl_7_test|strict_cis|95.43 ± 4.30%|0.9505|
|10_training_RX|shen_rtl_8_test|full_iq|65.62 ± 28.99%|0.6135|
|10_training_RX|shen_rtl_8_test|strict_iq|39.05 ± 50.32%|0.3359|
|10_training_RX|shen_rtl_8_test|full_cis|91.53 ± 6.67%|0.9111|
|10_training_RX|shen_rtl_8_test|strict_cis|93.92 ± 5.69%|0.9345|
|10_training_RX|shen_rtl_9_test|full_iq|67.02 ± 26.26%|0.6224|
|10_training_RX|shen_rtl_9_test|strict_iq|39.15 ± 50.49%|0.3369|
|10_training_RX|shen_rtl_9_test|full_cis|85.35 ± 2.82%|0.8472|
|10_training_RX|shen_rtl_9_test|strict_cis|95.82 ± 4.17%|0.9547|

root Shen采集日期/session未知，不能称同日cross-RX；Day1/2 drift另列。B200/N210为训练未含同型号的外推，RTL4/8是同型号实例转移，不能只看四RX均值。OSU是独立采集团队；该subset是否曾参与旧方法选择仍需执行历史核实，当前只作开发复现，不称研究史全新盲测。连续周期窗口不等于严格同步前导，所有结果均为开发机制证据。3seed只描述训练随机性，两OSU接收机不等于接收机总体。

G1需要联合Shen与OSU趋势判定。本单数据集报告不宣布通用频带规律；G2源域预测关系、G3新方法/GRL/对比/FAN比较、G4跨体制外部验证尚未完成。未启动旧P2-B/P3。
