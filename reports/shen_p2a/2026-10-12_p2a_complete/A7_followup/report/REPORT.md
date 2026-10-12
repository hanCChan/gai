# 严格IB生成Spec/CIS：诊断后的追加对照

本轮16次训练（8候选+8新增seed），与原始108分开；追加动机是RTL34在Spec/CIS的带外STFT位置发生较大跨日期变化。投影原始FFT的|f|≤62.5kHz后才重新STFT；STFT窗口仍可生成边缘能量，但信息源只有原始IB。保持52×126几何、rank16容量、2LR预算和train-only标量缩放。所有结果均为Day2开发证据。

|Receiver|Method|Acc mean ± sample SD|Macro-F1|TX34 recall|Parameters|
|---|---|---:|---:|---:|---:|
|n210_1|spec_strict_ib|94.38 ± 4.16%|0.9432|92.67%|658314|
|n210_1|cis_strict_ib|95.73 ± 3.62%|0.9565|87.67%|658314|
|rtl_6|spec_strict_ib|91.82 ± 1.43%|0.9173|81.00%|658314|
|rtl_6|cis_strict_ib|94.00 ± 1.95%|0.9389|87.33%|658314|

逐包配对比较分别冻结的同预算Full-STFT表示。改善可支持在该训练配方下去除原始OOB信息的功能性效果；它不唯一定位物理来源，也不证明收集标签或接收机硬件原因。追加实验在观察数据后提出，不能冒称原先预登记的盲测。没有启动P2-B/P3。

## 相对Full-STFT的逐seed净收益

n210_1 spec_strict_ib seed20261012: -1.70pp，纠错22 / 新增错误56，F1差-0.0172。
n210_1 spec_strict_ib seed20261014: -2.50pp，纠错19 / 新增错误69，F1差-0.0254。
n210_1 spec_strict_ib seed20261013: -8.85pp，纠错30 / 新增错误207，F1差-0.0897。
n210_1 cis_strict_ib seed20261012: -7.30pp，纠错20 / 新增错误166，F1差-0.0756。
n210_1 cis_strict_ib seed20261013: -1.80pp，纠错8 / 新增错误44，F1差-0.0180。
n210_1 cis_strict_ib seed20261014: -1.35pp，纠错15 / 新增错误42，F1差-0.0134。
rtl_6 spec_strict_ib seed20261012: +24.05pp，纠错544 / 新增错误63，F1差+0.3199。
rtl_6 spec_strict_ib seed20261013: +24.05pp，纠错515 / 新增错误34，F1差+0.3255。
rtl_6 spec_strict_ib seed20261014: +25.05pp，纠错510 / 新增错误9，F1差+0.3349。
rtl_6 cis_strict_ib seed20261012: +23.40pp，纠错493 / 新增错误25，F1差+0.2920。
rtl_6 cis_strict_ib seed20261014: +20.55pp，纠错465 / 新增错误54，F1差+0.2671。
rtl_6 cis_strict_ib seed20261013: +17.50pp，纠错440 / 新增错误90，F1差+0.2355。

解释约束：RTL的CIS严格IB选择lr=.001，完整STFT CIS选择lr=.0003；它们的配对差异同时包含配置筛选影响。RTL Spec与两种N210表示的选定lr均相同（.0003）。两种输入的train-only标量均值/方差分别估计，严格FFT投影也改变有限长度波形；这些结果支持本配方的功能差异，尚不能单独归因物理OOB硬件指纹。
