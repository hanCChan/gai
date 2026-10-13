# 跨RX/日期风险转移报告

**探索性分析，C2已被此前研发查看，不是新盲测。** 使用80份真实Adam快照，没有训练或新的候选方向。WiSig全部6个有向2/2互斥RX划分，Shen两个1/1划分；日期0->1、1->0两方向。每个选择仅使用判断侧C1，目标侧C2仅标记后果。

目标RX只是不参与本次A/B候选和C1判断；这些源RX可能在此前步骤已参与模型训练。该审计比同RX包重复更强，但仍没有验证整个训练过程从未见过的RX，更没有验证最终部署RX。

| 数据/方法 | 层次 | 快照/轮换观测 | 符号一致 | 有害判断精确率 | 召回 | 平衡准确率 | 风险AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| shen/mldg_rx | cross_RX | 20/40 | 0.900 | 0.571 | 0.800 | 0.857 | 0.869 |
| shen/mldg_rx | same_RX_individual | 20/40 | 0.950 | 0.714 | 1.000 | 0.971 | 0.937 |
| shen/mldg_rx | same_RX_pool | 20/20 | 0.900 | 1.000 | 1.000 | 1.000 | 0.944 |
| wisig/mldg_rx_date | cross_RX | 20/120 | 0.775 | 0.200 | 0.208 | 0.500 | 0.638 |
| wisig/mldg_rx_date | cross_RX_and_date | 20/240 | 0.733 | 0.156 | 0.143 | 0.506 | 0.603 |
| wisig/mldg_rx_date | cross_date_same_RX | 20/40 | 0.900 | 0.333 | 0.250 | 0.597 | 0.854 |
| wisig/mldg_rx_date | same_RX_individual | 20/80 | 0.887 | 0.850 | 0.895 | 0.923 | 0.950 |
| wisig/mldg_rx_date | same_RX_pool | 20/20 | 0.950 | 0.750 | 0.750 | 0.844 | 0.844 |

WiSig日期MLDG跨RX的精确率0.20等于目标有害率0.20，冻结有害判断的平衡准确率0.50；跨RX+日期约0.506。连续排序有部分信息，但早期AUC约0.882、短训练末段约0.556，缺乏稳定检测优势。Shen MLDG跨RX有较强探索性信号，但只有两条10轮轨迹（两个fold、同seed），每步目标仅一台RX。不能由此宣称跨数据集通用检测。

误选定义：接受非零倍率而目标C2均值为正（raw），以及超过目标描述性噪声幅度（above_noise）；错误完全拒绝要求full目标均值低于负噪声界。较差RX为目标RX损失变化最大值。零更新不能仅凭风险零被视为训练最优，mean_scale和拒绝率一并公开。

跨RX的行为结果：

dataset       method     kind            policy  split_observations  mean_scale  complete_rejection_rate  accepted_rate  target_mean_delta  target_worst_RX_delta  false_accept_raw_rate  false_accept_above_noise_rate  false_reject_beneficial_rate  target_any_RX_harm_rate
   shen      mldg_rx cross_RX       always_full                  40    1.000000                 0.000000       1.000000          -0.007308              -0.007308               0.475000                       0.125000                           0.0                 0.125000
   shen      mldg_rx cross_RX        fixed_half                  40    0.500000                 0.000000       1.000000          -0.008428              -0.008428               0.200000                       0.125000                           0.0                 0.125000
   shen      mldg_rx cross_RX mean_backtracking                  40    0.856250                 0.100000       0.900000          -0.009722              -0.009722               0.325000                       0.025000                           0.0                 0.025000
   shen      mldg_rx cross_RX    multi_RX_guard                  40    0.856250                 0.100000       0.900000          -0.009722              -0.009722               0.325000                       0.025000                           0.0                 0.025000
  wisig mldg_rx_date cross_RX       always_full                 120    1.000000                 0.000000       1.000000           0.000031               0.002501               0.441667                       0.200000                           0.0                 0.400000
  wisig mldg_rx_date cross_RX        fixed_half                 120    0.500000                 0.000000       1.000000          -0.000290               0.000948               0.408333                       0.175000                           0.0                 0.333333
  wisig mldg_rx_date cross_RX mean_backtracking                 120    0.812500                 0.166667       0.833333          -0.000201               0.001793               0.325000                       0.158333                           0.0                 0.325000
  wisig mldg_rx_date cross_RX    multi_RX_guard                 120    0.804167                 0.175000       0.825000          -0.000219               0.001771               0.316667                       0.158333                           0.0                 0.308333

多RX与普通回溯跨RX不同选择仅4/320个轮换观测（其中含CE诊断；对应真实快照更少）；同RX整组重现原6/80差异。WiSig MLDG跨RX两规则的above-noise误接受率同为0.1583；固定半倍率的平均/较差目标损失下降更大，但其更新量更小，不能直接推出分类更好。Shen1/1两规则必然相同。

所有RX/日期/phase子组、支持数、风险排序和逐步实际C2损失均公开。轮换有共享与互补结构，不能把120个划分当120个训练seed。trajectory_dependence中的轮换均值塌缩为原同队列共同变化，不能当固定不同RX预测证明；fixed_pair_support/ranks进一步列出真正固定RX对的重复支持，通常稀少。first-difference和lag1均为描述，没有80独立采集的置信区间或packet级p值。

**判断：MODIFY。** 发现部分源风险排序信息，不支持当前多RX保护比均值回溯稳定地提供额外跨RX检测。保留研究问题，但不凭此自动启动18或54项训练，不针对C2搜索阈值。
