# Receiver-held-out R0–R1 完整报告

2026-10-13，北京时间。已完成冻结34源内搜索+51正式拟合，零失败/重试/seed替换；三个配对seed，两数据集，全部checkpoint只用源域选择。停止在R1，R2只有设计，最终确认测试未运行。

累计reserved拟合成本4.290339 V100 GPUh；含数值预检、冻结评价及20秒进程启动余量的保守总额4.328058 GPUh，低于用户后续批准的7小时上限。原6小时不足保守完整矩阵估计约6.22，12个首作业后用户明确提高到7并保持原矩阵，批准记录公开。

## 主要发现

WiSig官方骨干、内部RX且未见开发日期：CE为48.01%±10.44pp；RX+日期MLDG为51.55%±17.20pp，平均提高3.54pp，但仅2/3个seed改善，配对变化范围[-1.83,11.23]pp。RX下10%分位平均提高3.63pp，平均RX最差TX Recall仍仅7.40%。这个配方有均值收益但没有解决seed稳定性，不能只引用平均增益。
Shen历史源池两折CV：CE为85.16%，SWAD为85.81%，平均+0.65pp且3/3个seed同向；RX下10%分位+1.01pp、平均RX最差TX Recall+3.38pp。SWAD是本轮Shen最强已实现方法，收益应归于已有训练配方。其余方法在这个有限固定配方下没有平均优势，不能推断它们普遍无效。
固定残差容量对照在源域验证约95.6%，但内部RX未见日期只有37.19%，相对官方CE的未知RX平均变化-10.82pp。该架构有4.87倍参数且显著更高成本，其源内改善没有转化为此场景泛化改善；仅一个固定容量对照不足定位原因。
这些结果保留了跨RX稳定性的研发问题，但没有确证两个数据集共享同一因果机制。正式官方CNN有11个普通方法checkpoint处于99/100epoch；有限horizon与固定meta-alpha/beta、probe步长和train/eval模式差异需要先核查，不能把尚未充分收敛或配方受限归结为必须发明新模块。本轮按登记horizon结束，未追加续训。

## 三个seed的RX等权结果

| 数据/骨干 | 评价场景 | 方法 | Accuracy均值±seed SD (%) | RX下10%分位 (%) | Macro-F1 (%) | 平均RX最差TX Recall (%) |
|---|---|---|---:|---:|---:|---:|
| shen/rank16 | source_pool_outer_RX | ce | 85.16±0.65 | 79.20 | 84.81 | 60.45 |
| shen/rank16 | source_pool_outer_RX | groupdro | 82.76±1.11 | 77.11 | 82.47 | 53.78 |
| shen/rank16 | source_pool_outer_RX | mldg_rx | 83.75±0.95 | 79.55 | 83.44 | 49.57 |
| shen/rank16 | source_pool_outer_RX | pcgrad | 84.14±0.64 | 78.89 | 83.61 | 55.40 |
| shen/rank16 | source_pool_outer_RX | swad | 85.81±0.51 | 80.21 | 85.43 | 63.83 |
| wisig/official | internal_seen_date | ce | 51.90±7.53 | 40.48 | 49.85 | 6.78 |
| wisig/official | internal_unseen_date | ce | 48.01±10.44 | 37.31 | 45.98 | 5.02 |
| wisig/official | source_seen_date | ce | 89.82±0.73 | 78.54 | 89.69 | 68.33 |
| wisig/official | source_unseen_date | ce | 76.58±2.41 | 69.18 | 74.81 | 24.20 |
| wisig/official | internal_seen_date | groupdro | 48.74±13.37 | 39.08 | 46.99 | 7.40 |
| wisig/official | internal_unseen_date | groupdro | 46.52±17.46 | 36.79 | 45.28 | 8.32 |
| wisig/official | source_seen_date | groupdro | 89.78±0.83 | 82.97 | 89.82 | 77.02 |
| wisig/official | source_unseen_date | groupdro | 73.92±4.81 | 66.78 | 72.31 | 24.74 |
| wisig/official | internal_seen_date | mldg_rx | 51.81±12.64 | 40.34 | 49.93 | 5.93 |
| wisig/official | internal_unseen_date | mldg_rx | 49.08±15.07 | 35.75 | 47.32 | 6.37 |
| wisig/official | source_seen_date | mldg_rx | 92.91±0.70 | 81.81 | 92.92 | 79.32 |
| wisig/official | source_unseen_date | mldg_rx | 79.14±3.67 | 70.64 | 77.39 | 27.74 |
| wisig/official | internal_seen_date | mldg_rx_date | 53.78±13.23 | 44.07 | 50.91 | 6.58 |
| wisig/official | internal_unseen_date | mldg_rx_date | 51.55±17.20 | 40.93 | 49.14 | 7.40 |
| wisig/official | source_seen_date | mldg_rx_date | 92.96±0.59 | 81.40 | 92.97 | 79.55 |
| wisig/official | source_unseen_date | mldg_rx_date | 81.20±2.88 | 73.45 | 79.64 | 30.64 |
| wisig/official | internal_seen_date | pcgrad | 52.89±12.29 | 45.10 | 50.15 | 5.97 |
| wisig/official | internal_unseen_date | pcgrad | 49.51±14.28 | 39.64 | 47.21 | 4.88 |
| wisig/official | source_seen_date | pcgrad | 92.43±0.48 | 80.95 | 92.48 | 76.13 |
| wisig/official | source_unseen_date | pcgrad | 79.03±3.74 | 71.89 | 77.28 | 26.40 |
| wisig/official | internal_seen_date | swad | 51.28±6.76 | 39.82 | 49.45 | 7.44 |
| wisig/official | internal_unseen_date | swad | 46.91±9.78 | 36.89 | 45.20 | 5.90 |
| wisig/official | source_seen_date | swad | 88.60±1.23 | 77.24 | 88.39 | 65.39 |
| wisig/official | source_unseen_date | swad | 75.68±2.52 | 67.67 | 73.87 | 24.42 |
| wisig/resnet | internal_seen_date | ce | 42.60±2.09 | 34.91 | 38.38 | 0.78 |
| wisig/resnet | internal_unseen_date | ce | 37.19±2.83 | 25.27 | 33.80 | 0.08 |
| wisig/resnet | source_seen_date | ce | 95.59±0.12 | 86.53 | 95.73 | 86.22 |
| wisig/resnet | source_unseen_date | ce | 77.22±1.41 | 66.96 | 74.15 | 9.30 |

## CE配对变化

| 数据/骨干 | 场景 | 方法 | 平均ΔAccuracy (pp) | 三seed Δ范围 (pp) | 正改善seed | 下尾Δ (pp) | 最差TX均值Δ (pp) |
|---|---|---|---:|---:|---:|---:|---:|
| shen/rank16 | source_pool_outer_RX | groupdro | -2.39 | [-3.94,-1.06] | 0/3 | -2.10 | -6.67 |
| shen/rank16 | source_pool_outer_RX | mldg_rx | -1.41 | [-2.75,0.05] | 1/3 | 0.35 | -10.88 |
| shen/rank16 | source_pool_outer_RX | pcgrad | -1.01 | [-1.64,-0.64] | 0/3 | -0.31 | -5.05 |
| shen/rank16 | source_pool_outer_RX | swad | 0.65 | [0.09,1.11] | 3/3 | 1.01 | 3.38 |
| wisig/official | internal_seen_date | groupdro | -3.17 | [-8.21,3.24] | 1/3 | -1.40 | 0.63 |
| wisig/official | internal_unseen_date | groupdro | -1.49 | [-8.10,5.88] | 1/3 | -0.51 | 3.30 |
| wisig/official | source_seen_date | groupdro | -0.05 | [-0.26,0.16] | 1/3 | 4.43 | 8.69 |
| wisig/official | source_unseen_date | groupdro | -2.66 | [-5.15,-0.19] | 0/3 | -2.40 | 0.54 |
| wisig/official | internal_seen_date | mldg_rx | -0.09 | [-4.19,5.68] | 1/3 | -0.14 | -0.85 |
| wisig/official | internal_unseen_date | mldg_rx | 1.07 | [-2.54,6.39] | 1/3 | -1.55 | 1.35 |
| wisig/official | source_seen_date | mldg_rx | 3.08 | [2.35,3.69] | 3/3 | 3.27 | 10.98 |
| wisig/official | source_unseen_date | mldg_rx | 2.56 | [1.01,3.75] | 3/3 | 1.47 | 3.54 |
| wisig/official | internal_seen_date | mldg_rx_date | 1.88 | [-1.75,8.79] | 1/3 | 3.59 | -0.20 |
| wisig/official | internal_unseen_date | mldg_rx_date | 3.54 | [-1.83,11.23] | 2/3 | 3.63 | 2.38 |
| wisig/official | source_seen_date | mldg_rx_date | 3.13 | [2.65,3.58] | 3/3 | 2.86 | 11.22 |
| wisig/official | source_unseen_date | mldg_rx_date | 4.62 | [4.23,5.15] | 3/3 | 4.27 | 6.44 |
| wisig/official | internal_seen_date | pcgrad | 0.98 | [-3.13,6.20] | 1/3 | 4.62 | -0.81 |
| wisig/official | internal_unseen_date | pcgrad | 1.50 | [-1.12,6.30] | 1/3 | 2.34 | -0.13 |
| wisig/official | source_seen_date | pcgrad | 2.61 | [2.39,2.89] | 3/3 | 2.40 | 7.80 |
| wisig/official | source_unseen_date | pcgrad | 2.45 | [0.65,3.57] | 3/3 | 2.72 | 2.20 |
| wisig/official | internal_seen_date | swad | -0.63 | [-1.59,-0.13] | 0/3 | -0.66 | 0.67 |
| wisig/official | internal_unseen_date | swad | -1.10 | [-1.72,-0.04] | 0/3 | -0.41 | 0.88 |
| wisig/official | source_seen_date | swad | -1.23 | [-1.66,-0.64] | 0/3 | -1.30 | -2.95 |
| wisig/official | source_unseen_date | swad | -0.90 | [-1.88,-0.04] | 0/3 | -1.51 | 0.21 |

## 解释范围与强基线

不从某一seed的结果直接结束研究问题。本次全部矩阵完成后比较。上表最优方法来自固定完整矩阵的描述，不用于挑选checkpoint；不能把内部开发成绩当最终盲测。WiSig剩余八RX、第四日期、等化版持续封存。Shen两折源池已大量观察，仅交叉验证/辅助确认。
WiSig改为十类闭集，所以旧八类W0精度不可直接对比。Shen本轮每类160拟合/40源内验证，标准化只用160，旧B1每类200拟合不可直接同协议对比。源内完整学习曲线、搜索及所有候选保留。有限两LR而固定alpha/beta/eta的配方可能限制已有方法表现；不得把它们在该配方下的欠佳当作机制普遍失效。
source_validation_vs_development.csv将每个冻结模型的源训练/验证分数、checkpoint与各部署场景分数逐seed/折配对列出，便于直接评估源域分数是否代表开发泛化；不对混合不同方法、RX与三个seed的点给出伪显著性或因果相关结论。
官方SWAD正式作业9个，检测到收敛9个，未检测收敛而按作者末段回退0个；回退不能称成功找到平坦谷。固定100/20epoch的收敛质量请结合训练/验证曲线解释，没有训练预算触发续训。
CE与SWAD同LR同seed的原始训练目标和源验证轨迹必须相同，平均操作不影响CE更新。二阶MLDG相对CE成本显著更高；按相同数据/horizon比较，并同时报告成本，不称计算等价。残差CNN的CE容量对照独立列出；容量收益不归因于元学习。PCGrad只采用两个随机RX分组任务，未跑全部逐RX任务的额外变体。
资源调度校正：初始控制器把慢的WiSig残差速度暂代尚未计时Shen，产生超预算替代估算，行政暂停记录完整保留。先对原矩阵每种Shen机制计时，再按各自速度及正式RX数量重估，才继续未启动的原作业；没有拟合失败或重试。完整解释见RESERVATION_ACCOUNTING.md及三份调度记录。
统计单位为RX/日期/seed。无packet级p值、无三seed假设性显著性结论。源RX和开发日期均有限，数据包按历史位置切分并非独立采集session。未等化输入同时包含信道/日期等因素，不把RX差异解释为纯硬件因果。
所有Macro-F1针对真值支持类，缺TX的Recall为NA；预测到缺TX仍作为错误。RX均值先等权日期再等权接收机，Shen合并两折十RX后再算分位，不平均两折的分位数。主指标平均RX最差TX Recall先对每RX×TX在有样本的日期等权平均Recall，再取该RX支持TX中的最小值，最后平均RX；逐日期最差Recall的平均另存，二者不混用。每RX/日期完整覆盖计数、混淆矩阵、每TX支持数和Recall均公开。
class_index/tx0–9的身份映射见identity_labels.csv（Shen对应TX31–40），开发日期映射见date_labels.csv；Shen date=-1明确表示未知日期。复现需重建既有输入缓存或具备相同SHA输入，干净checkout本身不含数据和权重。

## 训练过程风险

固定频率的首个平衡批次探针记录eval-mode的inner SGD候选与实际outer update，训练RNG保持不变。每RX×TX只有两包，探针为随机源RX模拟，不能当作未来未见RX的预测器，更不能证明因果机制。实际outer包含留RX元梯度，故其局部损失不是完全未参与外层优化的测试损失。RX+日期版本的探针是共同RX协议，未另做日期风险探针。
update_risk_per_seed.csv给出负梯度夹角、源训练有益且留RX有害的比例，inner/outer的RX损失正增幅与类别尾部风险分别保存。所有结果只能证明该小批次确定性局部诊断现象，独立包重测、后续轨迹预测、相同训练与推理模式的检验尚未确证。
完整训练风险与所选checkpoint之前轨迹分别列于update_risk_per_seed.csv和update_risk_selected_prefix.csv；若所选checkpoint早于第一次探针，该模型不具备prefix探针证据。SWAD采用平均窗口终点作为prefix边界，它不是对平均权重本身的风险测量。不能把未进入所选模型的后续更新直接当作部署风险原因。

## 研发决定

完整R0/R1基线和源模拟风险已交付。优先保留实测最强已有方法；本轮没有新算法效果或创新性结论。R2只提交数学机制候选和区分MLDG/PCGrad/Fishr/Group DRO的必要实验；模块C须在A+B确实改善总体而仍有可重复身份尾部退化之后才考虑。R2训练、额外系数/架构搜索或最终测试需要后续用户指令。
先行论文及准确的实现区别见docs/receiver_r01/PRIOR_WORK_MATRIX.md；条件R2见docs/receiver_r01/R2_DESIGN.md。历史G1/G2/VIRP/EISR/P0/C0/D0/B1/M0/receiver transform/W0/W1停止状态和文件保持不变。

## 可复核证据

公开source_10class_coverage.csv、data/numerical_preflight.json、selection.frozen.json、paired_initialization_sampling.csv、85作业成本/完整学习曲线/源内搜索、逐RX/TX/date/seed统计、混淆counts、配对变化、局部更新风险和derived-only evidence.zip。原IQ、输入feature、预测logit缓存和权重不上传。输入SHA与公开文件SHA目录说明完整保留。
