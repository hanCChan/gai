# D0.5 增量损害审计

决定：**EXPLORATORY_INCREMENTAL_DAMAGE_SIGNAL**，仅历史seed20261012的源池探索。原D0未改标准/结果。signed Δ=Strict−Full有向混淆概率差；正向质量取max(Δ,0)，净变化与改善质量另报。不是同包新错转移率，Full→Strict包含前端、scaler和分别训练的CNN，不是GRL/IB因果证据。

同一对Full/Strict CNN的五个评价RX，四RX形成参考，probe只做评价。8个预测对照、90有向边、全部十RX，两fold与硬件控制均公开，不特别选择TX35/36。

| fold | median_signed_spearman | median_positive_mass_capture |
| --- | --- | --- |
| 0 | 0.41584 | 0.33333 |
| 1 | 0.34315 | 0.93789 |

signed mean主预测在两折Spearman为0.41584/0.34315，Top10捕获正向质量33.33%/93.79%。均匀预期11.11%，但两折差异很大。

逐probe配对capture差值的中位数（pp前须乘100）：

| fold | median_capture_gain_over_recall_drop_only | median_capture_gain_over_class_difficulty_full | median_capture_gain_over_factorized_delta | median_capture_gain_over_absolute_full_confusion | median_capture_gain_over_absolute_strict_confusion |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.16322 | 0.21488 | 0.03226 | -0.35484 | -0.48019 |
| 1 | 0.05521 | 0.23268 | 0.06211 | 0.01242 | 0.00621 |

比Recall下降/Full类别难度/Δfactorized的中位增益两折为正，故冻结的有限screen通过。但fold0 **比Full绝对混淆低35.48pp、比Strict绝对混淆低48.02pp**，fold1仅多1.24/0.62pp。不能宣称Δ预测全面优于原错误结构。所有基线全报，不把较弱类难度对照当作最强竞争者。

| fold | control | median_signed_spearman | median_positive_mass_capture | median_capture_gain_over_uniform |
| --- | --- | --- | --- | --- |
| 0 | same_paired_cnn_loro | 0.41584 | 0.33333 | 0.22222 |
| 0 | different_family_reference | 0.34903 | 0.39632 | 0.28521 |
| 0 | cross_paired_cnn_transfer | 0.24896 | 0.03226 | -0.07885 |
| 1 | same_paired_cnn_loro | 0.34315 | 0.93789 | 0.82678 |
| 1 | different_family_reference | 0.34315 | 0.88999 | 0.77888 |
| 1 | cross_paired_cnn_transfer | 0.22259 | 0.00578 | -0.10533 |

跨另一对CNN折的signed Δ参考捕获率仅3.23%/0.58%，两折均**低于uniform**；虽然signed Spearman仍正，排序对正向损害的实际预算检索失败。这是关键反证：风险与训练模型/拟合折有关，共享CNN内跨RX可预测性不能替代跨拟合模型迁移。这项控制同时更换参考RX组与模型，参考CNN本身曾用probe所在RX组训练；它是历史源池的混杂压力诊断，不是所有模型都未见probe的独立验证，不能把下降唯一归因为模型参数。不同硬件参考控制不是只用七RTL互相预测，但单例硬件和少参考RX使外推仍有限。

结论：新增损害有内部信息，但不支持把“专门预测鲁棒化新增伤害”的优越性当作算法核心。保留一般source-predictable class-conditional failure问题，先用B0/B1检查尾部风险与普通方法是否足够。没有独立新硬件、没有packet级p值、90边不作90个独立硬件重复。

独立复核240个profile、21,600条向量，直接重建confusion、参考集、baseline、signed correlation、正质量及ties通过；[verification](independent_verification.json)。完整900有向Δ、240评价、全部向量存于本轮evidence.zip。定义/源码封存见configs/d05_b0；初始seal与启动修正后的seal均保留。

数值敏感性补充：[12位舍入后的ties核查](numeric_tie_sensitivity.json)，不改冻结主指标或通过标准。240个组合中个别capture最大变化6.45pp，主fold的capture中位数及基线增益方向均不变；signed Spearman约0.41579/0.34292。不能依赖浮点微差来解释个别小幅次要差异。
