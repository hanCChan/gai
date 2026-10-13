# 评价汇总的明确数学定义

本说明在开发/外层评价读取之前写入，只澄清日期与类别汇总顺序，不改训练、模型选择或作业矩阵。

每RX×日期先用10×10整数混淆计数计算Accuracy、支持真值类的Macro-F1、每支持TX的Recall。没有样本的TX为NA，不设为0；预测到无真值支持的TX仍算该样本分类错误，并降低其真实TX的Recall。

场景内Accuracy与Macro-F1先等权平均同RX各日期，再等权平均RX。RX间标准差（sample SD，ddof1）和10%分位（NumPy默认线性分位插值）均作用于先汇总日期后的RX分数。Shen两折合并十个外层RX以后再计算这些统计量，不平均两折分位。

主要身份尾部指标：对每个RX/TX在其有样本的场景日期等权平均Recall，得到R_(r,t)；对该RX至少一个日期有样本的TX取`min_t R_(r,t)`，再对RX平均或取RX的10%分位。单日期场景和无日期的Shen退化为普通平均RX最差TX Recall。不能用“先取每日期最差TX，再平均日期”替代该指标，因为最差TX可能随日期改变；后一种日期尾部描述单独保存为mean_RX_date_worst_TX_recall。

combined_per_RX.csv公开每模型/seed/场景的RX日期汇总及两种身份尾部描述。per_RX_date.csv保留逐日期统计。完整per_RX_TX_date.csv及混淆计数收入derived-only evidence.zip的tables/，并给出每TX支持数、正确数及Recall，足以复核所有汇总顺序。

Seed波动使用三个配对seed的sample SD；CE变化逐seed、同RX/日期/骨干配对，不把多个RX/日期/packet当作独立训练seed。RX×TX Recall下降至少5pp的单元计数只是描述，不作为自动训练停止或方法删除的门槛。

registration.json中的嵌套wisig保留W0/W1缓存来源与划分记录；其中旧八类encoder_tx、两个pilot及旧质量门槛只属于历史窗口。本轮顶层矩阵和EXECUTION_PLAN.md定义新的十类闭集、34/51作业；budget_amendment.json将本轮上限改为7GPUh。历史字段不作为本轮方法定义或拟合许可。
