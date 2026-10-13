# B1标准基线：获准32作业，2 reserved GPUh

用户已明确批准上一分支的B1审核计划（commit2b89877）。本轮只实施CE、RX×TX Group DRO、GRL、SupCon标准方法；不开发新Loss/Gate/network，不自动运行B2/ArcFace。旧D0/D0.5/B0及所有停止结论保持只读。

已批准矩阵文件原字节保留，本轮registration链接其SHA。16个内部LR搜索（2fold×4方法×2lr），固定seed20261015/10轮，内部4RX拟合、1RX验证，batch80、40groups各2包，一轮100步、每包一次。源内部固定末轮macro-F1选择lr，精确分数完全相同取较低lr；其余权重/温度/η不搜索。外层5RX不参与任何选择。所有搜索完整后冻结选择文件，再正式16次：2fold×4方法×2seed20261013/14，5RX拟合/5RX评价，batch100、50groups、一轮100步、20轮2000步。CE也重新训练。

训练全方法同一基础rank16 CNN（658314推理参数），初始化payload逐seed配对；缓存Strict CIS同样本、训练RX-only scaler、无新增增强、Adam/WD0/clip5。每stage/fold/seed预先保存一个batch schedule，所有方法/lr共享，metadata含SHA、每包每epoch一次、每group每batch2包核验。GRL base初始化后附训练RX head：512→128→4/5，梯度反转λ.1，RX CE权重1；推理丢弃head，其额外66180/66309参数及训练计算单报。SupCon在原512维L2特征做一视图标准同TX正对、排除self，τ=.1/baseτ=.1、权重.1，无投影或特殊对权重。GroupDRO q初均匀、η.01、每步乘exp(η组均值CE)并归一化，再加权组CE；log-space只作数值等价稳定实现，无特殊正则/难度调整。

目标数学/梯度、CNN既有forward/初始化一致性、schedule覆盖/重复性使用合成输入CPU检查，不增加拟合模型或GPU smoke。first已计入的search CE(lr1e−4)单GPU计时；以固定公式 first_seconds·[1+1.8·(15+16·2·1.25)] 外推，如超过7200秒或首作业失败则保存停止，不能静默减矩阵/调预算。搜索每job145秒、正式295秒上限，32job最坏7040秒加进程清理保留160秒余量；超时失败/异常日志保留，不自动重试/替换seed。

所有正式权重固定20轮后才评价外层源池；保存权重/优化器/日志/全logits/labels/rows，以及全部RX×TX、90有向pair/confusion/margin。比较每方法与同fold/seed、同输入Strict CE的paired变化；B0的Full/Strict仅作历史上下文，不能混入损失归因。按RX等权，分seed/fold/硬件族，保留所有类与不利结果，不把包或90边当独立硬件重复。报告平均Accuracy/macro-F1、最差RX×TX Recall、10pp严重paired损害、所有margin。描述性Pareto/共同改善判断不自动批准新算法；两seed与已用源池不足证明强方法普遍失效或独立硬件泛化。

只读取十源RX缓存Strict CIS/索引/manifest；无需原IQ、OSU、开发/锁定RX、Day3/4、新数据下载。数据/权重/原预测留服务器，发表派生完整表、真实日志、全部来源/源码/模型SHA。新分支experiment/b1-standard-baselines，每轮上传并回读全树、源码、报告、决策及SHA，完成B1停止。
