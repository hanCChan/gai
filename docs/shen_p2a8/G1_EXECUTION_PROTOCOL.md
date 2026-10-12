# 最新执行目标：G1 可迁移机制验证

最新用户方案USER_SUPPLIED_GENERALIZATION_UPDATE.md覆盖旧版关于局部Notch大批训练的优先级。M0已完成；M1的30次训练在启动前撤下，冻结注册与部分派生缓存保留，outputs/shen_p2a8/m1_superseded.json记录原因。G1未通过前不进入G2/G3，不启动新方法或复杂融合。

第一批研究是Shen多RX和OSU两RX的Full IQ、Strict IB IQ、Full CIS、Strict IB CIS四个强基线。统一Adam lr=.0003/WD=0/batch64/clip5，首轮最多20epoch、三seed20261012/13/14；对每类每训练RX使用预先确定200包，不根据模型结果抽样，所有候选和失败保留。这是控制规模的首次开发检验，未使用的训练包保留，不能宣称完整数据预算复现。

Shen冻结既有10训练/4开发/6本轮锁定。训练内部按receiver两折分组（交替索引5/5），只用source标签选择epoch：首seed两折各模型20epoch，取两折最佳epoch的向上取整平均，最终三seed全10RX训练同一固定epoch。最终四开发RX不参与epoch、scaler、超参选择。按RTL同型号实例及B200/N210型号外推分别报告，并保留14RX逐TX结果。root多RX文件真实日期未知，不能称同日期跨RX。额外drift Day1/Day2按其实际文件角色单列，不按row跨RX配对。Day3/4与六锁定RX不访问。

OSU只下载官方Diff_Receivers_Setup_Indoor_SameTx的50个IQ capture，共8GB，使用已有bundle_checksums逐一SHA核验，保留元数据。RX1→RX2和RX2→RX1对称检验，各自重新训练25类分类器，不与Shen标签对应。两RX只有两个物理环境，不能提供接收机总体稳定性证据。源RX不足以做receiver分组CV，固定20epoch，不用目标性能选模型。

OSU连续IQ不能伪装为Shen packet。本轮前端仅提取互不重叠8192点窗口，依据冻结lag1024平均周期相干度≥0.8筛选周期信号窗口，每TX/RX沿时间均匀取最多200个；不声称已严格同步前导。全窗、截断IB与CIS使用完全同一批窗口。每class至少20个合格窗口，否则该数据分支停止并报告，不能看目标效果调整阈值。采样率/载频用SigMF核实，BW/SF无本地声明时标记假设，不把OSU ±62.5k解释为已核实最佳物理带内。该筛选本身的选择偏差和payload混杂必须作为开发局限，后续投稿需同步前导或匹配的稳健capture协议。

Scaler仅来自当前fold训练RX或最终源RX；Full/Strict CIS分别拟合train-only标量，输入RMS归一化同一操作，不能把差分单独当物理机制。IQ采用相同PacketCNN；CIS采用相同rank16CNN，跨数据集分别设置类别数。评估固定checkpoint，保存全部样本ID、类别概率/margin、完整混淆矩阵、每RX Accuracy/Macro-F1、每TX Recall/F1、worstRX与均值离散程度、参数和实际前向时间。数据集之间不得直接平均不同类别数的F1。

无需训练的机制诊断：对预定义等宽归一化频率bins/nominal band，计算源域类间辨识统计与同TX跨RX分布变化，目标仅用于报告趋势；不从目标挑选新band。发射机条件方差不当作已验证可靠性分数，不据此立即设计新损失。

G1判定要求Shen多个开发RX及独立OSU采集均有一致的辨识度/稳定性冲突或相应错误变化。只改善RTL6、另一型号负迁移、或OSU不能复现时不宣布通用规律。首轮强基线不能单独证明G2的源域预测关系；只有G1支持后单独预登记G2。G3必须比较GRL/对比学习/FAN+HSIC等已有方法；G4扩展跨日期与WiSig，WiSig256点需另建前端，不照搬LoRa FFT。
