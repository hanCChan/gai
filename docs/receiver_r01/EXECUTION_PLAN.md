# Receiver-held-out R0–R1 执行计划

2026-10-13，北京时间。用户新方案及“理解方案，更新计划……严格根据方案执行”授权本轮。旧 B1、M0、receiver transform、W0/W1 均保持停止，历史代码、报告和证据不改写。本分支从已远端核验的 WiSig W0/W1 `e64c5a3` 建独立 worktree，继承全部历史文件。

研究问题固定为有限源 RX 下未知 RX 识别的平均、尾部与跨 seed 稳定性。R0/R1 不宣称原创算法。R2 本轮只提交机制设计；最终测试和 R2 拟合另批。最多 **6 reserved V100 GPUh**，包括进程初始化、数据装载、合成数值检查、拟合和 GPU 评价的整段占用，非 CUDA kernel 活动时长。首个完整搜索作业作计时校准，逐种机制复核，剩余矩阵按实测加 50% 余量；不足时保持矩阵完整并申请预算，不挑 seed 截断。

## 数据协议

WiSig：复用既有受审计的未等化开发缓存。14源 RX、10内部 RX、8最终 RX 名单和日期划分完全继承 W0/W1 的 metadata 规则。先核查源 RX 十 TX 的三日期有效覆盖。闭集十类替代旧八类变换共享性任务；不把旧八类结果当作新协议公平对照。缓存 provenance 和第三方来源的官方字节一致性未证实限制继续披露。

源日期0/1：每 RX×日期×TX 位置0–119训练，120–159源内 checkpoint，160–199独立包评价。日期2只作开发跨日期评价。内部10RX只在全部模型及 checkpoint 冻结后统一评价，按 RX×日期报告覆盖及现存真值类 Macro-F1，缺样 TX Recall=NA；不称盲测。最终8RX、日期3、等化数据不数值访问，无新数据下载。

WiSig 内层按排序源 RX 的每第4台取4台留域验证，其余10台训练；留域验证位置0–159、日期0/1。LR及内层 checkpoint 只看这4台源 RX 等权 CE。正式14源 RX 的 checkpoint 用自身位置120–159等权 CE，无内部或日期2选参。

Shen：只用十源 RX 原 Strict CIS/labels/rows/manifest 缓存，两折五训练/五评价及各折原内层四训练/一验证 RX 不变。为给所有方法共同的源内 checkpoint 和 SWAD 验证反馈，本轮每拟合 RX 每 TX 先160缓存包训练，余40源内验证；原 B1 使用200训练包，因此本轮重新配对 CE，不能直接拿 B1 作同数据对照。内层留出 RX 使用其全部200包/类。标准化只拟合实际160训练包，不含40验证包或外层 RX；日期缺失，不做 Shen 日期元任务。外层源池已大量观察，仅作历史源池交叉验证和辅助确认。

## 冻结方法与矩阵

共同 Adam、无增强、Glorot 初始化、同 backbone/seed 初始化 hash、每 RX×日期×TX 两包平衡批次、每包每 epoch 恰一次，clip norm5。WiSig 官方十分类 CNN 保持原结构、Dense L2=1e-4、Dropout=.5、Adam epsilon1e-7；Shen 原 rank16 CNN、Adam epsilon1e-8。WiSig 单个固定残差 CNN 仅 CE 容量对照，不搜索架构、不将容量收益归因于训练机制。

R0四方法：CE；CE+官方 SWAD LossValley（逐 outer step 稠密平均、每 epoch 分段，n_converge3/n_tolerance6/tolerance_ratio.3）；标准全二阶 MLDG；继承 RX×TX Group DRO eta=.01。R1额外标准 PCGrad（随机源 RX 二分任务，冲突对称投影，平均梯度），及 WiSig RX+日期 MLDG。

MLDG：随机不相交源 RX 元训练/留出；内层一步普通 SGD alpha=.01，仅分类 CE；外层 `(L_train + L_held(theta'))/2 + 共用L2`，beta=1。全二阶，不将 DomainBed 的一阶近似实现当作精确二阶复现。RX+日期每 outer step 两个互补元任务，日期顺序按注册 RNG 决定，A/d0→B/d1和B/d0→A/d1，覆盖该批所有 RX×日期样本，平均两目标。它属于已有思想的任务组织，不是原创。

两个 LR 候选，WiSig .0003/.0005；Shen .0001/.0003。其余系数全部固定，无额外 alpha/beta/eta 搜索。搜索 seed20261016；正式成对 seed20261013/14/15，不替换、不重试。WiSig 内层固定50epoch、正式100epoch；Shen固定10/20epoch。无结果触发缩短或训练重启。普通方法选源内 CE 最低 checkpoint；SWAD 用公开 LossValley 及其未收敛末段回退，不将回退伪装为已成功 SWAD。源内选择全部候选保留。

共34搜索+51正式=85拟合作业：WiSig官方6方法×2LR和×3seed；残差CE×2LR和×3seed；Shen5方法×2折×2LR和×2折×3seed。首作业包含在矩阵，不另加试训。数值测试只用合成张量，不研究拟合。成本按方法实际时长与相同 epoch/数据预算共同报告，不声称二阶训练和 CE kernel 开销相同。

## R1诊断与交付

固定 WiSig每5epoch、Shen每epoch 的首个平衡批次：用本次未进入元训练的源 RX，记录确定性 eval-mode 分类损失、梯度夹角、内层 SGD 后和实际 outer update 后每 RX×TX 损失变化。诊断不反馈优化、不消费训练 RNG；不把小批次局部伤害当作未知目标因果或跨任务预测证明。诊断采样单独 RNG，所有方法同频率。

全部模型与选择冻结后，输出每 RX/TX/日期/seed confusion、Accuracy、现存类Macro-F1、RX等权mean、10%分位、RX标准差、各RX最差TX Recall及分布、seed波动、CE配对差异、训练/推理成本及配对初始化/批次核验。统计分析单位为RX/日期/seed，不用packet伪造显著性；三seed和三开发日期不能证明总体稳健因果。不存在单个seed或3pp自动停止研究线规则。

交付报告、论文差异矩阵、完整学习曲线和源内搜索、过程风险分析、条件R2设计、源输入与公开证据SHA索引。只上传代码、协议、派生指标和证据，不上传原IQ、特征数组、权重或私稿。上传独立分支并核查远端完整树与报告/决定/证据SHA，停止在R1。
