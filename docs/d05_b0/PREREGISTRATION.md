# D0.5/B0 冻结规则

原D0保持只读。新问题是有向 signed Δconfusion，而非任一专家的绝对混淆。D0.5先完成；B0独立获准8次训练，不因D0.5结果追加或替换。新seed沿用G1已登记的20261013/14；原20261012保留。外层五RX只监测，不选epoch/学习率/增强；B0固定第20轮。数据是历史探索源池，不是盲测。

完整数值定义、基线、screen与预算见 registration.json。90有向边等口径；同折实际共享一对Full/Strict CNN，不能当作五次独立训练重复。signed mean在参考RX平均后直接排名；probe正向质量 max(Δ,0)单独评估，净变化/正向质量/改善质量同时报告。Top10边界ties分摊，保证10条预算。没有正向损害时指标记null，Spearman常量记null，不强行记0。Δ是类别条件概率差，不是同包“新错”的转移计数，禁止混称。

对照：Recall下降÷9、Full类别难度÷9、Full绝对混淆、Strict绝对混淆、Strict/Full各自factorized混淆之差、参考正向损害质量的类×去向factorization、uniform。positive-mass-factorized先对每参考RX的Δ取正，再平均；这与signed mean不同，单列反证。主screen只用于描述性解释，不能据此宣称新Loss必要。所有方法/控制全报。

B0：原小CNN/缓存/训练步数/Adam/clip/seed顺序；两折五源训练/五源评价，batch64、workers2、lr0.0003、WD0、20轮，无增强、rank16。训练缓存只读，scaler只取训练五RX的manifest统计。保存配置、初始化SHA、优化器、final.pt、日志、逐epoch状态和全部源评价logits。新增20份模型-RX预测每seed，共80,000最终输出；末轮已有validation输出直接保存，避免重复推理。每轮监测评价源RX只为诊断，不选择模型。

严重新增类别损害定义为同RX×TX Strict−Full Recall≤−10pp。每seed报整体差值、全部严重单元、最差类、90对变化。进入B1候选的描述性规则：三个seed总体Strict均有正收益；至少一个RX×TX在≥2/3seed严重下降；这些重复严重单元跨≥2物理RX，并在非collapse模型配对仍出现。没有要求每seed/每RX/每边都重现。collapse flag为非有限/运行失败/不满20轮或最终train Acc≤20%；保留，不换seed。若固定矩阵不完整，不给GO。只有三seed，不能证明分布意义稳定性。

总新增训练8次，四V100各两个作业，无额外smoke训练；每作业最多450秒，累计保守reserved GPU占用≤1小时（包含初始化/读取/validation）。超时作为失败，不重新运行。若资源异常先保存状态，不静默扩预算。训练参数不随中途结果改变。

不读取OSU、四开发/六锁定RX、Day3/4、原始IQ，不下载新大型数据、不运行B1/B2/新方法。新用户授权覆盖旧AGENTS/D0停机限制，仅到本轮D0.5+B0及B1计划。
