# 作者级强方法源码/输入与复现成本核查

核查日期2026-10-13；[固定作者commit、文件SHA](author_code_provenance.json)。本轮只读取小源码，**作者训练/冒烟/数据下载均0**。B1小CNN机制对照与B2论文级复现分表。

## Shen Receiver-Agnostic

[作者仓库](https://github.com/gxhen/receiverAgnosticRFFI) 的 deep_learning_models.py 为TensorFlow/Keras实现。adv_net输入固定NHWC(52,126,1)，残差CNN后完整Dense512、L2特征、TX head及GRL RX head；本B0是低秩rank16 Dense替换后的容量控制CNN，**不是完整作者网络**。作者main.py使用SGD lr1e−3 momentum0.9，最多500epoch，val_loss早停patience20与lr factor0.2/patience10；TX/RX CE权重均1，训练包内部90/10划分。源码还含需要新RX标注包的fine_tune和多RX协同推理；不与source-only主表混合。

静态核查发现多RXvalid_generator调用的rx_label参数切片仍用`[:num_train_samples]`而data/tx_label使用尾段。其最终影响需结合generator验证；不能直接修改后宣称逐字复现。该脚本还含固定本地h5权重/路径及场景分支，不能一键在我们的source cache上启动。应先比对生成器样本、布局、padding/flatten顺序、Dense容量、标签与split，再报告任何适配差异。说明来自 [固定源码](https://github.com/gxhen/receiverAgnosticRFFI/blob/8910103a3fe953a746f964459a37917c5ff1ce03/main.py)。

复现成本：完整作者Dense在13×31×64输入展开维度下单层约13.2M权重（实际尺寸须核对作者padding/stride），容量远大于当前658kCNN，不用B0吞吐代替成本。需单独TensorFlow依赖/版本和V100兼容检查、一次有预算的smoke计时、源内验证规则；本轮未执行，GPU小时**待测**，不预填。若改成PyTorch移植，需要数值/初始化对齐证明并标明port。作者许可证CC BY-NC-SA4.0，不将源码去署名粘贴成自创模型。

## IB-RFF FAN+IB

[作者仓库](https://github.com/BeechburgPieStar/IB-RFF) README提到WiSig和LoRa，但main.py的dataset choices仅ManyRx/ManySig，load_data.py也只有两者。run_all.sh按这两种WiSig子集四RX折运行。FAN constructor可接seq_len参数，**当前入口明确硬编码seq_len256、in_chans2**；不能把支持改构造参数等同于已支持LoRa8192或CIS二维图。

公开loader按每TX前100包取样、逐包功率归一化，从按RX/day/is_eq排列的pickle读取，再transpose。注释的维度说明与下游索引需结合实物核验：preprocessing按(N,2,L)语义访问I/Q，FAN的SAF/patch模块同样要求通道/长度布局；仅源码不能确定自定义pickle实形，不能猜transpose后就是正确输入。必须用真实格式对照、有限值/功率/标签/包ID/shape断言审计。Shen频谱CIS不能直接喂进该256点IQ入口；截取LoRa到256点或重做前端都是明确方法适配，不能冒称原协议复现。

main.py从源RX做70/30包级train/validation划分；测试RX分离。Adam，batch128，epochs上限200；val_loss选best、ReduceLR/early-stop。先alpha0预训练，再加载其权重做IB微调；损失NLL + alpha*hsic_normalized_cca(hidden,input)。HSIC使用核矩阵/矩阵逆，不是简单pairwise最差margin。run_all.sh给每数据集/折/CR或CRD不同alpha，**这些alpha的选择过程未在脚本证明，不能假定其与我们的外层未知RX不泄漏**。源内可见性/早停/alpha选择需单独冻结。

原作者复现建议未来先在WiSig ManyRx建立相同RX/day/前100包的可追踪输入，区分“原结果复现”“同协议类风险诊断”“我们的强机制比较”。Four-round ManyRx的RX32、TX10、两训练day、CR/CRD差异均以原main.py为准，不能把官方ManyRx名义200包误当作者实际100包预算。ManyRx压缩1.2GB，本轮未下载；四天重复同硬件/重叠子集不独立。

成本：未安装或执行FAN，patch_size16/32、DSCNN/HSIC、两阶段/200轮/早停会改变吞吐，额外GPU成本**待数据适配与单次smoke实测**。不给伪精确小时数，不把CNN+HSIC叫FAN+IB。公开代码没有完整LoRa入口，无法单凭README就声称LoRa复现完成。作者代码MIT；数据/权重另外遵循其许可，不上传。

## 直接先行工作与全文边界

[RIEI作者全文](https://arxiv.org/html/2411.03636)已有TX/RX解耦及Separable Condition，[D0矩阵](../d0_class_pair/LITERATURE_MATRIX.md)保留；它不是只管平均而忽视身份的空白起点。[ICC2026正式DOI](https://doi.org/10.1109/ICC59461.2026.11586949)题名为SupCon+Parametric Wasserstein Barycenters，本轮再次题名/DOI检索与出版社访问，仍未取得方法全文；不猜测其class-conditional目标、margin或worst-class保障。IB与RIFLE全文细节同样未完整排清。

[Group DRO](https://arxiv.org/abs/1911.08731)、[SupCon](https://arxiv.org/abs/2004.11362)、[DANN/GRL](https://arxiv.org/abs/1505.07818)是B1直接已知目标。本轮源码审计/统计结果未证明它们在公平预算下失败，未证明足以支持独立算法创新。任何下一轮新方法须先接受它们的实际反证。
