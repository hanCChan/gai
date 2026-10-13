# 有条件的强基线验证草案：未批准、未执行

D0 描述性 screen 通过，但不足证明强方法的剩余缺口。下一步合理目标是一次限定预算的**基线反证实验**；本文件不是新 Loss 实现许可。没有新算法命名或训练脚本。

最低可评估设计：两组源 RX 折 × 三个 seed，固定同一小 CNN、同一源池、前端、数据预算和总训练步数，分别比较 CE、CE+SupCon、ArcFace、RX×TX Group DRO、GRL、CE+HSIC IB surrogate 六个方法。共36个最终模型；IB采用20轮预训练+20轮微调，其余40轮，每个方法总步数匹配；按阶段计42个训练作业、共1440 model-epochs。未把已有单seed模型算作新增三seed公平基线。首个作业可承担计时/资源检查，包含在42作业内。类别均衡 CE 在当前均衡源池中与普通 CE 相同，不另算伪对照；不得因D0显示哪个TX难而调整权重或挑掉失败seed。

配对 seed/init、特征维度、优化器、训练包、增强和模型容量固定。RX×TX DRO 的组ID只来自源池，SupCon 使用跨源 RX 同 TX 的 positives。GRL 保持相同 backbone，head 成本另报。正则、温度、角度 margin、DRO参数及 checkpoint 规则必须在训练前登记；若搜索，每种方法同样预算，每个新增完整运行都计数。目标评价 RX 不能用于早停/参数选择。采用预冻结源训练内部 validation 与固定总轮数，外层留出 RX 只读评价。D0已使用的源池不能再宣称 pristine test。

这是比较**目标机制**的最小充分基线集合，不是完整复现所有论文。HSIC-on-CNN 是 IB surrogate 消融，**不能称为 IB-RFF FAN 原方法**；GRL小CNN也不是 Shen完整协同方案。最强直接竞争者必须另有 [IB-RFF 原实现](https://github.com/BeechburgPieStar/IB-RFF) 的 FAN 分类基线 vs FAN+IB 对照（至少2折×3seed×两阶段=12个训练阶段；是否能共享预训练取决于预算公平性），Ma跨RX Siamese/RIEI作结构不同的补充对照。ICC SupCon+Wasserstein barycenters、IB-RFF与RIFLE全文补齐是新颖性核验前提，不能用未复现的方法作已失败结论。

成本依据：历史四个658,314参数CNN的固定20轮训练作业 status.json 记录146.54–150.04秒，训练环境记录CUDA_VISIBLE_DEVICES=2。这是历史进程耗时，没有本轮 GPU 校准或硬件型号证明。按相同吞吐线性外推，1440轮约2.93–3.00小时；contrastive/DRO/IB与验证开销暂按1–3倍预算，**源池机制基线预估3–9 GPU小时**，不是实测保证。建议总预算上限9 GPU小时，首个已计入作业测吞吐后更新排期，超预算停止、保存全部结果，不静默删减方法。原FAN/其他结构与WiSig运行不能套用小CNN耗时，额外GPU时间**尚不可可靠量化**，须先核对实现参数/模型和计时，再单独报告，不能被隐藏在9小时里。

数据下载：Shen此阶段0 GB，读已存在源前端特征需在后续批准范围中列明；本D0仅分数，不涉及特征读取。WiSig另需ManyRx压缩1.2 GB及解压空间，未经批准不下载。

评价保留所有TX/RX：Accuracy、macro F1、min-TX Recall、90对confusion和尺度归一化margin，使用源参考风险构造固定预算风险名单，再评价外层RX；公平地报告准确率与最差类权衡。三seed×两模型折为主要训练重复，RX/硬件族分开；数据包不得作硬件重复。不根据先前TX36异常设特殊规则。

若 margin/SupCon/DRO/IB 在相同预算下消除类别损害或源风险不再迁移，停止新方法路线。若仍有跨seed反复出现、对独立RX有预测性的剩余困难，才提交单独算法PoC。新机制须区别于 RX×TX DRO、一般 hard-negative mining 与 IB 压缩；D0 不保证存在这种区别。批准少量GPU验证可以合理，但本轮按用户要求完成D0即停止。
