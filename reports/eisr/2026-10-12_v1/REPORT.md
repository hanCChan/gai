# EISR-v1 · E0/E1/E2 CPU 筛选

**决策：STOP_E2。**

这是历史源数据上的两级接收机分组内部筛选，不是 CNN 十折留一或独立外部盲测。VIRP-v1 负结果保持封存。

注册 SHA256：`02a256a777d99464a715834606efff78cbd033d420be9751e0e378cd120ba79c`。先冻结后推理，无新增 CNN 训练。四个真实 epoch20 `final.pt` 已经通过 metadata、训练历史、逐 epoch 日志、源 scaler 和原代码哈希核对；未使用 best checkpoint 预测。

两组专家分别在五源 RX 训练，并仅对另外五源 RX 重新进行 CPU 推理。Full/Strict 各 20,000 次预测逐原始行 ID 配对；两个一级折复用专家，因此十 RX 结果相关，不按包计算显著性。

## E1：互补性与 Accuracy oracle

十 RX 等权 Full Accuracy=68.3700%；Strict=82.9900%；Hard oracle=91.4000%；Convex soft oracle=91.5300%。

Hard oracle 相对每 RX 最强单专家的平均空间为 **8.4100 pp**；预登记要求 ≥3 pp；通过。Soft oracle 不参与关口，也不称为 Macro-F1 上界。

|RX|Full Acc|Strict Acc|Hard oracle|Soft oracle|相对 RX 最强单专家空间|Full 可挽救包|Full 引入错误包|
|---|---:|---:|---:|---:|---:|---:|---:|
|b200_mini_1|72.55%|84.15%|94.65%|94.65%|10.50 pp|210|442|
|pluto_1|69.95%|90.10%|93.30%|93.50%|3.20 pp|64|467|
|rtl_2|59.35%|81.35%|88.35%|88.50%|7.00 pp|140|580|
|rtl_5|57.75%|79.80%|88.80%|89.20%|9.00 pp|180|621|
|rtl_7|76.10%|90.10%|96.95%|97.10%|6.85 pp|137|417|
|b210_1|80.05%|83.75%|96.75%|96.80%|13.00 pp|260|334|
|rtl_1|67.95%|80.75%|91.25%|91.35%|10.50 pp|210|466|
|rtl_3|62.45%|77.45%|85.60%|85.60%|8.15 pp|163|463|
|rtl_6|64.75%|81.60%|88.90%|89.05%|7.30 pp|146|483|
|rtl_9|72.80%|80.85%|89.45%|89.55%|8.60 pp|172|333|

## E2：四 RX 拟合、一 RX 评价

每次仅使用同一对固定专家未见的四台 RX 训练一个线性选择器及校准简单基线；第五台 RX 不用于温度、融合权重、基线名称或选择器参数的选择。基线选择记录在每 RX 的 `selection.frozen.json` 中。

|RX|训练四 RX 选定基线|基线 F1|选择器 F1|ΔF1|基线最差 TX Recall|选择器最差 TX Recall|
|---|---|---:|---:|---:|---:|---:|
|b200_mini_1|source_weight_selected_fusion|0.8694|0.8737|+0.0042|63.00%|64.00%|
|pluto_1|source_weight_selected_fusion|0.8892|0.8834|-0.0058|58.00%|56.50%|
|rtl_2|source_weight_selected_fusion|0.8120|0.8128|+0.0009|60.50%|63.50%|
|rtl_5|source_weight_selected_fusion|0.8125|0.8085|-0.0040|60.00%|58.50%|
|rtl_7|source_weight_selected_fusion|0.9207|0.9202|-0.0005|83.00%|83.50%|
|b210_1|source_weight_selected_fusion|0.8568|0.8606|+0.0038|21.00%|22.50%|
|rtl_1|temperature_calibrated_average|0.8124|0.8045|-0.0079|23.50%|14.50%|
|rtl_3|temperature_calibrated_average|0.7638|0.7703|+0.0065|22.50%|21.50%|
|rtl_6|temperature_calibrated_average|0.8082|0.8027|-0.0055|4.00%|2.00%|
|rtl_9|temperature_calibrated_average|0.7797|0.7800|+0.0003|11.00%|4.00%|

平均 ΔMacro-F1=-0.000783（要求≥.01）；最小 RX Δ=-0.007900（要求≥0）；等权 RX 后最差 TX Recall 增益=-0.020500（要求≥.05）；严重 RX×TX 退化 0/100（要求0）。

E2 所有预注册基线都报告，但关口只对四训练 RX 选定者比较，不在第五 RX 挑最好的基线。Full/Strict 单专家同时报告。

选择器只利用预测置信度、Top1−Top2 margin、熵、分歧及概率分布差异，不输入真实 TX、RX 身份、真实类别 margin、原型或人工 TX36 规则。学习目标只来自四训练 RX 的一对一正确性分歧。

## 证据边界与复现

第20轮权重排除了本轮最佳 checkpoint 选择回流；此前源数据上的模型与超参数开发偏差仍存在。E1 用了十 RX 标签决定是否执行 E2，所以 E2 属于同批数据的阶段性机制研究，不能称完全独立确认。原 checkpoint 不包含 optimizer state/入口脚本哈希；现有代码版本、匹配日志、checkpoint 元数据与 source SHA 相互佐证，不声称具有密码学执行证明。

文献/官方代码核对见 `docs/eisr/E0_LITERATURE_AUDIT.md`。EISR 尚无已证明的新机制贡献；普通置信度选择、模型堆叠、概率融合和 MoE 均不能改名当创新。

冻结协议：`configs/eisr/v1_registration.json`、`.seal.json`；E0：`E0_checkpoint_provenance.json`、`E0_predictions/*`；E1：逐 RX/TX、逐包四种正确性与解析 soft 可行区间；E2：每 RX 的冻结选择记录、线性模型参数、全基线指标、逐包预测与混淆矩阵。

复现需要原始已核验的 G1 权重/缓存及新的空输出目录：`CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/keyan/venvs/shen-stage1/bin/python scripts/run_eisr_v1.py --output outputs/eisr/v1_reproduction`。

当前注册不启动 E3/E4、新 CNN、HSTU/CrossAttn、Shen 四开发或六历史锁定 RX、OSU 新评价。任何后续方法变化需要新协议，并保留当前结果。

## 独立复核和决策解释

独立核验注册/代码封存、40,000 条概率转换、20,000 个包的 Hard oracle 汇总、全部 18,306 个 soft 可行见证，并用线性规划对预先固定的 200 个包另算 soft 可行性；不使用主程序的区间求解实现。四个已有模型另对200条记录重推理，最大 logits 差异 2.86e-6，Top1 一致。

另从冻结的模型系数和概率记录重建全部 20,000 个 Hard Selector 决策，独立重算十次训练 RX 的基线种类及权重选择；四项 E2 判定与主程序完全一致。完整内容见 `independent_verification.json`。

源内选定基线的平均 Macro-F1=0.832456，Hard Selector=0.831673，差−0.000783，即 **−0.0783 pp**。五台RX正收益、五台负收益，要求十台均不下降。

十 RX 等权的最差类别是 TX35：基线 Recall49.45%，选择器47.40%，下降2.05 pp；要求提升至少5 pp。100个RX×TX单元没有超过10 pp的严重退化警报。这只通过“无严重退化”关口，其余三项均未通过。

逐包相对基线挽救286包、新增错误278包，净多正确8包（+0.04 pp Accuracy），并不等于Macro-F1或最差TX得到保护。不能用另一个指标改写预登记的停止判定。

Hard oracle空间8.41 pp、Full可挽救1,682个Strict错误包，证明固定专家存在互补空间；并未证明Strict造成了可辨识的硬件身份擦除，两个独立专家/不同scaler仍不能构成纯频谱删除因果对照。当前线性实例没有提供相对公平简单融合的稳定增量收益，不推断所有选择方法均不可能。

## 非预登记的描述性补充

在测试RX的“恰一个专家正确”分歧包上，风险分数的等权RX AUROC为0.7451，范围0.5607–0.8768。此项是结果后的描述性排名检查，不是通过门槛、统计显著性检验或允许重新调阈值的依据。它显示一定排序信息，而目前固定决策并未把它变成稳定的分类及身份保护收益。

补充分组统计最初误用了全体RX记录，已修正为逐RX过滤，并用E1逐RX四格表交叉核对样本数；原错误诊断标记为 invalid_grouping，未用于方法、基线或主关口。独立核验报告的初次JSON输出也有 NumPy bool 序列化错误，已修复并完整重跑复核；没有重跑专家或选择器实验、没有改变封存算法。

## 首次推理前的来源核验修订

初版注册1.0及程序保存在 `docs/eisr/pre_inference_v1_0`；在任何新概率输出之前，修订到1.1来源核验。g1_data.py的原SHA从git5661a32精确恢复，所有原有函数AST相同，仅新增角色检查。批量source_SHA中的旧P2-A runner不被G1调用，其旧文本未恢复，已明确披露；没有把其当前源码说成原执行一致。检查仍包括真实第20轮、完整配置、20行history与20行日志、固定optimizer控制流、真实源scaler及模型源代码。所有科学定义、门槛、模型和数据不变。

依据用户授权，当前版本状态 **STOP_E2**。不增加隐藏层、原型、风险损失、TX36人工规则，不进入E3/E4，也不读取开发/锁定/OSU RX寻找参数。
