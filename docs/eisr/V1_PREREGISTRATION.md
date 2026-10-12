# EISR-v1：E0/E1/E2 预登记

用户新方案及补充规则分别保存在 `USER_PLAN_20261012.txt`、`USER_RULES_20261012.txt`。新协议和程序在生成第20轮新预测之前封存 SHA256。VIRP-v1 的注册与负结果不修改、不重启。

## 授权与证据范围

仅执行 E0/E1/E2，全部使用 CPU；不训练新 CNN，不使用 best checkpoint 的旧预测；不访问四台 Shen 开发 RX、六台历史锁定 RX 或 OSU 数据。两个一级 CNN 折各用五台源 RX 训练，另外五台生成预测。第二级在同一五台未见 RX 内四 RX 拟合、一 RX 评价，交换一级折共十次评价，不称 CNN 十折留一。

这是历史源数据上的阶段性机制筛选。原有结构、学习率及表示的开发史仍需披露。E1 用这些 RX 的标签决定进入 E2，E2 不能当完全独立的新确认；同折五个评价 RX 共享固定专家，所有平均指标按 RX 等权，不做 packet 独立性显著性检验。

## E0：先核查、后固定推理

四个 `CV{0,1}_{full_cis,strict_cis}/final.pt` 必须实际存在，checkpoint epoch=20，config 与归档 config 一致，history 和 JSON 日志连续覆盖 1–20。核对训练代码固定 epoch 控制流：验证 F1 只保存 best 副本，不触发学习率调整、提前停止、参数重置或新训练策略。原源码 SHA 与归档 metadata 对齐，关联 git 版本检查并保存。

仅使用各模型当时的对应源训练 scaler；核验拟合键及从源缓存原始统计得到的 mean/std，不能用五台未见 RX 重新拟合。缓存及权重的全文件 SHA 在登记中固定，沿用每源 RX 2,000 包、每 TX 200 包及原 HDF5 行 ID。Full/Strict 标签与行 ID 必须完全一致。

模型及输入均沿用 float32；CPU eval/inference_mode、batch64、每个独立进程两线程。四个已有模型仅做推理，原 float32 logits 用 float64 softmax 转概率，并保留 logits 供源内温度校准。argmax 平局取最小类别索引，与 torch/numpy 一致。任何权重 provenance 条件失败，停止并记录证据不足，不改用 best.pt，也不自动复训。

原 checkpoint 未保存 optimizer state 或训练入口脚本哈希。固定 epoch 的核验依靠 checkpoint/config、逐 epoch 记录、匹配的源代码哈希以及历史训练代码控制流相互佐证；不声称有密码学执行证明。历史两个相关版本仅增加训练角色核验，不改变固定20轮或 optimizer 规则。

首次推理前 provenance 核验修订为1.1：初版注册、封存及程序保存在 `pre_inference_v1_0`，其下有修订时间和理由。`g1_data.py` 的归档 SHA 可从 git 5661a32 精确恢复，原有函数 AST 全部相同，仅新增 `validate_run_roles`；原 `shen_p2a/runner.py` 是批量记录中不被 G1 调用的旧程序，现哈希不匹配且未恢复原文，明确报告这一限制，不冒充原代码一致。其余归档执行相关文件须逐哈希匹配。修订不更换权重、数据、scaler、特征、模型、算法或任何评价门槛；新预测此时尚未生成。

## E1：固定的互补性关口

逐 RX 计算 Full/Strict Top1 正确性四格表：两者都正确、仅 Full 正确、仅 Strict 正确、两者都错。报告各 RX/TX 的 Full 可挽救与引入错误数，以及 Hard oracle 与 Convex soft oracle 的 Accuracy。

Hard oracle 为 `mean(Full_correct OR Strict_correct)`，只作为固定 Top1 二选一上界。每 RX 的可挽救空间减去该 RX 最强单专家 Accuracy，十 RX 等权平均必须 ≥0.03，才进入 E2。Soft oracle 不参与门槛，失败后不能靠更宽松的软上界继续。

Convex soft oracle 只允许逐包的一个标量 λ∈[0,1] 对两个概率向量作凸组合。解析求解正确类与其他类别的线性不等式交集，并保存可行区间及开闭端点，不用粗网格假称精确上界。保持实际 argmax 平局规则：正确类必须严格超过索引更小的类，可以与索引更大的类同分。不能泛化到 logits 融合、特征融合或 MoE；不输出 oracle Macro-F1 上界。

## E2：一版线性选择器与固定基线

每个一级折内部五次留一 RX。只从四台选择器训练 RX 中的“两个 Top1 不同且恰有一个正确”的包拟合监督目标：Full 正确为1、Strict 正确为0。选择器固定 StandardScaler + LogisticRegression，L2、C=1、lbfgs、max_iter1000、tol1e-8、seed20261012，不搜索模型或切换阈值。每个训练 RX 的 informative 包总权重相等，无类别加权。无 informative 包则默认 Strict；仅一个训练 outcome 则用该 outcome 常量，并报告退化情况。

输入只包含：两专家最大概率、Top1−Top2 margin、归一化熵及三者差值、Top1 分歧、概率总变差和 Jensen–Shannon 散度，共12个固定观测量。不包含真实标签、TX/RX 身份、真实类 margin、特征原型或手工 TX36 规则。两个专家同意时保持 Strict；存在分歧时，仅当线性模型输出≥.5 才选择 Full。

简单基线候选在每次四台训练 RX 上拟合/选择，并先保存 `selection.frozen.json`，再评价第五台：

1. 50/50 原概率平均。
2. 两专家各自在四训练 RX 上按等权 RX NLL 拟合温度，再50/50平均。固定 T∈[.25,4]、bounded log T scalar minimization、xatol1e-6、max_iter100，不改变求解器。
3. 源内选定的固定概率权重，Full 权重为0,.05,…,1；最大化四训练 RX 等权 F1。完全平局优先最接近.5，再较小 Full 权重。
4. 原最大置信度硬选择；Full 最大概率严格更高才选 Full，平局 Strict。

基线种类按四训练 RX 的等权 Macro-F1 最大值选定，完全平局按上述顺序。第五台的全基线结果均报告，但关口不在测试 RX 上挑最高者。Full/Strict 单专家同时列出。

四项用户指定关口全部满足才算 E2 通过：

|检查|阈值|
|---|---|
|十 RX 等权平均选择器 F1−源内选定基线 F1|≥.01|
|每台评价 RX 的 ΔF1|≥0|
|最差 TX Recall 改善|≥.05|
|支持数≥50的任何 RX×TX Recall 下降|≤.10|

最差 TX 先对每 RX/TX 分别计算 Recall，再对十 RX 等权平均每个 TX 的 Recall，最后取十 TX 最小值。另报每 RX 的最差 TX、逐类别差值、挽救数和新增错误。任何关口失败，保留负结果，不增加隐藏层、风险损失或人工规则。

E2 通过也只形成内部筛选证据，不自动启动 E3/E4 或新 CNN。全文与强基线比较中未核实的事项明确保留，不把简单融合或线性选择器当已证实的新算法贡献。
