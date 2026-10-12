# 当前执行状态

执行分支experiment/shen-p2a8，最新用户指令USER_SUPPLIED_G1_REVIEW_AND_NEXT_DECISION.md已执行至本轮关卡结束。

OSU下载完成：官方50文件8GB，逐文件独立SHA256一致。原冻结24次四基线/双方向/三seed训练全部完成，48份预测独立核验；RX1/RX2实际使用1612/1636个周期窗口，全部逐TX计数与训练曲线保留。没有改变窗口规则、频带、LR/WD或20epoch，没有用目标调参，没有以部分/合成数据替代。

M0六文件24000包审计与Shen20次G1训练此前完成；本轮按新方案新增只读Shen/OSU逐RX/seed/TX、源head与margin审计。CIS相对收益在OSU六个配对均同向，但目标准确率仅10%–14%，不能视为可用通用算法。Shen平均收益不能掩盖B200-1/TX36某seed从93%→1%的召回损失及170个新误判TX40。

G1操作关卡允许探索G2-0，随后完成十折九源拟合、一RX留出的固定CIS统计/线性分类器预测分析。九源内部RX分组CV Accuracy为字面负对照，同时保留Macro-F1对照；scaler、D/V、classifier、频带选择全在九源拟合并保存，随后才读取留出RX评价。未使用十源CNN制造泄漏。v1为F1对照版本，v2补充Accuracy对照，两个版本保留且80份预测逐字节相同。

G2-0未通过：S=D/(V+epsilon)平均排序相关0.4000，源CV Accuracy为0.9190；S选择的留出F1均值0.2558，对照0.6141，差−0.3584，10/10 RX均退化。80个九源拟合与160000次留出预测独立核验，26项回归检查通过。

依照关卡，G2-1/G3未启动，也不扩大400/800包预算或启动旧P2-B/P3，不为当前评分开发新损失。结论只针对固定统计特征/线性分类器，不否定所有可靠性建模，也没有证明CNN失效可被该指标预测。优化/BN早期根因、同步前导和物理溯源仍是未解决问题，需要另行冻结研究协议。

完整联合报告与CSV/图：reports/shen_p2a8/2026-10-12_g1_joint_g2_0/REPORT.md。

实时/原始归档入口：

- outputs/shen_p2a8/research_execution_status.json（最新总状态）
- outputs/shen_p2a8/osu_download/independent_verification.json
- outputs/shen_p2a8/g1_osu_v1/pipeline_status.json
- outputs/shen_p2a8/g1_joint_review_v2/decision.json
- outputs/shen_p2a8/g2_0_v2/decision.json
- outputs/shen_p2a8/g2_0_v2/independent_verification.json

历史124次P2-A、P1预测/权重和原始HDF5保留；M1 30次训练零次启动。Day3/4、六锁定RX均未访问。G2-0未使用四开发RX。当前没有继续占用GPU的正式训练。
