# 更新后的执行计划：D0.5+B0完成，提交B1审核后停止

输入：[用户原方案](USER_PLAN.md)，来源分支experiment/d0-class-pair-audit，公开commit dba18e8f5cb6bc9af48414c3e65fe7a7e7fb82be。独立worktree `/home/keyan/code-y-d05-b0`，独立分支experiment/d05-b0-class-risk。原D0/历史结果只读，所有停止路线不重开。

| 阶段 | 交付状态 | 结果/边界 |
|---|---|---|
| D0.5 | 完成+独立复算 | signed增量风险有内部信息，但不普遍超绝对混淆，跨拟合CNN损害检索失败 |
| B0 | 8/8实际拟合完成，三seed结果齐全 | 三seedStrict整体优势，9个严重单元≥2seed重复，7RX两折，零collapse |
| B1 | 具体审核矩阵/配置生成器完成 | 16内部搜索+16正式=32作业；0.7–2GPUh，申请总上限2；训练runner和GPU运行未启动 |
| B2 | 作者源码/输入/成本核查完成 | Shen完整Keras GRL≠低秩CNN；公开IB入口仅WiSig256点，LoRa/FAN适配与GPU成本待测；正式训练暂缓 |
| 新方法 | 未实施 | 不命名/实现新Loss、Gate、网络或人为类别规则 |
| GitHub | 每轮发表后核验 | 独立分支，回读源码/报告/decision/文件树及全部SHA；最后答复确认实际公开commit |

冻结科学定义见 [PREREGISTRATION.md](PREREGISTRATION.md)、configs/d05_b0/registration.json。初始来源检查过严造成8次零优化启动退出；完整保存original registration/seal/source/failure logs。后续仅改工程核验，实际训练仍8次、seed不变、原配方不变，初始GPU占用一并计费，详见ENGINEERING_CORRECTION.json。

当前主张收窄到“历史源池中有可重复的局部类别条件失败”，未证明signed损害profile跨拟合模型普遍转移，未证明强方法剩余缺口或可发表算法创新。完成本轮停止，[最终报告](../../reports/d05_b0/2026-10-13/REPORT.md)。不下载新数据，OSU/开发/锁定/Day3/4/原始IQ读取0。

复现需要registration指定源CIS/索引/manifest和旧预测，依赖`/home/keyan/venvs/shen-stage1`（Python3.11，torch2.5.1+cu121/NumPy1.26.4；本机GPU为V10032GB）。训练输出默认拒绝覆写；不可重新运行历史快照来隐藏失败。CPU D0.5与报告脚本环境还见requirements-d0.txt。用户后续另批B1前，不运行新的模型训练。
