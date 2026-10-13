# B1更新计划：执行已完成，发布核验后停止

用户明确批准：B1共32次作业，上限2 reserved V100 GPU小时。来源为远端D0.5/B0 commit `2b89877ae4ec7f95b733faa30381f6e3c6f50518`；独立worktree `/home/keyan/code-y-b1`、分支 `experiment/b1-standard-baselines`。历史报告、缓存和模型不修改。

|阶段|批准范围|完成状态|
|---|---|---|
|实现与预检|CE、RX×TX Group DRO、GRL、SupCon；同Strict CIS/rank16 CNN；目标数学CPU检查、40资产SHA、六共享schedule|通过；优化前seal保存|
|矩阵内计时|首个内部CE搜索，计入16个搜索；固定保守公式外推，不额外pilot|32.036871秒，投影.889913 GPUh，低于2|
|内部选择|两折×四方法×两LR，固定10轮；4RX拟合/1RX内部验证|16/16完成；精确macro-F1一次冻结LR|
|正式模型|两折×四方法×两seed，固定20轮；五源拟合/五源评价|16/16完成；不选外层epoch/LR|
|分析/核验|全部RX、TX、90有向对、混淆、margin、优化状态、成本；独立复算|32权重/96预测文件/800Recall/600配对/10000batch核验通过|
|发布|代码、更新计划、报告、决策、全数值/真实日志ZIP、SHA，GitHub远端回读|随本轮发布；以远端树和逐文件SHA核验为完成依据|
|后续|ArcFace、作者级Shen GRL/IB-RFF、额外权重搜索、新Loss/Gate/network|未批准，不执行|

实际32拟合作业、0失败、0重试，总0.4704956571 GPUh。完整[报告](../../reports/b1_baselines/2026-10-13/REPORT.md)和[证据](../../reports/b1_baselines/2026-10-13/EVIDENCE_INDEX.md)包含所有不利结果。首轮未同时保住平均Accuracy并改善全局最低Recall，不据此宣布强方法普遍失败或新算法GO。

有限结论仅适用于已批准的固定系数、两个LR、一个内部验证RX、10/20轮和两个正式seed。GRL增加训练用head参数，单报成本。已用历史源池没有独立盲测意义；B0的Full/Strict与本轮Strict CE对照分开，不归因于同一损失改变。

代码入口：

- `scripts/test_b1_baselines.py`：合成CPU数学/forward/采样检查，无实际拟合。
- `scripts/prepare_b1_execution.py`：严格输入/共享schedule/代码seal预检。
- `scripts/launch_b1_baselines.py`、`scripts/train_b1_baseline.py`：固定矩阵，失败/超时保留，无自动重试；末轮评价。
- `scripts/analyze_b1_baselines.py`：读取已保存预测，完整派生表，无训练/新推理。
- `scripts/verify_b1_baselines.py`：独立读取checkpoint/预测/CSV复算，不训练/推理。
- `scripts/report_b1_baselines.py`：纯CPU报告、PNG/PDF/SVG及完整证据ZIP。

已完成的历史目录禁止重新运行覆盖。复现需独立目录、同版本环境、注册中的十源RX缓存/标签/rows/manifest及对应SHA；将history_root迁移到新位置时创建自己的新注册/seal，不能改写本轮记录。权重和NPZ没有发布，公开派生表可直接独立复核，完整模型检查还需要服务器资产。

有服务器资产时，仅核验本轮的入口为：

```bash
/home/keyan/venvs/shen-stage1/bin/python scripts/verify_b1_baselines.py
```

当前没有理由重复训练或扩展测试。本轮完成发布核验后停止，下一步需要另行审核ArcFace/作者级数据适配、真实计时和预算。未经新批准不访问开发/锁定RX、Day3/4、OSU或新大型数据。
