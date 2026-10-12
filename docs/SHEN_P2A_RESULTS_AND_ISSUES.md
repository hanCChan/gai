# Shen P2-A 运行结果与问题（持续实验）

以 eb4ce2c 为冻结基线，新分支 experiment/shen-p2a；P1的32次运行、曲线、混淆矩阵及本地逐包预测已与冻结档案核对，全部保持不变。当前完成机制诊断代码、四卡真实数据冒烟、六文件SHA校验和各任务一个完整epoch的集成检查；主实验52筛选+56确认仍在运行，不能把单seed成绩写作稳健结论。

当前运行及配置见 `/home/keyan/code-y-p2a/outputs/shen_p2a/p2a_preregistered_v2`，初步快照见 [screening_snapshot](../reports/shen_p2a/2026-10-12_screening_snapshot/publication_manifest.json)。预登记在 [PREREGISTRATION](shen_p2a/PREREGISTRATION.md)。主流水线结束自动生成108次结果、独立sklearn指标核验、28组三seed、逐包配对和工程决策；P2-B/P3不自动启动。

已经完成的RTL34诊断见 [复盘](../reports/shen_p2a/2026-10-12_screening_snapshot/diagnostics/interpretation/REPORT.md)：三seed Day1训练均800/800，同日测试均200/200，Day2均0/200。重算Day2预测与P1逐元素一致。Day2原始RMS约翻倍，CFO/SNR原始元数据不同但单位/算法未知；CIS绝对值99分位均值反而略降，暂不支持极端比例值爆炸解释。普通Spec与CIS的粗统计均漂移；最大成分变化落在假设−101.6kHz附近的STFT位置。这是观察性关联，不能定位原始OOB因果、硬件或采集身份错误。

作者CIS和普通Spec的52频点裁剪覆盖约−203.125..195.3125kHz，其中只有17频点在假设标称IB内；因此二者不是严格IB表示。根据上述诊断，追加对称A7：原始FFT严格IB投影后重新STFT，保留同模型与2LR预算，8筛选+8确认。它在观察结果后提出，明确单独标为探索对照；原108的筛选、门槛和配置保持不变。只在主108完成后占用GPU。

旧论文目录只有PDF，未发现旧配置/日志/归档。文稿明确原6盲测已有结果，20台均存在研究使用历史；新10/4/6只能称本轮训练/开发/锁定，不可称整个研究历史独立盲测。旧代码执行闭环、实际日期/session及来源身份未核实。详见 [后续准备](shen_p2a/NEXT_STAGES_PREPARED.md) 和 [候选历史协议](../configs/shen_p2a/p3_history_review.candidate.json)。P2-B/P3代码仅合成检查；HSTU/Transformer共享tokenizer、主模型仅差7参数。TDL使用分数延迟且单正弦Doppler并非完整Jakes，Ma仅代表性基线。

保留的问题与修复：首次系统Python缓存/冒烟与P1环境不一致，保留后切回固定2.5.1+cu121/1.26.4/1.13.1重做；首轮正式启动的四个任务在CUDA统计初始化前失败，尚无训练epoch，日志完整保留。修复并通过四类完整epoch检查后重启。实际训练停滞（如RTL Full lr=.001隐藏表示全零）作为实验结果保留，不删除失败候选。

Fs/BW/SF仍为文献假设；未新读Day3/4或六锁定接收机，未下载OSU。原始IQ、权重、缓存和全部逐包预测留本地，仓库上传代码和汇总/复盘证据。Spec/CIS管线吞吐基于预计算特征，另测STFT成本；不能与P1在线STFT直接解释成模型速度优势。
