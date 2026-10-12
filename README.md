# gai：LoRa RFFI 开发实验

[研究报告与完整证据索引](reports/RESEARCH_REPORT_INDEX.md)：G1/G2-0、VIRP-v1 和 EISR-v1，包含冻结规则、独立复核与证据 ZIP。

最新 [EISR-v1 CPU 筛选](reports/eisr/2026-10-12_v1/REPORT.md)：E0四个真实固定20轮权重核验通过，E1 Hard oracle平均可挽救空间8.41pp通过；E2十次四RX训练/一RX评价的线性选择器相对源内选定简单融合平均ΔMacro-F1为−0.0783pp、五RX下降、最差TX Recall下降2.05pp，按规则停止。没有新增CNN训练或开发/锁定/OSU RX评价。[冻结规则](docs/eisr/V1_PREREGISTRATION.md)及[文献核对](docs/eisr/E0_LITERATURE_AUDIT.md)保留。

最新候选 [VIRP-v1 的 V0 审计](reports/virp/2026-10-12_v1_v0/REPORT.md) 已完成：两折 CIS 变化中位数 0.0306/0.0581，覆盖率 22.83%/57.93%，未达到预登记的 0.10/80% 门槛，按规则停止；没有启动 B0–B4 或目标 RX 新评价。[冻结规则](docs/virp/V1_PREREGISTRATION.md)和[文献核对](docs/virp/V0_LITERATURE_AUDIT.md)均保留。

保留旧RF-HSTU/OOB实现，新增受开发协议保护的Shen P0/P1训练、审计与报告流程。

- [当前结果与问题](docs/SHEN_RUN_RESULTS_AND_ISSUES.md)：2026-10-12北京时间快照，首筛16/16、复验16/16完成，八组均完成三seed。
- [结果CSV与每次运行记录](reports/shen_evidence/2026-10-12_p1_three_seed/)：Accuracy、全验证集Macro-F1、混淆矩阵、错误例、校准、训练曲线、参数量和吞吐。
- [P0修复与四卡smoke](docs/SHEN_P0_AUDIT.md)、[运行说明](docs/SHEN_EVIDENCE_REPRODUCE.md)、[P2/P3待确认方案](docs/SHEN_P2_P3_PLAN.md)。

## 运行条件

Shen新流程使用Python3.11，torch2.5.1+cu121；其余版本见requirements-shen-stage1.txt和报告环境记录。旧工具要求Python3.10或更高。原始Shen HDF5需放到data/external/shen_receiver_agnostic；原始数据与权重保留在服务器。冻结协议已放入configs/shen_evidence/protocol.candidate.json，SHA与真实训练时的原协议相同。

```bash
python -m unittest discover -s tests -v
python scripts/audit_shen_evidence.py --output outputs/shen_evidence/p0_audit_new
python scripts/launch_shen_evidence.py --stage smoke --output outputs/shen_evidence/p0_20261012/four_gpu_smoke_v2
python scripts/launch_shen_evidence.py --stage p1 --smoke-dir outputs/shen_evidence/p0_20261012/four_gpu_smoke_v2 --output outputs/shen_evidence/p1_seed20261012
python scripts/launch_shen_evidence.py --stage p1 --matrix configs/shen_evidence/lr_search_matrix.json --wait-for outputs/shen_evidence/p1_seed20261012 --smoke-dir outputs/shen_evidence/p0_20261012/four_gpu_smoke_v2 --output outputs/shen_evidence/p1_lr0003
python scripts/finish_shen_p1.py --output outputs/shen_evidence/p1_final_balanced
```

使用新的空输出目录；既有/失败实验不覆盖。最多30epoch，Day2全数据集Macro-F1选模，无重复CFO补偿。协调器根据统一两个lr候选冻结配方，再按登记规则进行另两个seed复验；单seed的多个lr不会当作不同seed。

发布版confirmation_matrix.json及seed20261013/14配置已对应本次实际冻结的学习率；selection.frozen.json可复核选择理由。跨服务器复现实验后应重新执行自己的完整初筛/冻结流程，而不是改变该历史快照。

本轮禁止Day3–4及最终接收机信号访问；P2/P3不自动运行。所有结果均为开发选模结果，不能当作未见接收机或最终测试证据。第三方代码保留原来源、署名与许可，见NOTICE及external中的说明。
