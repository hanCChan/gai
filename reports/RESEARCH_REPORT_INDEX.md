# 第二篇 LoRa 研究：报告与证据索引

L0 交互现象分析分支为 `experiment/l0-interaction-phenomenon`；当前B1分支为 `experiment/b1-standard-baselines`；D0.5/B0 为 `experiment/d05-b0-class-risk`；D0为 `experiment/d0-class-pair-audit`，历史报告来自 `experiment/shen-p2a8` 等独立分支。下列报告、预登记、复核代码、CSV 和图表均保留。原始 IQ、CNN 权重和B1原预测NPZ留在原服务器，复现需要协议记录的输入及 SHA256。

|阶段|报告|结果|预登记与代码|
|---|---|---|---|
|L0 交互现象分析（计划+代码）|[代码审查](code_review/2026-10-19/CODE_REVIEW.md)|停止 R2 单步控制器的论文机制；提出 TX×RX 交互与 H-link/H-CFO/H-SNR/H-hw 竞争假说；只推理 L0 尚未在服务器运行，0 训练|[研究方案](../docs/interaction_l0/RESEARCH_PLAN.md)、[L0 代码](../scripts/interaction_l0/README.md)、[文献索引](../docs/interaction_l0/LITERATURE_INDEX.md)|
|B1标准机制基线|[完整报告](b1_baselines/2026-10-13/REPORT.md)|32作业完成，0.470496 GPUh；均未兼顾平均Accuracy与全局最低Recall；B2/新算法不启动|[冻结](../docs/b1_baselines/PREREGISTRATION.md)、[更新计划](../docs/b1_baselines/EXECUTION_PLAN.md)、[完整证据](b1_baselines/2026-10-13/EVIDENCE_INDEX.md)|
|D0.5/B0三seed|[最终报告](d05_b0/2026-10-13/REPORT.md)|增量损害部分可预测但跨CNN检索失败；9个严重类单元跨seed重复；8次拟合完成，B1待审|[冻结](../docs/d05_b0/PREREGISTRATION.md)、[B1具体计划](../docs/d05_b0/B1_REVIEW_PLAN.md)、[完整证据ZIP](d05_b0/2026-10-13/evidence.zip)|
|D0 类别判别审计|[报告](d0_class_pair/2026-10-13_D0/REPORT.md)|有限正证；强基线/跨seed未知；完成后停止，训练与推理0|[文献矩阵](../docs/d0_class_pair/LITERATURE_MATRIX.md)、[冻结规则](../docs/d0_class_pair/PREREGISTRATION.md)、[审计代码](../scripts/audit_d0_class_pair.py)、[全部证据ZIP](d0_class_pair/2026-10-13_D0/evidence.zip)|
|G1 / G2-0|[联合报告](shen_p2a8/2026-10-12_g1_joint_g2_0/REPORT.md)|G2-0 频带可靠性路线停止；Shen/OSU 历史结果及证据边界保留|[G1 注册](../configs/shen_p2a8/g1_registration.json)、[G2-0 实现](../scripts/run_shen_p2a8_g2_0.py)|
|VIRP-v1 / V0|[完整报告](virp/2026-10-12_v1_v0/REPORT.md)|CIS 变化/覆盖率未达登记要求，停止；零新分类器训练|[规则](../docs/virp/V1_PREREGISTRATION.md)、[注册](../configs/virp/v1_registration.json)、[审计程序](../scripts/audit_virp_v0.py)|
|EISR-v1 / E0–E2|[完整报告](eisr/2026-10-12_v1/REPORT.md)|E0/E1 通过；E2 线性选择器未通过，停止；零新 CNN 训练|[规则](../docs/eisr/V1_PREREGISTRATION.md)、[注册](../configs/eisr/v1_registration.json)、[CPU 流程](../scripts/run_eisr_v1.py)|

完整证据下载：

- [VIRP-v1 V0 证据 ZIP](virp/2026-10-12_v1_v0/VIRP_v1_V0_evidence.zip)
- [EISR-v1 E0/E1/E2 证据 ZIP](eisr/2026-10-12_v1/EISR_v1_E0_E1_E2_evidence.zip)

VIRP 和 EISR 的 `ARTIFACT_MANIFEST.json` 保存各包内文件的 SHA256。ZIP 保留冻结时的原始字节；Git 中报告 CSV 按项目已有规则规范为 LF。科学定义和关口不因上传而修改。EISR 首次推理前的1.0→1.1来源核验修订及其旧注册也完整保留。

原始提交历史也保存在 [Git history bundle](SOURCE_HISTORY_20261012.bundle)（SHA256：`7aa7bb8b8e699e60d46e07b63d24d9c864cad7c5369f8c1717ad47ab834fc44b`）。使用 GitHub 连接上传的分支快照时，提交 ID 可能不同于本地提交，但发布文件树逐字节核对一致。EISR 来源核验需要原始 `5661a32`、`8b81fbb` 等提交；首次从 GitHub 克隆后可恢复这些来源对象：

```bash
git fetch reports/SOURCE_HISTORY_20261012.bundle refs/provenance/gai-pre-publication-20261012:refs/provenance/gai-pre-publication-20261012
```

这只导入原始历史，不改变工作分支和冻结实验文件。

代码数值检查：

```bash
python -m unittest discover -s tests -p test_virp_v0.py -v
python -m unittest discover -s tests -p test_eisr.py -v
```

以上测试只验证数值路径、解析 oracle 和分组/特征约束；实际停止决定另有逐包证据及独立复核，不将测试通过当研究方法通过。
