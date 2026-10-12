# 第二篇 LoRa 研究：报告与证据索引

本索引对应 `experiment/shen-p2a8` 分支。下列报告、预登记、复核代码、CSV 和图表均保留；两份 ZIP 包包含完整逐包诊断证据。原始 IQ 和 CNN 权重留在原服务器，复现需要协议记录的输入及 SHA256。

|阶段|报告|结果|预登记与代码|
|---|---|---|---|
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
