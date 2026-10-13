# P0 证据索引

|文件|用途|
|---|---|
|REPORT.md / decision.json|P0-C结论及三问题答复|
|registration.json / seal.json|结果前规则与代码SHA|
|packets.csv / chirp_blocks.csv|2000包、16000块逐项指标|
|per_RX.csv / per_RX_TX.csv|10 RX和100个RX×TX单元|
|per_hardware_type.csv / per_source_fold.csv|硬件类型及固定两折，非独立采集统计|
|synthetic_trials.csv / synthetic_validation.json|合成时移与负对照|
|source_manifest.json / signal_access_log.json|十源文件来源、元数据和只读访问|
|independent_verification.json|100单元独立重算、1低能量案例及复核访问|
|representative_profiles.json / representative_candidate_curves.json|每RX×TX首个选定行的频率轮廓/候选曲线，无原始IQ|
|literature_*manifest.json|公开资料来源/检索SHA与失败记录|
|provenance.json|私有拒稿材料hash、缺失原稿及原目录状态|
|figures/*.png / *.pdf|独立图件|
|ARTIFACT_MANIFEST.json|文件完整性SHA|

私有拒稿信/原稿、原始IQ和CNN权重不在报告或ZIP。完整准确样本行号见仓库 configs/p0_sync/sample_manifest.json。
文献全文/代码快照只存本地outputs/p0_sync/20261013_a0_literature；索引记录获取成功/失败，ZIP不包含他人全文。
