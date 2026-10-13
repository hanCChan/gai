# C0 证据索引

|文件|内容|
|---|---|
|REPORT.md / decision.json|自包含结论；C1/C2未启动及停止理由|
|OSU_per_capture_assets.csv|120个正式capture的日期/标签/原始文件/元数据状态|
|OSU_per_seed_assets.csv|13个模型×seed记录，区别历史权重报告与当前存在|
|Table_II_historical_reaggregation.csv|历史逐seed汇总再汇总，人口/样本std分列；非逐窗口复现|
|checkpoint_provenance.json|六份正式权重的SHA/旧配置一致性，不含权重|
|OSU_prediction_search.json|旧预测文件搜索及格式，早期/跨RX协议不冒充正式主实验|
|Shen_source_headers.json|10源训练HDF5元数据/结构，无IQ值|
|Shen_prediction_asset_headers.json|20个既有源留出NPZ头部与SHA，不含分数值|
|INPUT_MANIFEST.json|本轮实际资料、源码、元数据、缓存/权重SHA，私有资料仅hash|
|access_log.json|白名单HDF5元数据访问；零IQ值读取|
|independent_verification.json|stdlib独立重汇总及旧JSON一致性|
|ARTIFACT_MANIFEST.json|最终公开文件完整性SHA|

更新计划、用户答复范围、P0远端核验和文献矩阵在docs/c_evidence/。
原稿/拒稿信、权重、原始IQ、完整缓存均不上传。
因C0关口停止，没有本轮K-accuracy、错误相关性或C2性能图；不生成替代实验。
