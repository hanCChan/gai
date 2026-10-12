# VIRP-v1 · V0 预登记审计

**决策：V0 未通过，停止当前第一版；不进入 B0–B4。**

这是训练前的信号与表征可行性审计，不是分类性能或论文创新性验证。

注册 SHA256：`f1e60b34b1b8a3bb5e09a51487f49ebe7fd94f2b00c1dd00edc9fc093a41464a`。所有阈值、六维扰动族及审计程序均在新信号结果之前冻结。

两折各仅用五台源 RX 估计响应及 CIS 标准差。覆盖核验使用另外五台源 RX；四开发 RX、六历史锁定 RX 和 OSU 均未访问。

重新读取并验证十个源训练 HDF5 的 SHA256；沿用 G1 每 TX/RX 200 包的同一索引，原始 float64 重建 complex128。每源包评估其拟合折内全部四个 donor 响应，共 20,000 包、80,000 扰动实现；另有 40,000 个零扰动/全频增益控制。

标准化 δ 对 `log10` 或 dB 表示相同。每包 δ 是四个固定响应 δ 的中位数；上界、相关性及伪影指标额外检查全部实现。

|折|包数|median δ|δ≥0.05 包比例|全部实现 δ 的 q95|最小 IQ 相关性|核心 median δ|
|---|---:|---:|---:|---:|---:|---:|
|0|10000|0.030621|22.83%|0.127155|0.999478|0.026562|
|1|10000|0.058083|57.93%|0.163680|0.999364|0.049565|

## 全部硬性关口

|折|检查|观测|预登记要求|通过|
|---|---|---:|---|---|
|0|median_packet_delta_min|0.030620687|>= 0.1|否|
|0|packet_fraction_delta_ge_005_min|0.2283|>= 0.8|否|
|0|all_realizations_delta_q95_max|0.12715547|<= 1.0|是|
|0|nonfinite_count_max|0|<= 0|是|
|0|max_abs_response_db|1.5698242|<= 3.0|是|
|0|all_realizations_iq_correlation_min|0.99947792|>= 0.98|是|
|0|negative_control_delta_max|4.0383326e-06|<= 1e-06|否|
|0|median_packet_core_delta_min|0.026561873|>= 0.05|否|
|0|low_energy_residual_share_q95_max|0.3467996|<= 0.5|是|
|0|boundary_residual_share_q95_max|0.26545283|<= 0.5|是|
|0|extreme_cis_fraction_increase_max|7.9365079e-07|<= 0.001|是|
|0|coherence_change_absolute_q95_max|0.00014219247|<= 0.05|是|
|0|profile_half_split_stable_pair_fraction_min|1|>= 0.8|是|
|0|profile_loo_tx_relative_change_q95_max|0.15482687|<= 0.25|是|
|0|profile_smooth_explained_fraction_median_min|0.56420694|>= 0.5|是|
|0|heldout_spectral_improvement_fraction_min|0|>= 0.4|否|
|0|natural_cis_scale_ratio_min|0.1954179|>= 0.1|是|
|0|natural_cis_scale_ratio_max|0.1954179|<= 10.0|是|
|1|median_packet_delta_min|0.058083018|>= 0.1|否|
|1|packet_fraction_delta_ge_005_min|0.5793|>= 0.8|否|
|1|all_realizations_delta_q95_max|0.16367996|<= 1.0|是|
|1|nonfinite_count_max|0|<= 0|是|
|1|max_abs_response_db|2.2060634|<= 3.0|是|
|1|all_realizations_iq_correlation_min|0.99936379|>= 0.98|是|
|1|negative_control_delta_max|3.3327011e-13|<= 1e-06|是|
|1|median_packet_core_delta_min|0.049564977|>= 0.05|否|
|1|low_energy_residual_share_q95_max|0.33189563|<= 0.5|是|
|1|boundary_residual_share_q95_max|0.29440463|<= 0.5|是|
|1|extreme_cis_fraction_increase_max|0|<= 0.001|是|
|1|coherence_change_absolute_q95_max|0.0001830525|<= 0.05|是|
|1|profile_half_split_stable_pair_fraction_min|1|>= 0.8|是|
|1|profile_loo_tx_relative_change_q95_max|0.27959696|<= 0.25|否|
|1|profile_smooth_explained_fraction_median_min|0.65949224|>= 0.5|是|
|1|heldout_spectral_improvement_fraction_min|0|>= 0.4|否|
|1|natural_cis_scale_ratio_min|0.49202184|>= 0.1|是|
|1|natural_cis_scale_ratio_max|0.49202184|<= 10.0|是|

## 证据边界

纯幅频、零相位全包 FFT 修改仅称 frequency-response-inspired spectral augmentation；不声称是实测、因果可实现的接收机硬件响应。

同 TX 的自然 CIS 距离比较使用不同包的类别平均图，包含采集链路/信道/内容及采样误差，不能解释成同一发射波形的硬件因果效应。

半包谱形稳定性、leave-one-TX-out、源留出谱形覆盖、低能量分母和边界残余均已单列。每 RX 只有一个源训练文件，无法从这些数据建立同一 RX 的跨独立采集文件稳定性；不将两折或 80,000 扰动当独立环境做显著性检验。

未训练分类器，因而没有设备 Recall 或已证明的身份保护；per_RX_TX.csv 仅报告各设备扰动影响。未进行 OSU 新窗口审计或改写 G1 结果。

## 复核和复现

`configs/virp/v1_registration.json`、`.seal.json` 为冻结协议；`raw_source/*/manifest.json` 为原始数据出处；每折 `responses.frozen.npz`、`fit.frozen.json` 记录源拟合；`trials_*.csv` 保存逐包逐扰动；`packets.csv` 保存主指标；`negative_controls.csv` 保存控制结果。

复现必须使用新的空输出目录：`/home/keyan/venvs/shen-stage1/bin/python scripts/audit_virp_v0.py --output outputs/virp/v1_v0_reproduction`。不修改本注册、阈值或结果目录。

V0 失败后，不扩大幅频范围，不新增 CFO/IQ imbalance/时变扰动，不改 STFT，不启动 B0–B4。任何下一候选方法需独立注册并引用本次负结果。

## 独立复核与决策解释

独立程序从全部 80,000 个逐扰动记录重新汇总 20,000 个包；用一般最小二乘重新推导 40 个源响应，再从原始 HDF5 核验全部 100 个 TX/RX 单元各一个包及其四次扰动，共 100 个包、400 次独立重算。注册/代码封存及源域访问范围校验通过。响应最大绝对误差 7.22e-16，δ 重算误差为 0。见 `independent_verification.json`。

用户指定的 median δ≥0.10 和覆盖率≥80% 在两折均失败。这两项已足以停止；结论不依赖补充工程门槛或数值负对照的严格容差。

两折响应峰值分别为 ±1.570 dB、±2.206 dB 以内。只缩小超界曲线的规则没有把源经验变化放大到 ±3 dB；也没有在观察结果后增加幅度。最小 IQ 相关性为 0.99936，NaN/Inf 为 0，说明数值有限与高相关性通过，尚不足以提供规定幅度的 CIS 表征变化。

源留出谱形覆盖：两折均没有一台源留出 RX 达到预先定义的 ≥10% 距离改善。半包谱形方向一致性和六维平滑结构检查通过，但 leave-one-TX-out 的 q95 相对变化在折1为 0.2796，超过追加工程门槛 0.25。

零扰动和 +3 dB 常量控制的绝大部分 δ 约为 1e-13；b210_1 原始行3061（TX34）分别为 3.45e-6、4.04e-6，超过冻结的 1e-6 容差。该例最大差异位于末帧，clean 分母功率为 1e-24（STFT epsilon 能量尺度），全包分母中位数约 4.60e-5；68.32% 控制残余平方能量位于最低1%分母。这个局部浮点敏感性已记录于 `negative_control_case_diagnostic.json`，不通过更改 epsilon、丢包或调整阈值消除。

结论限于当前六基函数、源谱差异构造、幅频上界及历史 Full CIS 前端的第一版候选。不能据此证明所有幅频增强无效，也没有 B0–B4 分类性能或 Group DRO 身份保护收益结果。按用户授权的 Stop/Go，V1/V2/V3 均未执行。

单独的分布图见 `figures/V0_audit.png`，另有 PDF/SVG。文献核对见 `docs/virp/V0_LITERATURE_AUDIT.md`；未取得全文的材料明确标记，不将公开摘要或代码比较说成完整创新性审查。
