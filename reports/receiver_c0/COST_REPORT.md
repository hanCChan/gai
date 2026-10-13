# 受控步骤等价优化与成本

本次120个计时case、24个原候选数值重放、24个稀疏诊断case，共168次一次性Adam步骤；每次从旧状态重置并丢弃结果，连续训练轨迹0、正式拟合0。四卡工程占卡总额0.073113763 GPUh，独立360秒watchdog，不扣历史3GPUh余额。

优化只做等价执行：每步缓存pre-C1，按1/.5/.25降序提前退出，复用已接受损失；C2只在固定稀疏时点读取pre和选中倍率，不遍历全部倍率。拒绝时目标损失直接为0。缓存不跨改变参数的步骤。四卡所有legacy/optimized对应case的倍率、参数和Adam状态精确一致，C2不改变决策；原代码及规则不修改。

原WiSig执行器每步112次分批forward、25600个包前向；Shen32次、6400包。新无C2控制最少2个全池pass，最多4个；实际WiSig14或28分批forward，Shen4。稀疏C2仅增加1或2个全池pass。

warm步骤统计（首case的CUDA冷启动另计，不按每步重复计入）：

                         max_gradient_seconds  max_decision_seconds  max_total_seconds
dataset mechanism                                                                     
shen    legacy_mean                  0.076114              0.485208           0.561322
        legacy_multi                 0.076114              0.668376           0.737745
        optimized_mean               0.076114              0.109651           0.179019
        optimized_multi              0.076114              0.110646           0.179016
        standard                     0.076114              0.002009           0.077412
wisig   legacy_mean                  0.071091              0.474110           0.531691
        legacy_multi                 0.071091              0.963659           1.030182
        optimized_mean               0.071091              0.220784           0.287307
        optimized_multi              0.071091              0.221698           0.288221
        standard                     0.071091              0.001553           0.072033

四卡显存：
{
  "0": {
    "allocated_decision_peak_MiB": 775.306640625,
    "reserved_peak_including_gradient_cache_MiB": 2214.0
  },
  "1": {
    "allocated_decision_peak_MiB": 775.306640625,
    "reserved_peak_including_gradient_cache_MiB": 2214.0
  },
  "2": {
    "allocated_decision_peak_MiB": 775.306640625,
    "reserved_peak_including_gradient_cache_MiB": 2214.0
  },
  "3": {
    "allocated_decision_peak_MiB": 775.306640625,
    "reserved_peak_including_gradient_cache_MiB": 2214.0
  }
}

allocated峰值是decision阶段的计数；reserved峰值包含此前完整MLDG梯度的allocator缓存，是观测到的整步内存保守包络，不冒称精确梯度allocated峰值。时长另含真实梯度、源验证估算、每epoch3秒日志/checkpoint余量、稀疏C2完整事务余量、90秒setup与50%余量。三种方法的保守条件估算见CSV；不能把三份短轨迹样本视为未来完整训练的确定硬上界，正式预算与超时watchdog还需批准。

如果以后科学条件满足，18项完整矩阵的条件估算为12.207 reserved GPUh，建议独立上限14 GPUh；本次未申请或使用这笔正式训练预算。不得引用原R2-A1.03GPUh直接推算控制器训练。
