# P2-A 预登记与权限边界

冻结基线 eb4ce2c0db783f9241777f8b94328500a1014113。所有 P1 原始结果和历史预测保留在原目录。新分支 experiment/shen-p2a；仅访问 n210_1/rtl_6 的 Day1 Train、Day1 Test 和 Day2 Test 六个原始文件。Day2 是研究开发集，不作为独立最终性能。Fs=1MHz、BW=125kHz、SF=7 仍是未核实的文献假设。

## 首轮 52 个候选

GPU0 16：相同 658314 参数、rank16 作者卷积前端的 Spec/CIS，各接收机各 2LR；原始 CIS 轻量网络 2LR；RTL 作者/轻量 CIS 对称增加 WD=1e-4。GPU1 15：统一频域 RI 的 IB、near、mid、far、allOOB 两接收机，以及 RTL 的保护带、尺度与窗敏感性。GPU2 13：相同严格 IB 主输入的 Single、IB+IB Concat/Gate 容量对照、IB+OOB Concat/Gate 两接收机，以及 RTL 三种环带 Gate。GPU3 8：RTL Full/IB 各 2LR×2WD，完整 30epoch 并记录旧 patience7 的反事实早停。

统一 Adam、batch64、CE、clip5、max30epoch、patience7、min_delta1e-4；A6 除外仅关闭早停。Spec 与 CIS 使用同一个 STFT 配置和 52×126 几何；Spec 删除第一帧，与 CIS 分子列严格对齐。各自只用 Day1 Train 标量均值/标准差。rank16 头保留空间前端但限制线性头秩，不能说成只改变参数量而不改变假设空间。作者大网络 P1 两 LR 原始 CIS 结果复用，避免重复选择。

IB |f|≤62.5k；near (62.5k+guard,125k]；mid (125k,250k]；far (250k,500k]；默认 guard12.5k，敏感性 0/25k。各环均保留完整8192频率坐标。独立 RMS 不含 IB 统计。ib_relative 显式携带 IB 能量侧信息，不能称纯 OOB。正式 Hann 路径严格执行 FFT 投影→IFFT→Hann→FFT→重新投影。原包先加 Hann 仅作泄漏诊断，不能用于严格主/辅信息隔离的结论。主 IB 和 A6 使用 P1 完全相同变换。

## 三 seed 确认规则（在读取 P2-A 结果之前登记）

首轮 seed20261012；以完整 Day2 十类 Macro-F1 选每个 receiver/comparison_group/method 的候选（并列选较小 WD，再选原 .001）。选完冻结配置，seed20261013/20261014 不再调参。确认 A1、A2 raw容量、A4全部五个统一频域频带、A5全部五个方法、A6 Full/IB，共28组56次新增。A2正则化和A4敏感性/环带融合为首轮探索，不宣称三seed稳健性。所有候选结果公开，不事后隐藏差的频带。

逐包比较同 receiver/seed/sample_id/label，报告纠错、新增错误、每设备净收益。A5 进入 P2-B 的门槛：至少一台接收机的一种 OOB 融合在三个 seed 上准确率、Macro-F1 均严格优于 SingleIB 及相同结构的 IB+IB 容量对照。此门槛是开发工程规则，不是显著性检验，也不证明硬件机制或接收机泛化。即使满足，也只给建议，不自动启动 P2-B/P3。

## 环境与失败记录

固定 /home/keyan/venvs/shen-stage1/bin/python，torch2.5.1+cu121/numpy1.26.4/scipy1.13.1/sklearn1.5.2。首次系统环境缓存及冒烟保留为 cache_legacy_environment_v1 / four_gpu_smoke_v1，不能混入正式结果。固定环境重建缓存、四卡冒烟、六文件哈希及作者 CIS 十类样本逐元素一致检查通过后才正式启动。P1 原训练未记录梯度，不能从曲线推测其梯度；A6 新记录梯度和训练停滞指标。
