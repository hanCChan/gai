# P0-A0：数据、拒稿意见与历史代码溯源

## 原始材料

完整拒稿信来自用户指定 `/home/keyan/code-y/jugaoyijian/yijian`，
已逐条阅读编辑意见及 Reviewer 1–3，并存为仓库外只读私有副本
`/home/keyan/private/p0_sync_20261013/rejection.txt`。不提交原文或私人信息。
通知日期 2026-08-20，编辑认为贡献有限，拒稿通知禁止该稿及其修订稿重投 IoT-J。

与 P0 最直接相关的是 Reviewer 1 第8条、Reviewer 2 Major Concern 5：
chirp group/intra-chirp 位置如何从实际输入得到、其同步要求没有定义清楚。
这不是同步导致分类失败的实验结论。Reviewer 1 第9–11条和 Reviewer 2
第6、8条同时要求处理独立样本单位、训练协议混杂和窗口/文件聚合反转。
闭集“认证”、容量/输入不公平和有限部署表现的批评保持为后续任务边界。

本地检索未找到被拒 RF-HSTU 原稿 PDF 或另外提到的
`P0_Sync_Audit_Codex_Brief.md`。用户本轮粘贴的完整执行决策保存在 USER_PLAN.md，
本轮依该实际收到的规范执行，不声称读过未收到的附加文件。
现有 `/home/keyan/code-y/oob-lunwen1/OOB_IOTJ.pdf` 首页为
“Diagnosing Receiver-Sensitive Out-of-Band Dependencies in LoRa Radio-Frequency
Fingerprint Identification”，属于独立 OOB 诊断研究，不能代替被拒原稿。
其核心结果不作为本轮新贡献；本轮只读取首页确认身份。

## 数据集—任务与独立单位

|数据|本轮取得的资料|适用任务|当前可核验的单位/缺口|本轮信号访问|
|---|---|---|---|---|
|Shen 20 RX|作者原文、G1 注册/索引、十源 HDF5 元数据与 SHA256|跨 RX，前导内表征分析|包行号；每源 RX 一个训练 HDF5，独立 session/连续捕获/起点真值未知|仅十源 train，各200行，共2000行|
|Shen 60 TX|原文、作者 README、loader、HDF5 格式示意图|身份泛化、度量注册、未知设备评估|说明中为提取的前导 IQ；26 文件总数与表中19具名文件不完全对应，实际目录尚未取得|零原始数据下载/读取|
|OSU|已归档数据审计、M0/G1 旧报告|日期、距离、位置和配置变化|连续录制文件；旧连续窗不是独立包或独立采集，严格前导仍需核验|零原始 IQ/新窗口处理/目标评价|

十源 RX：B200 mini一台、B210一台、Pluto一台、RTL七台。
不使用开发 RX（b200_1/n210_1/rtl_4/rtl_8）、六历史锁定 RX 或任何日期文件。
源文件的 data=(8000,16384) float64，I/Q 各8192；label/CFO/SNR=(1,8000)。
本轮全部十文件及四字段属性均为空。校验只能证明和历史文件一致，不能证明作者原版或处理历史。
源 IQ 的 RMS 不是逐包一，因此不能从论文“RMS 归一化”倒推出当前发布文件已按同序处理。
重复结构与名义参数兼容，不能确认本地物理参数；CFO/SNR 字段的单位和生成过程也未独立确认。
同 TX 在不同 RX 的相同行号不作为同步发射/同一波形配对。

## Shen 60 TX 的有限成本审查

[作者 README](https://github.com/gxhen/LoRa_RFFI/blob/main/Openset_RFFI_TIFS/readme.md)
确认 60 台设备、N210 接收，说明26个 HDF5，但训练/评估表只具名19个。
表内含两套增强训练，不能把增强包当新的独立采集。
[格式示意图](https://raw.githubusercontent.com/gxhen/LoRa_RFFI/main/Openset_RFFI_TIFS/readme_images/HDF5.jpg)
标示每行 I/Q 各8192，作者 loader 也是按前后两半重建 IQ。

按19个具名文件表格求和为122,000行。若每实数 float32，单信号数组约
7.995 GB；若 float64，约15.991 GB。这里只是未压缩数组的条件性估算，
不含另7个文件、元数据、压缩率或镜像差异，不是实际下载大小。
最小无增强训练+residential注册+rogue组合共20,000行，条件性数组规模约
1.311/2.621 GB（float32/float64）。这是未来范围示例，不是下载或训练授权。
Dataport 下载页本轮无法读取，正式压缩包大小、校验和与下载权限未知。
获取/下载任何完整新数据前应单独说明范围和真实成本。

## 代码与历史边界

保留 G1/G2-0、VIRP V0 和 EISR E2 的负结果，不复用失败模型当强基线。
`src/shen_stage1/frequency.py` 只估计 modulo Fs/N 的周期相位频率代理，
不是前导起点检测器；本轮测试独立确认相差 Fs/N 的频率得相同相角。
作者原文的上游同步程序/硬件设置在现有公开 loader 中没有实现。
新诊断没有训练、权重加载、CIS 参数改写、真实 IQ 补偿或重采样。
Git 初始状态、远端头及相同文件树/不同历史的记录见 GIT_BASELINE.json。
