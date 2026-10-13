# WiSig ManyRx：第三方获取成功与封存校验

2026-10-13，北京时间。**下载已完成，ZIP CRC、SHA256和元数据结构校验通过。** 保持信号封存，没有用于研究训练、统计分析、参数选择或跨制式确认。

## 文件与结构

- 本地：`/home/keyan/data/wisig-manyrx-kaggle/ManyRx.kaggle.v1.zip`
- ZIP：1,310,655,283字节；一个成员 `ManyRx.pkl`，2,037,337,668字节，CRC32=3709036759。
- SHA256：`b0eca705816b03a47290286fd23ca1a10e02c6f9bd29b5461cfd92c13a86fbdd`。
- 必需字段：`tx_list/rx_list/capture_date_list/equalized_list/max_sig/data`；10TX、32RX、4日期、256复采样点，等化标记[0,1]，max_sig=200。
- 日期：2021_03_01、2021_03_08、2021_03_15、2021_03_23。
- 共2560个数组占位结构；未等化249666包，等化247684包，52个空结构，87个结构不足200包。

这些字段与[作者加载工具](https://github.com/WiSig-dataset/wisig-examples/blob/master/data_utilities.py)和[官方说明](https://cores.ee.ucla.edu/wisig/overview/)一致。官方说明ManyRx并非完全均衡；**两个处理版本包数不同，不能默认逐包配对**。若今后研究等化前后身份信息，必须先取得可靠包对齐依据或采用分布级比较；本次没有验证同步发射或对应包ID。

检查器将仅三个NumPy构造入口替换为本地惰性元数据类，拒绝其他pickle globals和persistent IDs。读取压缩成员与buffer长度以校验结构，但不构造数值数组、不计算信号统计、不运行第三方任意类；没有使用通常的pickle.load执行原数据对象。合成数组占位和拒绝未知global的检查通过。详细[元数据](metadata_summary.json)与[逐结构清单](array_inventory.csv)公开，信号buffer不公开。

## 来源和许可

来源：[Kaggle副本v1](https://www.kaggle.com/datasets/jeffhei1206/wisig-rf-fingerprinting-dataset)，发布于2023-03-27。公开API清单列出 `ManyRx.pkl/ManyRx.pkl`；解压文件大小2,037,337,668字节符合清单。仅下载单文件，未下载约10.4GB副本全集、Full WiSig或其他子集。

它是第三方副本。结构匹配与CRC/SHA256证明下载完整性，**不能证明与官方ZIP或pickle字节一致**；本次未获得官方checksum。Kaggle页面许可字段为Unknown，不能将它当作额外授权；[原官方页面](https://cores.ee.ucla.edu/downloads/datasets/wisig/)声明数据采用CC BY-NC-SA4.0，并要求WiSig和ORBIT论文引用。下载来源、许可与代码页面的哈希记录在[source_provenance.json](source_provenance.json)。

## 执行经过与保留证据

旧官方配额轮询已停止，三次WAITING_OFFICIAL_QUOTA日志与状态保存。服务器未配置Kaggle username/token，CLI清单调用失败；公开API可读文件清单并提供单文件下载，无需新增凭证。官方gdown单次尝试仍报共享文件配额超限，随后停止，无定时重试、无作者邮件或账户操作。

首次单连接传输在251658240字节中断，原始partial完整保留。之后核验partial首尾64KiB与v1对象范围一致，使用四连接32MiB分块、每请求最多三次传输尝试。某范围达重试上限后，保留28个完成块，只对剩余四块改用8MiB请求与持久连接，响应头先验证206/Content-Range/ETag再读取body。这个补齐阶段另有一次客户端变量遮蔽错误，在范围读取前中止；修复与失败日志均保留。最终按字节顺序重组，校验整包CRC与SHA256。所有partial和分块保留，没有覆盖已收到的数据，没有把网络中断写成配额绕过。

[下载状态](download_status.json)、[停止的官方任务](official_stopped_status.json)、[gdown错误](gdown_single_attempt.log)、[续传日志](kaggle_remaining_ranges_fixed.log)与[EVIDENCE_INDEX](EVIDENCE_INDEX.md)可审计。GitHub仅发布脚本、事实、元数据和hash，不上传dataset、IQ、权重、数值特征、凭证或私稿。

## 研究决定

与本获取任务独立的[严格留出TX共享性实验](../../receiver_transform/2026-10-13/REPORT.md)完成20个encoder后未通过全部四fold×seed门槛；固定rank2机制停止，分类PoC为0。WiSig成功获取不改变这一止损决定，不因此追加GPU作业。继续封存信号，未来研究需后续指令。
