# WiSig ManyRx 获取状态

2026-10-13，北京时间。**Kaggle 单文件分块续传进行中，尚不能称为下载完成。**

旧官方配额轮询已停止并保存日志。Kaggle公开API清单列出 `ManyRx.pkl/ManyRx.pkl`，解压文件2037337668字节；只获取这个文件，不下载约10.4GB全集。公开单文件ZIP目标大小1310655283字节，无需服务器Kaggle凭证；CLI本身因未配置username而失败。

首次单流传输在251658240字节中断，部分文件原样保留。后续核验前后64KiB的字节范围响应，四连接、32MiB分块、每范围至多三次传输尝试。仅完成传输、ZIP CRC、SHA256与隔离结构校验后才能将状态升级。

官方gdown单次尝试仍返回共享配额超限，已停止，没有定时重试或扩大Full WiSig下载。来源与官方许可见[source_provenance.json](source_provenance.json)。Kaggle列出的许可为Unknown，原WiSig采用CC BY-NC-SA4.0；没有官方checksum，不能断言副本字节相同。

与获取独立的[接收条件变换实验](../../receiver_transform/2026-10-13/REPORT.md)已失败停止：不使用WiSig数值，不进行外部泛化确认。原始IQ、dataset文件、特征、权重及凭证不上传。
