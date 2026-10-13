# WiSig 独立验证：仅 metadata 与成本核查

已核对 [UCLA 官方总页](https://cores.ee.ucla.edu/downloads/datasets/wisig/)、[官方下载页](https://cores.ee.ucla.edu/wisig/downloads/) 与 [作者论文](https://arxiv.org/abs/2112.15363)。完整集合为174 TX、41 USRP RX、一个月内四次采集。已处理 preamble 是256个 IQ 样点，有 equalized/non-equalized 两版本。下载0字节数据，仅浏览说明与小代码文件。

| 子集 | TX | RX | 每TX–RX–day名义包数 | day | 官方压缩体积 |
|---|---:|---:|---:|---:|---:|
| ManyRx | 10 | 32 | 200 | 4 | 1.2 GB |
| ManySig | 6 | 12 | 1000 | 4 | 1.4 GB |
| ManyTx | 150 | 18 | 50 | 4 | 2.5 GB |
| SingleDay | 28 | 10 | 800 | 1 | 1.0 GB |

官方仅 ManySig/SingleDay 完全均衡，其余至少90%的每RX下TX满足名义包数。不能直接假定 ManyRx 每个交叉单元都有200包。上述压缩体积不是解压后 RAM/磁盘保证；确认下载前须核对实际资产和格式。子集包含重叠 TX、RX、信号，不能把 ManyRx/ManySig 当互相独立的测试集；same packet 的等化前后版本也不独立。

若批准后做独立验证，优先候选 ManyRx：10 TX 便于复用90个有向类别对口径，32 RX 支持物理 RX 留出。需要先冻结真实 RX ID/硬件型号/日期/包覆盖和分组，按物理 RX 划分，不以包随机划分替代未知 RX。源码 loader 及目标数据访问规则尚未在本轮执行；没有提前选出最优 RX。四天重复 RX 不是128台独立硬件。不能把跨数据集验证描述成对20台异构LoRa硬件的复现。

[官方示例](https://github.com/WiSig-dataset/wisig-examples) 已提供 d003/d011_ManyRx_nrx 与训练 RX 数量影响研究。因此 receiver diversity 本身并非新发现。不自动转向候选B或C，也不在D0下载 WiSig。
