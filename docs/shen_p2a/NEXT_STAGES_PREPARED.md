# 后续阶段代码准备与历史问题

P2-B/P3没有训练入口，也不自动启动。`src/shen_next` 仅包含模型与损失、训练元数据配对和增强函数；单元检查使用合成输入。配置在 `configs/shen_next/prepared_only.json`。

RF-HSTU复用原仓库的最小RFHSTUBlock，不声称完整复现官方HSTU。HSTU与标准Transformer共享相同严格IB时域IQ tokenizer：256点patch、128dim、7层；653251与653258参数，仅差7个。标准Transformer为匹配容量，FFN隐藏64，需明确这是一种容量受限的Transformer而非通常4D的FFN。CNN659370参数；分别比较主编码器和同样CrossAttn的融合结构。辅助OOB是FFT频率patch，独立频率位置参数；不把频率token解释成时间/chirp位置。CrossAttn839235/839242参数。真实执行前还需冻结等参数main-only容量控制与对称搜索预算。

GRL至少2个训练接收机，梯度翻转系数独立；Spec Siamese为相同设备不同训练接收机正对，用CE+NT-Xent。批次每设备只一对，避免同设备重复包成为负例。准备的TDL为有限支持sinc插值的分数延迟，支持1MHz假设下5–300ns（不能四舍五入成零）；噪声10–40dB和单正弦0–5Hz。单正弦不是Jakes，离散有限支持和近采样分辨率的延迟也不能声称完整论文物理复现。增强后才严格提取各频带。来源：[Ma2025](https://arxiv.org/html/2512.12070)、[ShenTMC](https://arxiv.org/abs/2207.02999)。本实现为代表性基线，不包括无监督预训练或完整作者协议。

用户提供的旧论文 `/home/keyan/code-y/oob-lunwen1/OOB_IOTJ.pdf`：V-A明确原六盲测 `b200_2,b200_mini_2,b210_2,n210_2,n210_3,pluto_2`，RQ6/TableVI已报告它们的结果。因此20台接收机均存在研究使用历史。14台原开发名称由图4及缓存文件头支持；旧六开发fold为rtl2/rtl5/b2001/b200mini1/b2101/pluto1，N2101/RTL6虽未列为fold，已进入旧信号审计/训练/预处理开发，并进入P1/P2方法筛选。

10/4/6划分可区分训练、选择和本轮锁定角色；不代表这6台在整个研究历史中未被查看。按型号前缀划分，4开发中B2001和N2101均没有同型号训练接收机，B200-mini不能自动归并为B200；RTL4/8属于训练已见的RTL组。具体硬件版本和实体仍未独立核实。日期/session/本地Fs与CFO/SNR单位未核实。论文投稿状态及旧代码/归档的哈希闭环仍未核实：当前目录只有PDF，无旧配置与归档，不能把论文文字当作实际运行日志。本轮没有重新打开任何锁定接收机HDF5。真实独立研究留出需额外新数据及另行授权；现阶段不下载OSU。

融合容量控制也已实现：CrossAttn辅助改为原始IB频域（信息源相同、架构及参数量完全相同）；main-only增加hidden723的池化残差MLP，HSTU839190 / Transformer839197参数，分别比同encoder的OOB CrossAttn少45个（0.006%）。后者仅匹配总容量，函数结构不同，不能取代前者。以上只合成检查，无真实训练。
