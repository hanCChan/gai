# 接收条件变换共享性检验与条件 PoC

2026-10-13，北京时间。用户授权下载 WiSig，并在共享性通过后执行 PoC；进一步明确留出 TX 必须同时从 CNN 和变换拟合中排除。历史 B1/M0–M2 不变，本分支继承其记录，不重新执行。

研究对象是接收条件的统计变换，不是可识别的纯 RX 硬件失真。未配对的包不使用逐包跨 RX MSE。不因结果修改划分、秩、正则、门槛或种子。

## 下载与来源

停止旧配额轮询，保存全部日志和现有文件。核查 Kaggle v1 文件清单，只下载 `ManyRx.pkl/ManyRx.pkl` 的公开单文件 ZIP，避免下载 10.4 GB 全集。原始官方数据采用 CC BY-NC-SA 4.0；Kaggle 列出的许可为 Unknown，不能据此获得不同使用权。第三方与官方字节一致性没有官方校验和证明。校验文件名/字节数、ZIP CRC、SHA256 和隔离的结构元数据；不执行任意 pickle 类，不分析 WiSig 数值，不上传信号。Kaggle 成功时不再请求配额受限的官方文件；失败才执行一次标准 gdown。Full WiSig 扩大下载范围需要先列清单和估算。

## 有限共享性实验

仍只使用封存的十个 Shen source RX、每 RX×TX 200 个 G1 原始行。严格两折，每折五个 fitting RX、另五个 outer RX；共享性检验不打开该模型的 outer RX。

固定五组留出身份：31/32、33/34、35/36、37/38、39/40。两种子 20261013/20261014，共 20 次 IQ CNN CE 训练。每模型只训练八个身份，留出两身份从 encoder 优化和变换拟合同时排除。

每个 fitting RX×TX 的原封存类内顺序分为前 120 个 encoder 包、后续 40 个 mapping 包和最后 40 个 gate 包。编码器为固定小型 1D CNN，32 维特征，原始 8192 复采样点、仅逐包 RMS，不使用 CIS。每轮每组均衡抽取两包，全部训练包使用一次。固定 20 轮 Adam LR=0.001、梯度裁剪5，没有 LR 或系数搜索，没有 early stopping、重试或 seed 替换。此协议独立于先前 B1 的 CIS 训练协议。

变换在每模型的五 fitting RX 内全部 20 个有向 RX 对上估计。比较恒等、均值平移、标准收缩协方差 CORAL、固定 rank2 的 ridge 低秩残差仿射。低秩残差由八 TX 的类均值估计，ridge 系数为原型矩阵总能量的0.1/维数；不能假称这是一个原创方法。特征使用仅 fitting/mapping 数据确定的一个标量尺度，禁止 gate 决定尺度。

主指标为留出 TX 的类均值平方距离和 Gaussian W2（含类协方差）分布距离。每一个 fold×seed 层分别等权汇总五身份划分、20 RX 对和两留出 TX。预固定通过条件：低秩模型对两个指标均比该层最强恒等/平移/CORAL 至少减少5%，每层至少三组 TX 划分改善；同时十类目标原型最近邻身份检索比最强控制下降不超过5个百分点，各 encoder 最终训练准确率至少50%。全部四层通过才启动分类 PoC。此门槛是工程决策规则，不是显著性或泛化定理；有向 RX 对非独立，不把包数作为独立重复数。

失败时发布真实结果、来源、训练代价并停止当前 rank2 机制；不据此否定所有非线性变换。若通过，再封存一次四方法×两折×两种子的16次固定20轮 PoC：IQ CE、MixStyle、标准 RX×TX Group DRO、已验证结构的变换增强。所有方法共同初始化、批次、输入和优化器，outer RX 仅作最终评估，不拟合变换。发布 Accuracy、Macro-F1、各 RX 最差类平均风险、低分位 Recall 和全部 RX×TX 单元。正式训练的变换构造和参考 encoder 必须在读取 outer 结果前封存；不通过共享性时不执行此条件阶段。

## 文献与主张边界

Pan 等的最新 v2（2026-05-26）描述 DRIFT 的解耦、GRL、接收机中心约束；附件称其为普通 Style Transfer 不够精确。Zhang 等 RIEI 已研究接收机/发射机解耦；Chai 等 RAE 使用 frozen backbone 上的 MAML 与 AFA。我们检验的是未见身份上的受控统计映射，不能仅凭增加低秩映射声称原创或突破现有论文。低秩模型失败就不推进新算法。

参考：[Pan v2](https://arxiv.org/html/2510.09405v2)、[RIEI](https://arxiv.org/html/2411.03636v1)、[RAE](https://www.sciencedirect.com/science/article/pii/S157087052600065X)、[MixStyle](https://arxiv.org/abs/2104.02008)、[官方下载](https://cores.ee.ucla.edu/wisig/downloads/)、[官方加载工具](https://github.com/WiSig-dataset/wisig-examples/blob/master/data_utilities.py)。
