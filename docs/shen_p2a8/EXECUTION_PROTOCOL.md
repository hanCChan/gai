# P2-A8 执行协议

**已被后续用户指令调整优先级。** 2026-10-12新增USER_SUPPLIED_GENERALIZATION_UPDATE.md后，M1 30次训练未启动，部分缓存准备已终止；当前执行G1_EXECUTION_PROTOCOL.md。下文为原冻结方案的保留记录，不能当作当前启动授权。

依据用户提供的完整方案 USER_SUPPLIED_PLAN.md。审查提交1106c43与本地a02db3c的Git树一致。历史124次训练、P1结果与原始数据不覆盖。

顺序为M0→M1→决策→M2强基线→满足条件才M3。旧P2-B/P3、HSTU、复杂融合不启动。Day3/4和六台本轮锁定接收机不访问；不把历史使用过的数据称为研究史全新盲测。

M0仅访问六个已允许的n210_1/rtl_6文件；逐文件SHA、全部24000包IQ、HDF5属性、TX32/33/34与全部其他TX分布、对称频率位置、完整混淆差异均审计。跨RX只按类别比较分布，不按包序号配对。饱和、截断、噪声底只能给代理量，缺少ADC/完整捕获不能做因果确认。

M1固定658314参数Spec rank16 CNN、52×126、Adam lr=.0003/WD=0/batch64、最多30epoch/patience7/min_delta1e-4/clip5、seed20261012/13/14。12个既有Full/Strict独立scaler结果只有核对所有配方和预测后才复用。公共scaler按每RX的Day1 Train Full和Strict映射等权合并均值与方差，对两个输入完全相同；因此公共scaler12次均为新组合。局部Notch三种×2RX×3seed=18次，总新增30次。

全部Notch使用8192点原始FFT同一种硬投影再IFFT。宽度固定为15625Hz（128个FFT bins）；负位置以−101562.5Hz为参考，正位置使用同一bin集合的精确反射。随机控制位置用独立随机种子20261012，在STFT可见、标称IB外、与两目标Notch不重叠的离散中心列表中一次抽取，两RX所有seed共用。区间和mask SHA写入运行前registration.json，禁止根据结果重选。

保留既有Day2 best checkpoint/早停策略以确保12次历史复用可比；此策略仅作机制开发，不能作为独立验证。新增运行同时保存最终epoch权重与预测，区分best与last。M2模型选择必须转为训练RX分组验证；不能复用M1的Day2选优规则作泛化证据。

新缓存必须检查十类原始float64 IQ独立重算、原始FFT被移除点残差、仅被移除信息变化时输出不变、映射与label/sample_id、train-only scaler及全部四卡checkpoint/梯度冒烟。正式运行固定原环境。GPU分配可用4卡等负载，不改变配方；完整实际任务矩阵与失败保留。

H1比较局部负Notch/正Notch/随机Notch与Strict收益，不能仅凭显著位置上升宣称因果；H2比较独立与公共scaler的Strict−Full配对差；H3报告所有TX的召回、误判去向、margin、纠错和新增错误。3seed仅描述训练随机性，不当独立环境；不以packet bootstrap冒充跨域置信区间。

M2在M1机制可解释性决策后冻结10/4/6角色、源RX分组CV、训练预算、全部强基线与型号分类。多接收机普通文件缺失真实日期/session时只称cross-RX（日期未知），不宣称同日期；drift Day1/Day2单独用于时间评估。额外独立采集不存在时明确外部验证未完成，不虚构。
