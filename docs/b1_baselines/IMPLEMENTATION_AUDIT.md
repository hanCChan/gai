# B1实现核对与已知范围

四标准目标数学独立重写，参照固定SHA的Group DRO/SupCon作者实现与DANN/GRL论文，来源见reference_code_provenance.json。没有复制作者大模型或新方法名称。

- CNN包装只返回原512维归一化feature；合成输入验证逐参数初始化与原FactorizedAuthorCNN、forward logits完全一致，推理参数658314。
- GRL反向只使feature梯度乘−0.1；head梯度仍为正常CE。合成闭式梯度通过。头输出数对应当前拟合RX4/5，不包含验证RX。
- DRO对每batch全部group均值CE作乘性权重更新，log-space与原q·exp(ηloss)归一化等价，q脱离梯度。合成2group权重、梯度与累计更新通过，不启用作者可选loss normalization/adjustment/BTL。
- SupCon采用一视图同TX全positive、排除self、τ=.1/baseτ=.1，与标准目标一致；无投影网、hard-negative规则或特殊类对。独立NumPy逐anchor公式、梯度有限性与尺度归一化检查通过。
- 每epoch每包一次，每RX×TX每batch2包，4方法/2LR共享同stage/fold/seed的预存batch文件，六套schedule预检覆盖/分组/固定seed一致性。
- 固定10/20训练；search进程仅打开当前4训练RX/1内部验证RX。正式LR来自冻结选择SHA，外层RX仅训练结束后评价。所有输入hash与不可变code seal在first优化前确认。

这些是标准同架构机制基线，不是Shen完整GRL网络、Ma Siamese或IB-RFF FAN+IB复现。数据是历史已使用Shen源池、正式仅两seed；现有强方法原论文级充分性仍需后续另批B2。不得把本轮负结果概括为所有对比/最差组/IB理论无法解决。
