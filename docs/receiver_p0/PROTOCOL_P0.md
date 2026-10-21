# P0 协议:设备对判别失败定位 + 三分支 PoC(2026-10-21 冻结)

范围:P0-A/B/C(纯推理)、WiSig mirror(纯推理)、PoC(小网格训练,无新骨干、无新数据)。
基线提交:`22e43743`(D 阶段完成)。分支:`experiment/p0-pair-margin`。
数据边界(不变):Shen 仅 Strict CIS 缓存(`shen_p2a8/g1_cache_v1`),原始 IQ 不读;
WiSig 仅开发缓存(源 14 RX + 内部 10 RX,日期 0–1,位置 0–199),最终 8 RX / 日期 2+ / 等化版不访问。

## 1. P0-A:全部有序对的 margin 矩阵(30 个冻结 Shen 模型)

- 模型池:R0/R1 formal 30 个(2 折 × 5 方法 × 3 seed),checkpoint 哈希核验,禁止任何重训。
- 池定义:source_train = 每折源 RX(inner_train+inner_validation)的 first160/包
  (与拟合数据完全一致);source_val = last 40(选模包,仅记录);
  outer = 5 台外层 RX 全部 2000 包。
- 逐 (RX, t, u) 计算:margin mean/q10/q50、frac_neg、有向混淆率 conf_{t→u},
  外加整十分类正确率(防第三类误导读)。90 个有序对全算,Pluto 1→0/3→2/8→7 仅作历史案例对照。
- 同一前向取 512 维归一化特征 + 128 维分类头前向量(logits 与 model(x) 逐位核验一致)。

## 2. P0-B:编码器 vs 分类头分解

- 特征空间:逐 (RX,t,u) 的 d_class(类间均值距/类内离散)、有效秩(512 与 128 维)。
- 头几何:||w_t−w_u||、bias 差。
- 源域重拟合对照(无网络训练):用 r1/L0 的 refit 配方(steps=500, lr=3e-3, l2=1e-4,
  固定 seed 20261019)在 source_train 的 512 维特征上拟合全新 Linear(512,10) 头,
  在 outer 上重测 margin/混淆。分类:若重拟合头同样塌缩 → 特征已不可分;
  若重拟合头大幅缓解 → 冻结头边界过拟合源接收条件。
- 解释禁令:rank-16 瓶颈下的有效秩 3.4–3.8 不得解释为"RF 指纹只有 3.5 维身份信息"。

## 3. P0-C:源域风险可预测性(外层 RX 标签仅作诊断,不进任何规则)

源风险(仅 source_train margins,CE 与全方法两版都算,冻结规则用 CE):
- R_q = −(seed 均值的源 RX q10 中位数)——最差 margin 风险
- R_spread = max−min 源 RX q10——跨源 RX 不稳定度
- R_neg = max 源 RX frac_neg——源域已有翻转
- R_conf = mean 源混淆率——普通源域难度基线
外层失败:conf_{t→u} 对外层 RX/seed/方法平均。
统计:逐折 Spearman(90 对,置换检验 20000 次)、偏相关(对 R_conf)、top-10 重叠率、
seed 稳定性、两折一致性。**不得把 90 个相关对当独立样本做参数检验推断。**
预注册判据:
- 若 R_q(或 R_neg)与外层失败 ρ ≥ 0.5 且两折同向且偏相关 > 0 → "源域可预测"成立,
  允许进入针对性 PoC;
- 若仅 R_conf 显著而偏相关 ≈ 0 → 困难对只是"本来就难",target-free 算法缺依据;
- 若两折不一致 → 结论限 fold 0。

## 4. 困难对集 P 的冻结(select_pairs.py)

CE 3 seed 的 source_train margin,每折取:
top-K=8 by R_q ∪ {R_neg ≥ 0.05 的全部有序对}(方向独立保留)。
P 在 PoC 训练前一次性冻结;outer 数据不参与。K=8 的选择依据:SWAD 在 Pluto 上
塌缩的 4 个单向对(两方向 8 个)的量级,先于 PoC 结果定死;敏感性由 PoC 网格的
lambda 与分支消融承担,不再回头改 K。

## 5. WiSig mirror(6 个 CE 冻结模型:official 3 + resnet 3)

- 池:源 14 RX 日期 0–1 位置 120–199(R0/R1 开发池,L0/D 评估池);
  内部未见 RX 日期 0–1 位置 0–199。日期 2、final 8 RX 不访问。
- 同 P0-C 风险定义与统计。WiSig 与 Shen 的现象只作"同一判别风险框架"的对照,
  不预设共同物理成因(频响增强叙事已被 D1 否定,不复活)。

## 6. PoC:三分支小网格(预注册,全部运行,不依据 P0 结果挑分支)

训练配方与 R0/R1 逐位一致(balanced_schedule 同 seed 生成、Adam 3e-4 eps 1e-8、
clip 5、20 epoch、deterministic、source-only receiver-equal val CE 选 checkpoint)。
损失:L = CE + λ·L_branch,分支:
- **sub**(特征):逐源 RX 的 P 对原型间距 softplus(γ_sub − ||μ_t−μ_u||/(1+√within));
- **drift**(头):Δ_{t,u,r}=(w_t−w_u)ᵀ(μ_{t,r}−μ̄_t) 投影约束 softplus(γ_d+Δ);
  只约束判别方向上的 RX 偏移,不约束全分布对齐;
- **pair**(联合):跨源 RX 的 P 对低分位 margin 稳定性,批内 per-(r,t,u) 取 min,
  logsumexp 聚合最差 RX。
网格:λ ∈ {0.1, 1.0} × 3 branch × 2 fold × 3 seed = 36 fits;
另加 CE 锚点 λ=0 重跑 2 fold × 3 seed = 6 fits(验证配方复现);
共 42 fits ≈ 42 × 2 min ≈ 1.4 GPUh(上限 2 GPUh,超则先停)。
**CE 锚点必须先通过复现核验(同 epoch 选择、历史 CE 轨迹一致)后,PoC 对比才可解读。**

预注册假设(全部报告,不做选摘):
- H1(pair 分支主假设):pair 在 P 对上的外层 conf 显著低于 CE 锚点(逐对 paired,
  3 seed),且平均外层精度不劣于 CE 锚点 −0.5 pp;
- H2(机制假设):若 P0-B 判定"特征已塌缩"(refit 头复现塌缩),则 sub ≥ pair;
  若"头边界漂移"(refit 头缓解),则 drift ≥ pair;若两者均轻,pair 可能无增益;
- H3(drift 特异性):drift 的增益若与 sub 相当,则"判别方向投影约束"是对全分布
  对齐的充分替代(与 CIDG/ConstStyle 的区分点);
- H4(负结果条款):任何分支若 P 对 conf 无改善或平均精度掉 > 1 pp,该分支出局,
  下阶段不再以该机制立项。
统计口径:P 对 conf 的 seed-mean paired 差、per-RX 分解、平均精度、塌缩对数
(conf>0.3)。不做多重比较校正的探索性 p 值一律标注 exploratory。

## 7. 停止条款

P0+PoC 完成即停,产出决策表;不启动 D3/E0/E1/CRR 任何新诊断。
若 CE 锚点复现失败 → 停,先查配方再谈 PoC。
若三分支全部触发 H4 负结果 → 下阶段不立项任何 pair-loss 方法,转"可预测性边界"
写作(源域风险 → 外层失败的映射本身即结论)。

## 8. 产物

outputs/receiver_p0/{p0a,p0c,p0_wisig,pairs,poc,poc_eval}/ + reports/receiver_p0/REPORT.md。
代码 SHA 记录于各 manifest;历史报告不覆盖。
