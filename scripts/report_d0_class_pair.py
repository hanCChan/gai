#!/usr/bin/env python3
"""Render results and package all numeric/figure evidence, without score reads."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reports/d0_class_pair/2026-10-13_D0'
def table(frame):
    def cell(x): return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,frame.columns))+' |',
                     '| '+' | '.join(['---']*len(frame.columns))+' |']+
                    ['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)])
def main():
    d=json.loads((P/'decision.json').read_text());e=pd.read_csv(P/'risk_prediction_evaluation.csv');r=pd.read_csv(P/'per_RX.csv');t=pd.read_csv(P/'per_RX_TX.csv');h=pd.read_csv(P/'replicated_harmful_edges.csv');m=pd.read_csv(P/'worst_RX_pairs.csv')
    paired=t.pivot(index=['receiver','fold','hardware_family','tx'],columns='method',values='recall');paired['delta']=paired.strict_cis-paired.full_cis
    cells=pd.DataFrame(d['predictability_cells']);cells['median_capture']=cells.median_gain_over_uniform+10/90
    controls=e[e.predictor=='pairwise_mean'].groupby(['method','fold','control'])[['spearman','top10_error_capture','gain_over_factorized']].median().reset_index()
    q=r.pivot(index=['receiver','fold','hardware_family'],columns='method',values=['accuracy','worst_tx_recall']).reset_index()
    q.columns=['receiver','fold','hardware_family','Full_accuracy','Strict_accuracy','Full_min_TX_recall','Strict_min_TX_recall']
    artifact_table=[]
    for name in ['input_provenance','per_RX','per_RX_TX','confusion_counts','directed_pairs','unordered_pairs','strict_full_pair_changes','replicated_harmful_edges','worst_RX_pairs','risk_prediction_evaluation','risk_prediction_vectors','predictability_cells']:
        artifact_table.append({'file':name+'.csv','rows':len(pd.read_csv(P/(name+'.csv')))})
    text=f'''# D0 类别判别结构与文献缺口审计

日期：2026-10-13 UTC。决定：**{d['status']}**。源池内描述性 screen 通过，建议仅有条件开展一次强基线反证；未批准新算法研发/训练。D0完成后停止。

本轮新训练0次、新推理0次、数据下载0 GB；OSU、开发/锁定RX及原始IQ读取0。仅复用二十份源RX归档预测：20,000个RX内包，Full/Strict共40,000条分类输出。与历史C0时序问题不同，本轮不需要真实连续时间，不重启其停止路线。[当前计划](../../../docs/d0_class_pair/EXECUTION_PLAN.md)、[用户方案](../../../docs/d0_class_pair/USER_PLAN.md)、[预登记](../../../docs/d0_class_pair/PREREGISTRATION.md)、[冻结JSON](../../../configs/d0_class_pair/registration.json)。

## 1. 可重复损害：源池内有，跨seed与独立新硬件尚未验证

RX均衡平均Accuracy：Full **68.37%** → Strict **82.99%**，提升**14.62 pp**，十台RX的平均Accuracy均提升。100个RX×TX单元中 **{int(sum(paired.delta<0))}** 个Recall下降。折0最差类别普遍改善；折1出现严重类别损失，两组CNN的结论并不完全一致。

{table(q.round(4))}

![平均Accuracy与最差类别](accuracy_and_worst_class.png)

冻结规则要求“同一有向混淆率增加在两折都出现”。满足的是**8/90**条边，所有边均列于CSV，没有根据TX36特殊筛选。下表包含净均值改善的边：跨RX符号重复不能替代跨RX净损害，正Δ为恶化。

{table(h[h.both_folds_harm].round(5))}

其中5条边的十RX平均Δ混淆率为正，另外3条平均改善。TX35→32是较强例子：6台RX损害，跨两折且包含B210；十RX均值增加21.8 pp。但其主要严重失效集中在折1（同一个Strict CNN），折0只有一台出现该边的轻度增加。B210源RX上TX35 Recall由90.5%降至8.5%，其所在RX平均Accuracy仍由80.05%升至83.75%。这是真实平均收益掩盖类别损失的例子，不能当作跨seed、所有硬件的规律。

**margin反证**：按冻结的逐包RMS归一化间隔统计，90对中“最差RX平均margin为负”的数量由Full的10对降为Strict的1对；总体判别结构大部分得到改善，并非所有类别被鲁棒化损坏。最差值Full为−0.8411，Strict为−0.7371。逐RX margin均值/中位数/q05、排序违反与混淆分别报告；平均margin为正也可能有错误尾部。原始logit尺度不同，因此不以原始margin变大作为信息保留证明。全部90对、两折/硬件族的最差margin见ZIP与CSV。

## 2. 类别对风险可预测：内部源RX留出有信息，独立部署仍未知

主评价是同CNN五个评价RX中的留一，其余四个RX风险profile预测probe的90个有向混淆率。五RX均未参与该CNN拟合；但这是历史已使用源池内诊断，不是新锁定RX实验。probe标签仅用于最后计算错误风险，不参与risk profile构造。

{table(cells[['method','fold','receivers','median_spearman','median_capture','median_gain_over_factorized']].round(5))}

固定10/90条边的均匀预算预期错误捕获11.11%。四个method×fold cell的pairwise预测中位相关均为正，捕获66.32%–86.75%的probe错误质量，超“类别难度×错误去向”对照的配对中位增益为21.56–56.78 pp。增益按同probe相减后取中位，**不是两个中位数相减**。tie按比例分配，避免有大量零错误边时按TX编号任意破tie。

![固定预算风险预测与对照](risk_predictability.png)

硬件控制与跨模型迁移（后者有模型混杂，非主结论）：

{table(controls.round(5))}

不同硬件参考控制仍提供正相关和正检索增益，因而此信号不完全依赖“七台RTL互相预测”。但折1的RTL probe仅剩一台B210参考，非RTL各硬件型号也仅一台；不能据此证明普遍硬件不变性。完整逐probe、class-only、factorized和uniform对照都保留，未只报有利probe。

统计单位限制：不把200包/类别或90条边当作独立硬件重复；不做包级p值/显著性、也不把共享CNN的多个RX视为独立训练重复。只有两个拟合折、一个seed。Spearman/检索预算为描述性而非确定的部署成功概率；仍可能反映TX本身的困难和样本选择，不代表可因果干预的信号机制。

## 3. 文献是否已覆盖：宽泛动机已覆盖，剩余实证缺口 UNKNOWN

[完整差异矩阵](../../../docs/d0_class_pair/LITERATURE_MATRIX.md)逐项记录优化目标、类对分析、最差类、类条件/对目标、目标数据可见性、代码和直接关系。IB-RFF已明确讨论过度对齐与判别性损害，Ma跨RX同TX对比、SupCon/ArcFace、Group DRO、RIEI解耦都是直接竞争。CE本身就对所有竞争类别的margin做log-sum-exp优化；单加margin项、类别难度权重、RX×TX最差CE都不能直接称新贡献。

本轮没有上述强方法的同协议逐包预测，**没有证明它们已充分解决，也没有证明它们解决不了**。ICC SupCon+Wasserstein、IB-RFF、RIFLE全文未取得的部分明记未知。不能把缺少逐类表格等同于算法没有保护类别，也不能据此宣布新颖性排清。

## 4. 最小下一步与成本：只提交有条件预算，不执行

[基线验证草案](../../../docs/d0_class_pair/NEXT_BASELINE_POC_DRAFT.md)为6种目标机制×2折×3seed，共36个最终模型/42个阶段作业，统一总40轮，共1440 model-epochs。包含CE、SupCon、ArcFace、RX×TX DRO、GRL、HSIC IB surrogate。历史20轮小CNN作业146.54–150.04秒，线性吞吐估计约3小时；考虑新目标开销，建议3–9 GPU小时的条件预算，需首作业计时确认。FAN原IB/其他结构完整复现需另核成本；CNN-HSIC不能冒称IB-RFF原方法。当前新增训练仍是0次。

[WiSig metadata核查](../../../docs/d0_class_pair/WISIG_FEASIBILITY.md)：174TX/41RX/四次采集；候选ManyRx压缩1.2GB，子集不完全均衡并与其他子集有重叠。未下载、未选RX、未运行。其用途是未来独立验证，不能以重叠子集或日期重复冒充独立硬件重复。

## 资产核验、反证与复算

20份NPZ逐文件SHA通过；4个固定epoch20权重仅计算哈希、不加载；配置证明源训练/评价RX分离、scaler只拟合训练RX、初始权重一致。Full/Strict同RX rows/labels严格一致并对应G1索引，每TX200包；概率与softmax一致。跨RX相同行号不被当作同一空口包。来源为C0已核验资产，路径和SHA见input_provenance.csv及冻结JSON。

分别训练的Full/Strict CNN和前端/scaler共同改变，D0不直接测GRL/IB；差值无法作为“接收机对齐破坏身份”的因果证据。固定20轮checkpoint不是根据D0最佳结果挑选。历史训练曾记录评价折validation结果，固定final权重避免最佳epoch选择，但不能使源池成为全新封存测试。

独立复算不导入D0 metrics模块：**14,400个标量指标、240个risk profile**通过，包含直接重建confusion、独立rank相关、tie预算及所有参考集排除probe。封存SHA通过，详见independent_verification.json。首次运行在SciPy返回字段处失败、未写结果；API兼容修正与新增synthetic case记录在pre_execution_seal.json，original_pre_execution_seal.json保留。Matplotlib tick API兼容修正只影响画图，不改结果或冻结指标。

CPU主诊断耗时1.78秒（不含浏览文献/画图/独立验证）；Python3.9.7、NumPy1.20.3，具体环境见execution.json。源码包含synthetic数值检查、独立验证、图表和报告打包脚本。每条输入和全部派生证据可追踪；未上传私有稿件/拒稿信、原始IQ、权重或原始NPZ。

完整数值与六套PNG/SVG图表见 **[evidence.zip](evidence.zip)**；每个成员SHA/行数见evidence_index.json。大型逐对表/风险向量/SVG集中存于ZIP，小汇总CSV直接公开，避免GitHub预览截断。解压到此目录后可检查ZIP内每个SHA；依赖预测资产的完全复算另需本地registration指定文件。

{table(pd.DataFrame(artifact_table))}

**最终反证结论**：当前“某个TX的偶然异常”“平均提升意味着所有类别改善”“所有pairwise信息都由类别难度解释”“仅RTL互相预测”这四种简单解释不完全符合源池数据；“跨seed可重复”“强方法后的剩余缺口”“可发表新算法”“独立WiFi/LoRa硬件迁移”和因果机制均未证实。采用有限正证决定，完成D0即停止，不自动开始新Loss/Attention/Gate/CNN。
'''
    (P/'REPORT.md').write_text(text)
    files=sorted([x for x in P.iterdir() if x.suffix in {'.csv','.png','.svg','.json'} and x.name not in {'evidence_index.json'}])
    with zipfile.ZipFile(P/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in files:
            info=zipfile.ZipInfo(path.name,date_time=(2026,10,13,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    index={'archive_sha256':hashlib.sha256((P/'evidence.zip').read_bytes()).hexdigest(),'members':[{'name':x.name,'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in files],'raw_predictions_included':False,'manuscripts_or_weights_included':False}
    (P/'evidence_index.json').write_text(json.dumps(index,indent=2)+'\n')
    with zipfile.ZipFile(P/'evidence.zip') as z:
        assert z.testzip() is None
        for x in index['members']:assert hashlib.sha256(z.read(x['name'])).hexdigest()==x['sha256']
    print('Report and full evidence archive verified:',len(files),'members', (P/'evidence.zip').stat().st_size,'bytes')

if __name__=='__main__':main()
