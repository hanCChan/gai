#!/usr/bin/env python3
"""Render frozen results, readable reports, figures, and a public evidence ZIP."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'reports/d05_b0/2026-10-13'
def table(frame):
    cell=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in frame.itertuples(index=False,name=None)])
def save(fig,name):
    for ext in ('png','svg'):fig.savefig(P/(name+'.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig)
def main():
    d=json.loads((P/'d05/decision.json').read_text());b=json.loads((P/'b0/decision.json').read_text());cost=json.loads((P/'b0/execution.json').read_text())
    r=json.loads((ROOT/'configs/d05_b0/registration.json').read_text());profile=pd.read_csv(P/'d05/profile_evaluation.csv');summary=pd.read_csv(P/'b0/seed_summary.csv');changes=pd.read_csv(P/'b0/recall_changes.csv');repeat=pd.read_csv(P/'b0/cross_seed_repetition.csv');optimization=pd.read_csv(P/'b0/optimization_status.csv')
    figs,axes=plt.subplots(1,2,figsize=(12,4))
    methods=r['d05']['predictors']
    for f,ax in enumerate(axes):
        values=profile[(profile.fold==f)&(profile.control=='same_paired_cnn_loro')].groupby('predictor').positive_mass_capture.median().reindex(methods)
        ax.barh(methods,values.values,color=['#dc6c34' if m=='signed_delta_mean' else '#487eb0' for m in methods]);ax.set_xlim(0,1);ax.set_title(f'Fold {f}: top10/90 positive-damage capture');ax.set_xlabel('Median fraction of positive delta mass');ax.grid(axis='x',alpha=.2)
    figs.tight_layout();save(figs,'d05_damage_prediction')
    figs,axes=plt.subplots(1,3,figsize=(15,4))
    order=[rx for fold in r['folds'] for rx in fold]
    for ax,seed in zip(axes,[r['original_seed']]+r['new_seeds']):
        matrix=changes[changes.seed==seed].pivot(index='receiver',columns='tx',values='delta_recall').reindex(index=order,columns=range(31,41))*100
        im=ax.imshow(matrix,cmap='RdBu',vmin=-85,vmax=85);ax.set_title(str(seed)+'\nStrict - Full Recall (pp)',fontsize=11);ax.set_xticks(range(10));ax.set_xticklabels(range(31,41));ax.set_yticks(range(10));ax.set_yticklabels(order,fontsize=7)
    figs.subplots_adjust(wspace=.35)
    figs.colorbar(im,ax=list(axes));save(figs,'b0_all_recall_changes')
    figs,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,column,title in [(axes[0],'train_loss','Training CE'),(axes[1],'train_accuracy','Training accuracy')]:
        history=pd.read_csv(P/'b0/training_history.csv')
        for (seed,fold,method),values in history.groupby(['seed','fold','method']):
            ax.plot(values.epoch,values[column],label=f'{seed%100}/F{fold}/{method}',alpha=.7)
        ax.set_xlabel('Epoch');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].set_yscale('log');axes[1].legend(fontsize=6,ncol=2);figs.tight_layout();save(figs,'b0_optimization')
    controls=pd.read_csv(P/'d05/summary.csv');primary=pd.DataFrame(d['primary_summaries'])
    gain_columns=['median_capture_gain_over_recall_drop_only','median_capture_gain_over_class_difficulty_full','median_capture_gain_over_factorized_delta','median_capture_gain_over_absolute_full_confusion','median_capture_gain_over_absolute_strict_confusion']
    d05_text=f'''# D0.5 增量损害审计

决定：**{d['status']}**，仅历史seed20261012的源池探索。原D0未改标准/结果。signed Δ=Strict−Full有向混淆概率差；正向质量取max(Δ,0)，净变化与改善质量另报。不是同包新错转移率，Full→Strict包含前端、scaler和分别训练的CNN，不是GRL/IB因果证据。

同一对Full/Strict CNN的五个评价RX，四RX形成参考，probe只做评价。8个预测对照、90有向边、全部十RX，两fold与硬件控制均公开，不特别选择TX35/36。

{table(primary[['fold','median_signed_spearman','median_positive_mass_capture']].round(5))}

signed mean主预测在两折Spearman为0.41584/0.34315，Top10捕获正向质量33.33%/93.79%。均匀预期11.11%，但两折差异很大。

逐probe配对capture差值的中位数（pp前须乘100）：

{table(primary[['fold']+gain_columns].round(5))}

比Recall下降/Full类别难度/Δfactorized的中位增益两折为正，故冻结的有限screen通过。但fold0 **比Full绝对混淆低35.48pp、比Strict绝对混淆低48.02pp**，fold1仅多1.24/0.62pp。不能宣称Δ预测全面优于原错误结构。所有基线全报，不把较弱类难度对照当作最强竞争者。

{table(controls[['fold','control','median_signed_spearman','median_positive_mass_capture','median_capture_gain_over_uniform']].round(5))}

跨另一对CNN折的signed Δ参考捕获率仅3.23%/0.58%，两折均**低于uniform**；虽然signed Spearman仍正，排序对正向损害的实际预算检索失败。这是关键反证：风险与训练模型/拟合折有关，共享CNN内跨RX可预测性不能替代跨拟合模型迁移。这项控制同时更换参考RX组与模型，参考CNN本身曾用probe所在RX组训练；它是历史源池的混杂压力诊断，不是所有模型都未见probe的独立验证，不能把下降唯一归因为模型参数。不同硬件参考控制不是只用七RTL互相预测，但单例硬件和少参考RX使外推仍有限。

结论：新增损害有内部信息，但不支持把“专门预测鲁棒化新增伤害”的优越性当作算法核心。保留一般source-predictable class-conditional failure问题，先用B0/B1检查尾部风险与普通方法是否足够。没有独立新硬件、没有packet级p值、90边不作90个独立硬件重复。

独立复核240个profile、21,600条向量，直接重建confusion、参考集、baseline、signed correlation、正质量及ties通过；[verification](independent_verification.json)。完整900有向Δ、240评价、全部向量存于本轮evidence.zip。定义/源码封存见configs/d05_b0；初始seal与启动修正后的seal均保留。
'''
    if (P/'d05/numeric_tie_sensitivity.json').exists():
        d05_text+='\n数值敏感性补充：[12位舍入后的ties核查](numeric_tie_sensitivity.json)，不改冻结主指标或通过标准。240个组合中个别capture最大变化6.45pp，主fold的capture中位数及基线增益方向均不变；signed Spearman约0.41579/0.34292。不能依赖浮点微差来解释个别小幅次要差异。\n'
    (P/'d05/REPORT.md').write_text(d05_text)
    repeated=repeat[repeat.noncollapsed_severe_seeds>=2]
    b0_text=f'''# B0 三seed类别退化复验

决定：**{b['status']}**。固定20轮、Full/Strict×2fold×2新增seed，共8次实际训练完成；原20261012保留，共12个模型、三seed。没有重挑seed，没有训练重跑，没有最佳epoch选择。B1/B2训练均未开始。

{table(summary.round(5))}

Strict三个seed平均优势为**+14.620/+18.885/+13.785pp**。每seed严重RX×TX损害（Recall下降至少10pp）为9/6/14个；最严重Δ为−82/−29.5/−75pp。全RX×TX最低Strict Recall分别0.5%/50.5%/4.5%，对应Full11%/8.5%/0.5%。因此两个新增seed的总体最差类也改善；“每seed最差类都被牺牲”不成立。

至少2/3seed重复≥10pp损害的单元9个，跨7台RX、两个fold，均来自非collapse模型；其中**B200 mini TX33与RTL6 TX32在三个seed都下降≥10pp**。全部重复单元如下，不能把同折RX视为独立训练次数。

{table(repeated.round(5))}

TX35的巨大下降主要在20261012/14出现，在20261013可消失或反向改善。失效幅度和身份有seed依赖，不能只复述旧seed的灾难案例。整体收益与局部可重复损害并存，支持B1对现有方法进行反证，不支持直接写新Loss。

![全部RX×TX三seed变化](../b0_all_recall_changes.png)

模型优化诊断：12模型固定20轮、历史240行完整；无非有限损失，final train Acc范围{optimization.final_train_accuracy.min()*100:.2f}%–{optimization.final_train_accuracy.max()*100:.2f}%，无冻结的collapse flag。Full常接近100%训练Acc而源留出Acc较低；Strict训练误差仍高于Full，受表征/优化共同影响，不能纯归因于域对齐。新seed14 Full最终损失高于另两seed，保留其曲线，不能删掉以求好看。

![优化曲线](../b0_optimization.png)

新8个真实final.pt独立加载核验：config/epoch/源scaler一致，Adam每参数**3140次更新**，与ceil(10000/64)×20一致。相同seed四模型的初始payload完全配对；两个新seed初始化不同。独立重建60个model×RX预测、600个Recall、6000格confusion、全部严重重复率通过。历史分数由CPU缓存，新训练保存GPU末轮float32转float64输出；没有重复CPU推理，微小浮点边界不是同设备逐位复现声明。[核验](independent_verification.json)。

实测V100：八正式进程94.98–97.48秒，各20轮；训练内部status约90.62–92.51秒，其余初始化/来源核验/退出；四GPU两个队列轮次。全部正式进程加初始失败占用的保守累计 **{cost['reserved_gpu_hours']:.5f} reserved GPUh**，在1小时上限内。它包含读取/评价，非纯CUDA kernel时间。保存了80,000个新增最终logits/标签/row索引（40个model×RX NPZ）；GPU每epoch源评价仅监测，不用来决策。权重与完整NPZ留服务器，不公开上传原始预测。

工程启动失败必须保留：首次8进程因我错误地把历史整个目录字节哈希要求一致，在创建优化器之前退出，优化步数0、无history/final。EISR已记录的g1_data新增访问校验函数和未导入shen_p2a.runner例外未被复用。修正为历史5661a32字节+所有旧函数AST相同、unused module不被导入的证明。保留所有初始status/log、original registration/seal/两份原源码；后续8个获准拟合任务用同seed/同配方完成，实际拟合8次、拟合重试0、换seed0。初始51.19 reserved-GPU秒纳入成本。科学阈值/样本/指标没有更改；见ENGINEERING_CORRECTION.json。

源码、历史日志、模型SHA、scaler/indices校验、逐RX/逐TX/全部5400方向margin、2700方向变化、优化状态与严重重复率均存证据包。没有OSU、开发/锁定RX、Day3/4或原始IQ访问。全部为历史源池探索性研究，不是独立盲测。
'''
    (P/'b0/REPORT.md').write_text(b0_text)
    combined=f'''# D0.5 + B0 最终交付与Stop/Go

已完成用户授权：D0.5、B0限额8次训练、B1实施审核矩阵、作者GRL/FAN源码与适配成本审计；**完成后停止，B1/B2/新Loss未执行**。

1. **新增损害比普通类难度可预测吗？** 同CNN对内有：signed Spearman0.416/0.343，超Recall/类难度/factorized对照。但fold0明显不如绝对混淆，跨CNN正向检索低于均匀。不能把Δ预测器优越性当研究主张。[D0.5完整报告](d05/REPORT.md)。
2. **严重退化跨seed重现吗？** 有有限复现：三seed平均收益均正，9个单元≥2/3seed下降≥10pp，跨7RX、两折，无collapse；仅2单元三seed都严重，灾难幅度有随机性。[B0完整报告](b0/REPORT.md)。
3. **简单/强方法能否解决？** 仍UNKNOWN。B0全是历史CE；D0.5预测baseline不是新的神经训练baseline。[B1审核计划](../../../docs/d05_b0/B1_REVIEW_PLAN.md)给出16个内部搜索+16正式模型，32作业、480轮，估0.7–2GPUh/建议总上限2；尚无许可、无训练结果。
4. **足够支撑独立算法论文吗？** 尚不足。局部风险值得受控强基线反证，但source Δ跨拟合模型检索失败是重要限制；SupCon/DRO/GRL/IB等直接竞争未排清。暂不实现新pairwise loss。[作者方法核查](../../../docs/d05_b0/AUTHOR_IMPLEMENTATION_AUDIT.md)区分同架构机制与论文级方法，FAN/LoRa输入和完整成本待审计/实测。

投入：8次实际训练，8次零步工程启动失败原样保留；新增最终分类输出80,000；累计reserved GPUh {cost['reserved_gpu_hours']:.5f}。只读十源RX CIS缓存与归档分数，数据下载0GB，开发/锁定/OSU/Day3/4/原始IQ访问0。原D0和所有历史负结果未改。

独立复核：D0.5 240 profile、B0 60 model×RX/600 Recall/6000 confusion以及8个真实checkpoint/3140 Adam步；全部通过。B1仅有可执行配置生成和约束检查，四目标训练runner未在本轮添加，不能称B1已跑通。

公开证据：[evidence.zip](evidence.zip)，包含所有完整派生CSV、图表、逐epoch日志/配置、初始失败记录及每成员SHA。大型向量/margin放ZIP，小汇总直接GitHub可读；原始NPZ/权重不发布。代码、更新计划、注册、决策和来源SHA同分支公开。全树/文件/ZIP回读校验作为发表最终步骤。
'''
    (P/'REPORT.md').write_text(combined)
    absolute_superiority=all(row['median_capture_gain_over_'+name]>0 for row in d['primary_summaries'] for name in ('absolute_full_confusion','absolute_strict_confusion'))
    cross_support=all(row.median_capture_gain_over_uniform>0 for row in controls[controls.control=='cross_paired_cnn_transfer'].itertuples())
    final=dict(status='B1_REVIEW_REQUEST_ONLY_KEEP_CLASS_CONDITIONAL_RISK',d05=d['status'],b0=b['status'],signed_delta_superiority_over_absolute_confusion=absolute_superiority,cross_paired_cnn_positive_damage_transfer_supported=cross_support,replicated_local_damage_supported=b['screen_passed'],new_algorithm_necessary_proven=False,actual_new_fitting_jobs=8,retained_zero_step_startup_failures=8,reserved_gpu_hours=cost['reserved_gpu_hours'],b1_requested_training_jobs=32,b1_requested_formal_models=16,b1_requested_gpu_hour_cap=2.0,b1_started=False,b2_started=False,new_loss_implemented=False,stopped_after_authorized_work=True)
    (P/'decision.json').write_text(json.dumps(final,indent=2)+'\n')
    # Publish full run logs/configs/provenance, never weights or raw predictions.
    evidence=P/'run_evidence';evidence.mkdir(exist_ok=True)
    for phase in ('b0','b0_startup_failure'):
        source=ROOT/'outputs/d05_b0'/phase
        for path in source.rglob('*'):
            if path.is_file() and path.suffix in ('.csv','.json','.log'):
                dest=evidence/phase/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    members=[p for p in P.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.png','.svg','.log') and p.name not in ('evidence_index.json','PUBLIC_ARTIFACT_SHA256.json')]
    with zipfile.ZipFile(P/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(members):
            info=zipfile.ZipInfo(str(path.relative_to(P)),date_time=(2026,10,13,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    index=dict(archive_sha256=hashlib.sha256((P/'evidence.zip').read_bytes()).hexdigest(),members=[dict(path=str(path.relative_to(P)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(members)],raw_npz_or_weights_included=False)
    (P/'evidence_index.json').write_text(json.dumps(index,indent=2)+'\n')
    with zipfile.ZipFile(P/'evidence.zip') as z:
        assert z.testzip() is None
        for m in index['members']:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    print('Reports/figures/archive verified',len(members),'members',(P/'evidence.zip').stat().st_size,'bytes')
if __name__=='__main__':main()
