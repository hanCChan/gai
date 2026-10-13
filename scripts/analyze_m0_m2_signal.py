#!/usr/bin/env python3
"""Analyze all registered results without fitting, threshold search or new signals."""
import csv, hashlib, json, shutil, sys, zipfile
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/m0_signal/2026-10-13'
PUB=ROOT/'reports/m0_signal/2026-10-13'

def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def dump(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    PUB.mkdir(parents=True,exist_ok=False)
    packets=read('packet_diagnostics.csv');preds=read('frozen_predictions.csv');metrics=read('frozen_RX_control_metrics.csv')
    changes=read('synthetic_awgn_deltas.csv');numeric=read('numerical_controls.csv')
    lookup={(p['receiver'],p['row']):p for p in packets}
    fields=['low_rel_1e4','low_rel_1e2','proxy_low_reliability_fraction','repeat_coherence',
            'repeat_residual_power','mirror_circularity','cis_extreme_10_fraction']
    receivers=sorted({p['receiver'] for p in packets});rx=[];tx=[]
    for receiver in receivers:
        selected=[p for p in packets if p['receiver']==receiver]
        row=dict(receiver=receiver,packets=len(selected))
        for field in fields:
            a=np.array([float(p[field]) for p in selected]);row[field+'_mean']=float(a.mean());row[field+'_median']=float(np.median(a))
        row['mirror_half_rank_correlation']=float(spearmanr([float(p['mirror_half1']) for p in selected],[float(p['mirror_half2']) for p in selected]).statistic)
        rx.append(row)
        for cls in range(31,41):
            subset=[p for p in selected if int(p['tx'])==cls]
            tx.append(dict(receiver=receiver,tx=cls,**{field:float(np.mean([float(p[field]) for p in subset])) for field in fields}))
    table(PUB/'per_RX_signal_summary.csv',rx);table(PUB/'per_RX_TX_signal_summary.csv',tx)
    # Associations are descriptive; within RXxTX centering limits RX/TX confounding.
    associations=[];original=[p for p in preds if p['variant']=='identity']
    for run in sorted({p['run'] for p in original}):
        selected=[p for p in original if p['run']==run]
        for field in fields:
            xv=np.array([float(lookup[p['receiver'],p['row']][field]) for p in selected]);yv=1-np.array([float(p['correct']) for p in selected])
            xx=xv.copy();yy=yv.copy();groups={}
            for i,p in enumerate(selected):groups.setdefault((p['receiver'],p['tx']),[]).append(i)
            for ids in groups.values():xx[ids]-=xv[ids].mean();yy[ids]-=yv[ids].mean()
            sd=float(xx.std()*yy.std());corr=float(np.mean(xx*yy)/sd) if sd else 0.
            associations.append(dict(run=run,fold=selected[0]['fold'],seed=selected[0]['seed'],metric=field,
                                     packets=len(selected),within_RX_TX_error_correlation=corr,
                                     interpretation='descriptive; source pool reused; no RX-causal conclusion'))
    table(PUB/'natural_error_associations.csv',associations)
    variants=sorted({p['variant'] for p in preds});summary=[];pert=[];per_tx=[]
    for variant in variants:
        chosen=[p for p in preds if p['variant']==variant]
        records=[p for p in metrics if p['variant']==variant]
        recalls=[]
        for receiver in receivers:
            for cls in range(31,41):
                group=[p for p in chosen if p['receiver']==receiver and int(p['tx'])==cls]
                value=float(np.mean([int(p['correct']) for p in group]));recalls.append(value)
                per_tx.append(dict(variant=variant,receiver=receiver,tx=cls,recall_two_seed_average=value))
        summary.append(dict(variant=variant,accuracy=float(np.mean([int(p['correct']) for p in chosen])),
            minimum_RX_TX_recall_two_seed_average=min(recalls),
            mean_margin=float(np.mean([float(p['margin']) for p in chosen])),
            mean_feature_separability=float(np.mean([float(p['separability_ratio']) for p in records])),
            correct_to_wrong=sum(int(p['correct_to_wrong']) for p in records),wrong_to_correct=sum(int(p['wrong_to_correct']) for p in records)))
    table(PUB/'control_summary.csv',summary);table(PUB/'per_RX_TX_control_recall.csv',per_tx)
    for db in [40,20]:
        selected=[p for p in changes if int(p['added_snr_db'])==db]
        pert.append(dict(added_snr_db=db,trials=len(selected),median_rms_delta=float(np.median([float(p['rms_delta']) for p in selected])),
                         median_low_den_1pct_change_share=float(np.median([float(p['low_den_1pct_change_share']) for p in selected]))))
    table(PUB/'synthetic_awgn_summary.csv',pert)
    # Explicit alternative TX/RX cascades produce indistinguishable observed IQ.
    rng=np.random.default_rng(20261013);s=rng.normal(size=8192)+1j*rng.normal(size=8192)
    beta=.15j;tx=.04+.02j;det=1-abs(tx)**2
    ar=(1-beta*np.conj(tx))/det;br=(beta-tx)/det
    first=s+beta*s.conj();inter=s+tx*s.conj();second=ar*inter+br*inter.conj()
    equiv=float(abs(first-second).max());assert equiv<1e-12
    proper=np.ones((4,3),complex);mirror=np.exp(1j*np.arange(12).reshape(4,3))
    base=proper[:,1:]/proper[:,:-1];mixed=(proper+.15j*mirror)[:,1:]/(proper+.15j*mirror)[:,:-1]
    break_error=float(abs(mixed-base).max());assert break_error>.01
    supplementary=dict(tx_rx_cascade_equivalence_max_abs=equiv,
        cascade1=dict(tx_beta_imag=.15,rx_beta=0),
        cascade2=dict(tx_beta_real=.04,tx_beta_imag=.02,rx_a_real=ar.real,rx_a_imag=ar.imag,rx_b_real=br.real,rx_b_imag=br.imag),
        mirror_mixing_ratio_failure_max_abs=break_error,
        interpretation='Same observation from distinct TX/RX contributions; no unique causal RX estimate')
    dump(PUB/'supplementary_identifiability_control.json',supplementary)
    execution=json.loads((OUT/'execution.json').read_text())
    decision=dict(status='HOLD_M3_NO_NOVEL_METHOD',training_authorized_conditionally=True,budget_cap=None,
        training_jobs_executed=0,optimizer_steps=0,
        A='Local noise/low-energy mechanism supported by synthetic controls; natural residual is not independently calibrated additive noise; causal natural mechanism remains unresolved',
        B='STOP_RX_CAUSAL_COMPENSATION_WITHOUT_REFERENCE: composite TX/RX mirror effects are not identifiable from these IQ observations',
        M2='Fixed traditional controls evaluated; frozen-network input change does not by itself identify physical identity loss or prove a new estimator is needed',
        M0='Readable primary material and mathematical screening completed; TQCD/WL/GAN full derivation comparison incomplete; no novelty claim',
        WiSig='Official ManyRx download task only; signal remains sealed; no cross-standard confirmation performed',
        next_condition='Obtain direct-competitor derivations and independent natural-noise/reference validity before proposing one concrete correction; do not add arbitrary modules or coefficient search')
    dump(PUB/'decision.json',decision)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar([p['receiver'] for p in rx],[p['proxy_low_reliability_fraction_mean'] for p in rx]);axes[0].tick_params(axis='x',rotation=65)
    axes[0].set_ylabel('Fraction below 10 x repeat-residual proxy');axes[0].set_title('Proxy includes noise + mismatch')
    axes[1].bar([p['variant'].replace('_','\n') for p in summary],[p['accuracy']*100 for p in summary]);axes[1].set_ylabel('Frozen CE accuracy (%)');axes[1].set_ylim(0,100)
    for ext in ['png','pdf']:fig.savefig(PUB/('mechanism_controls.'+ext),dpi=180)
    plt.close(fig)
    for name in ['execution.json','synthetic_controls.json','pre_execution_seal.json','input_hashes.csv','frozen_RX_control_metrics.csv','access_log.json']:
        shutil.copy2(OUT/name,PUB/name)
    download=Path('/home/keyan/data/wisig-manyrx-sealed/download_status.json')
    if download.exists():shutil.copy2(download,PUB/'wisig_download_snapshot.json')
    lines=['# M0–M2：机制筛选与固定传统处理结果','',
        '决策：**HOLD M3；不启动新算法训练。** 用户不设资源上限，但真实机制、身份保留、可辨识性与竞争方法比较仍是前提。B1与所有历史负结果未修改。','',
        f"完成十源RX的20,000包原始存储IQ读取、Strict CIS逐包重建、四个冻结CE模型的三种输入控制，共120,000条模型预测；{execution['added_awgn_trials']}次合成加噪对照。0次训练、0 optimizer step。历时{execution['seconds']:.1f}秒（含I/O/哈希/诊断/冻结GPU推理；不是拟合GPUh）。",'',
        '|固定输入|Accuracy|最低RX×TX Recall（两seed平均）|正确→错误|错误→正确|',
        '|---|---:|---:|---:|---:|']
    for p in summary:lines.append(f"|{p['variant']}|{100*p['accuracy']:.3f}%|{100*p['minimum_RX_TX_recall_two_seed_average']:.2f}%|{p['correct_to_wrong']}|{p['wrong_to_correct']}|")
    lines+=['','## A：局部数学性质与自然机制分开','',
        '高SNR独立噪声正对照支持局部对数谱商方差随低能量增大；共享乘性响应与常数商负对照通过。IQ层AWGN包含重叠窗噪声相关。自然低能量占比见每RX/每TX表，固定相对阈值并非噪声校准。分子低能量也会影响对数谱商。','',
        '|人工新增SNR|中位CIS RMS变化|分母最低1%单元占变化能量（中位）|','|---|---:|---:|']
    for p in pert:lines.append(f"|{p['added_snr_db']} dB|{p['median_rms_delta']:.5f}|{100*p['median_low_den_1pct_change_share']:.2f}%|")
    lines+=['',
        '自然数据的1024延迟残差含重复失配、相位漂移、非平稳性和噪声；没有独立静默段/接收链校准，不能当作测得的热噪声。自然错误关联在RX×TX内部去均值，各折与seed全部报告，不将来源池相关性当自然RX噪声因果证明。','',
        '## B：可辨识性限制','',
        f"合成不同TX/RX广义线性级联产生相同观测，最大绝对误差{equiv:.3g}；镜像混合使纯乘性商不变性失效。自然非圆性有重复统计，但可能同时反映波形、TX与RX。未取得独立参考，停止“只消RX镜像并保TX指纹”的因果补偿候选。",'',
        '## M2：传统控制与身份证据','',
        '重复残差代理正则化与盲圆化都固定规则、无系数搜索、无RX标签选择。四个CE模型仅在其五个未拟合源RX上评估，原scaler不重估，identity回读历史logits一致。表格给出所有收益/损害；margin和固定模型特征类间/类内比是辅助证据。改变前端造成分布偏移，所以控制下降既不能独立证明物理身份删除，也不能自动授权新算法。','',
        '## 文献、数据与执行状态','',
        '[CIS全文](https://arxiv.org/pdf/2107.02867)已指出低SNR问题；[微分谱比值全文](https://arxiv.org/html/2503.22378v1)已有重复平均和跨制式实验。两者的信道假设不同，不能互换。TQCD、广义线性会议与GAN+RLS未取得完整推导；PIR正文核对状态见文献记录。因此不能宣称新颖性空白。','',
        'WiSig仅官方ManyRx压缩下载任务；快照记录Google Drive配额限制/进度，未反序列化、未读取信号或做确认。源数据是已上游处理的8192点存储IQ，不称未经处理ADC或已确认完整包边界。十源池已用于历史研究，7/10是RTL，没有盲测意义。开发/锁定RX、Shen Day3/4、OSU未访问。','',
        '下一步需要竞争方法正文和可验证的自然噪声/镜像参考条件；本轮不因为预算宽松补跑模型。所有数值、代码seal、输入hash、访问记录、完整派生预测和日志在证据索引中。原IQ、权重、私稿与评审信不发布。']
    (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Include every derived numeric row, registration and exact sources; exclude signals/weights.
    members=[]
    with zipfile.ZipFile(PUB/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(OUT.iterdir()):
            assert p.suffix in ['.csv','.json']
            z.write(p,'derived/'+p.name);members.append(dict(member='derived/'+p.name,sha256=sha(p),bytes=p.stat().st_size))
        for rel in ['scripts/run_m0_m2_signal.py','scripts/analyze_m0_m2_signal.py','scripts/download_wisig_manyrx.py',
                    'configs/m0_signal/registration.json','docs/m0_signal/EXECUTION_PLAN.md','docs/m0_signal/LITERATURE_AND_HYPOTHESES.md']:
            p=ROOT/rel;z.write(p,'source/'+rel);members.append(dict(member='source/'+rel,sha256=sha(p),bytes=p.stat().st_size))
        log=Path('/home/keyan/private/m0_signal_execution.log');z.write(log,'logs/execution.log')
        members.append(dict(member='logs/execution.log',sha256=sha(log),bytes=log.stat().st_size))
    dump(PUB/'evidence_members.json',members)
    (PUB/'EVIDENCE_INDEX.md').write_text('# Evidence index\n\n'
        '完整派生数值/预测、合成控制、封存配置/代码、输入hash与执行日志：`evidence.zip`；逐成员SHA：`evidence_members.json`。\n\n'
        '公开摘要包括每RX/每TX诊断、所有自然错误关联、每RX/每TX控制收益损害、数值检查和decision。无IQ/权重/凭据/私稿。\n')
    print(json.dumps(dict(summary=summary,synthetic=pert,decision=decision['status'],zip_bytes=(PUB/'evidence.zip').stat().st_size),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
