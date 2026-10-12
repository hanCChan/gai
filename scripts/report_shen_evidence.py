#!/usr/bin/env python3
"""Aggregate complete/failed runs; verify paired IDs and independently recompute F1."""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
ROOT=Path(__file__).resolve().parents[1]


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runs',type=Path,nargs='+',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',type=Path,help='Frozen per-input/receiver learning-rate selection')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    chosen={} if a.selection is None else {(r['receiver'],r['kind']):r['lr'] for r in json.loads(a.selection.read_text())['choices']}
    metrics=[]; statuses=[]; groups={}; per_class=[]; calibration=[]; cases=[]
    for run in sorted(run for base in a.runs for run in base.iterdir()):
        if not run.is_dir() or not (run/'status.json').exists(): continue
        status=json.loads((run/'status.json').read_text()); statuses.append({'run':run.name,**status})
        if status['status']!='completed': continue
        m=json.loads((run/'metrics.json').read_text()); cfg=json.loads((run/'config.json').read_text())
        rows=list(csv.DictReader((run/'predictions.csv').open())); rows.sort(key=lambda r:int(r['row']))
        ids=[r['sample_id'] for r in rows]; labels=np.array([int(r['label']) for r in rows]); pred=np.array([int(r['prediction']) for r in rows])
        expected_ids=[f'Test/{m["receiver"]}_day2_wireless_test.h5:{i}' for i in range(2000)]
        if ids!=expected_ids: raise AssertionError('Missing, duplicate or changed Day2 prediction IDs')
        if not np.isclose(m['macro_f1'],f1_score(labels,pred,labels=list(range(10)),average='macro',zero_division=0),atol=1e-7):
            raise AssertionError('Full-dataset Macro-F1 disagrees with independent sklearn calculation')
        if not np.isclose(m['packet_accuracy'],accuracy_score(labels,pred),atol=1e-7): raise AssertionError('Accuracy mismatch')
        m.update(run=run.name,protocol_sha256=cfg['protocol_sha256'],source_commit=cfg['source_commit']); metrics.append(m)
        m['selected_recipe']=not chosen or m['lr']==chosen[(m['receiver'],m['kind'])]
        profile='selected' if chosen else str(m['lr'])
        if m['selected_recipe']:
            groups.setdefault((m['receiver'],m['seed'],profile),{})[m['kind']]=(ids,labels,pred)
        probabilities=np.array([[float(r[f'p_{c}']) for c in range(10)] for r in rows])
        if not np.allclose(probabilities.sum(1),1.,atol=1e-6): raise AssertionError('Probabilities must sum to one')
        confidence=probabilities.max(1); correct=pred==labels
        ece=0.
        edges=np.linspace(0,1,16)
        for low,high in zip(edges[:-1],edges[1:]):
            mask=(confidence>=low)&(confidence<high if high<1 else confidence<=high)
            if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
        calibration.append({'run':run.name,'receiver':m['receiver'],'seed':m['seed'],'kind':m['kind'],
                            'ece_15_equal_width_bins':float(ece),
                            'negative_log_likelihood':float(-np.log(np.clip(probabilities[np.arange(len(labels)),labels],1e-12,1)).mean()),
                            'multiclass_brier_sum':float(((probabilities-np.eye(10)[labels])**2).sum(1).mean())})
        cm=confusion_matrix(labels,pred,labels=list(range(10)))
        for c in range(10):
            tp=cm[c,c]; support=cm[c].sum(); precision=tp/max(cm[:,c].sum(),1); recall=tp/max(support,1)
            per_class.append({'run':run.name,'receiver':m['receiver'],'class':c,'raw_label':c+31,'support':int(support),
                              'precision':float(precision),'recall':float(recall),'f1':float(2*precision*recall/max(precision+recall,1e-12))})
    pairs=[]
    for (receiver,seed,profile),models in groups.items():
        for left,right in itertools.combinations(sorted(models),2):
            ids,labels,pred_l=models[left]; ids_r,labels_r,pred_r=models[right]
            if ids!=ids_r or not np.array_equal(labels,labels_r): raise AssertionError('Paired predictions are not aligned')
            cl=pred_l==labels; cr=pred_r==labels
            for outcome,mask in [('left_only_correct',cl&~cr),('right_only_correct',cr&~cl),('both_wrong',~cl&~cr)]:
                for i in np.flatnonzero(mask)[:20]:
                    cases.append({'receiver':receiver,'seed':seed,'recipe_profile':profile,'left':left,'right':right,'outcome':outcome,'sample_id':ids[i],
                                  'label':int(labels[i]),'left_prediction':int(pred_l[i]),'right_prediction':int(pred_r[i])})
            pairs.append({'receiver':receiver,'seed':seed,'recipe_profile':profile,'left':left,'right':right,'packets':len(ids),
                          'left_only_correct':int((cl&~cr).sum()),'right_only_correct':int((cr&~cl).sum()),
                          'both_correct':int((cl&cr).sum()),'both_wrong':int((~cl&~cr).sum()),
                          'right_minus_left_accuracy':float(cr.mean()-cl.mean()),'independent_acquisition_unit':'unknown'})
    write_csv(a.output/'calibration.csv',calibration); write_csv(a.output/'paired_case_examples.csv',cases)
    write_csv(a.output/'metrics.csv',metrics); write_csv(a.output/'per_class_metrics.csv',per_class); write_csv(a.output/'paired_differences.csv',pairs)
    write_csv(a.output/'run_status.csv',[{'run':r['run'],'status':r['status'],'error':r.get('error','')} for r in statuses])
    summary=[]
    for rx,kind,lr in sorted({(m['receiver'],m['kind'],m['lr']) for m in metrics if m['selected_recipe']}):
        selected=[m for m in metrics if m['receiver']==rx and m['kind']==kind and m['lr']==lr]
        if len({m['seed'] for m in selected})!=len(selected): raise AssertionError('A tuning candidate must not count as another seed')
        summary.append({'receiver':rx,'kind':kind,'lr':lr,'seeds':len(selected),'accuracy_mean':float(np.mean([m['packet_accuracy'] for m in selected])),
                        'accuracy_std':float(np.std([m['packet_accuracy'] for m in selected],ddof=1)) if len(selected)>1 else None,
                        'macro_f1_mean':float(np.mean([m['macro_f1'] for m in selected])),
                        'macro_f1_std':float(np.std([m['macro_f1'] for m in selected],ddof=1)) if len(selected)>1 else None})
    write_csv(a.output/'seed_summary.csv',summary)
    lines=['# Shen P1 开发集证据报告','',f'完成 {len(metrics)} 项；失败/未完成 {sum(r["status"]!="completed" for r in statuses)} 项。',
           '','每台接收机独立 Day1 训练、Day2 选模，闭集十类。下列为选模开发集结果，并非最终测试或未见接收机结果。',
           '','| 接收机 | 输入 | lr | Seed | Packet Accuracy | 全集 Macro-F1 | 参数量 | 训练秒数 | 模型包/秒 | 管线包/秒 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        if not m['selected_recipe']: continue
        lines.append(f'| {m["receiver"]} | {m["kind"]} | {m["lr"]} | {m["seed"]} | {m["packet_accuracy"]:.2%} | {m["macro_f1"]:.4f} | {m["parameters"]:,} | {m["training_seconds"]:.1f} | {m["model_packets_per_second"]:.1f} | {m["pipeline_packets_per_second"]:.1f} |')
    lines += ['', '完整候选在metrics.csv；三seed汇总只计算冻结配方，不把同一seed的多个学习率当作独立重复。弱基线可能有优化停滞，机会水平结果不能证明输入缺少设备信息。', '','1. **OOB是否可观测？** 在Fs=1MHz/BW=125kHz假设下，六个开发文件均有非零OOB。默认12.5kHz保护带时，Day2能量占比中位数N210约0.0891%、RTL约0.1143%。改变保护带能量明显下降；可能包含泄漏、接收机滤波、噪声，不等于已证明功放指纹。',
              '','2. **独立且互补的设备信息？** OOB-only明显超过10%机会水平可作为独立于IB输入的信息线索；仍可能有设备与采集过程的混杂。Full IQ和IB的差异不能替代E05/E06互补性实验，本轮没有融合实验，互补性尚未验证。',
              '','3. **强CIS/CNN达到什么水平？** 见实测表；CIS模型参数多、架构不同，该配方未复现作者所有训练目标/增强，不能把全部增益归因于CIS或声称优于原论文。',
              '','4. **是否足以继续OOB可靠性选择？** 本轮最多支持进入已规划的互补性/多接收机证据检查；未测普通融合真实接收机负迁移，因此还不能确认可靠性门控的研究动机。P2/P3未自动启动。',
              '','5. **不足以支持论文贡献的结果？** 单seed提升、带外非零能量、训练batch过拟合、两台各自训练、仅人工扰动收益、注意力图均不足以支持接收机不变性或可靠性学习创新。',
              '','每次预测的sample_id、标签、十类概率和错误案例在对应运行目录；paired_differences.csv已严格核对相同包ID和标签。未以packet独立性计算环境级显著性，独立session单位仍未知。三seed只衡量训练随机性，正式对照还需五seed及冻结后独立评估。',
              '','训练/forward时间单位为秒；模型吞吐不含预处理，管线吞吐含加载/预处理/传输。CIS与IQ预处理开销不同，不能只报GPU forward吞吐。',
              '','原报告、权重保留。修复前snapshot和当前commit分别在审计/config中；新指标版本不覆盖旧数值。Day3–4及最终接收机没有访问入口；采集日期/session/作者原包身份与既有OOB诊断重叠未独立证明。']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':len(metrics),'statuses':len(statuses),'paired_comparisons':len(pairs),'output':str(a.output)},indent=2))

if __name__=='__main__': main()
