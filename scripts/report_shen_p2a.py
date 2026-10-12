#!/usr/bin/env python3
"""Report every candidate, selected seeds, independently checked metrics and paired errors."""
import argparse,csv,json,sys,itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import write_json

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def prediction(path):
    rows=read_csv(path);ids=[r['sample_id'] for r in rows];assert len(ids)==2000 and len(set(ids))==2000
    rows=sorted(rows,key=lambda r:r['sample_id']);return rows,np.array([int(r['label']) for r in rows]),np.array([int(r['prediction']) for r in rows])
def check_run(path):
    cfg=json.loads((path/'config.json').read_text());m=json.loads((path/'metrics.json').read_text());r,y,p=prediction(path/'predictions.csv')
    a=accuracy_score(y,p);f=f1_score(y,p,labels=list(range(10)),average='macro',zero_division=0)
    assert abs(a-m['packet_accuracy'])<1e-6 and abs(f-m['macro_f1'])<1e-6
    assert set(y)==set(range(10));np.testing.assert_array_equal(confusion_matrix(y,p,labels=list(range(10))),np.loadtxt(path/'confusion_matrix.csv',skiprows=1,delimiter=',',dtype=int))
    return cfg,dict(m,packet_accuracy_independently_checked=a,macro_f1_independently_checked=f),r,y,p

def pair(a,b):
    # a baseline, b comparison; use full IDs and true labels rather than row position alone.
    ra,ya,pa=a[2:];rb,yb,pb=b[2:];assert [r['sample_id'] for r in ra]==[r['sample_id'] for r in rb];np.testing.assert_array_equal(ya,yb)
    ca,cb=pa==ya,pb==ya;corrected=~ca&cb;new=ca&~cb
    totals={'packets':len(ya),'both_correct':int((ca&cb).sum()),'corrected':int(corrected.sum()),'new_errors':int(new.sum()),'both_wrong':int((~ca&~cb).sum()),'net_correct':int(corrected.sum()-new.sum()),'accuracy_delta_pp':float((cb.mean()-ca.mean())*100)}
    devices=[{'raw_label':c+31,'packets':int((ya==c).sum()),'corrected':int((corrected&(ya==c)).sum()),'new_errors':int((new&(ya==c)).sum()),'net_correct':int(((corrected.astype(int)-new.astype(int))*(ya==c)).sum())} for c in range(10)]
    return totals,devices

def plot(out,summary,diagnostics):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for family in ('A1_matched_representation','A4_frequency_rings','A5_information_complementarity','A6_stability'):
        selected=[s for s in summary if s['comparison_group']==family]
        if not selected:continue
        fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True)
        for ax,rx in zip(axes,('n210_1','rtl_6')):
            values=[s for s in selected if s['receiver']==rx];x=np.arange(len(values))
            ax.bar(x,[s['packet_accuracy_mean']*100 for s in values],yerr=[s['packet_accuracy_sample_std']*100 for s in values],capsize=3)
            ax.set_xticks(x,[s['method'] for s in values],rotation=30,ha='right',fontsize=8);ax.set_title(rx);ax.set_ylim(0,105);ax.set_ylabel('Day2 development accuracy (%)')
        fig.suptitle(family+'; three seeds, sample SD');fig.tight_layout();fig.savefig(out/f'{family}.png',dpi=180);plt.close(fig)
    if diagnostics.exists():
        fig,axes=plt.subplots(2,3,figsize=(12,6))
        for i,repr in enumerate(('spec','cis')):
            for j,(role,c) in enumerate((('train',34),('validation',34),('train',31))):
                path=diagnostics/f'rtl_6_{role}_device{c}_{repr}_mean.npy'
                if path.exists():axes[i,j].imshow(np.load(path),aspect='auto',origin='lower',vmin=-7 if repr=='spec' else -1,vmax=-1 if repr=='spec' else 1);axes[i,j].set_title(f'{repr}, {role}, TX{c}')
        fig.tight_layout();fig.savefig(out/'RTL34_mean_feature_maps.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pipeline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    try:
        runs={};all_metrics=[];statuses=[]
        for root in (a.pipeline/'screening',a.pipeline/'confirmation'):
            for folder in sorted(root.iterdir()):
                if not folder.is_dir():continue
                status=json.loads((folder/'status.json').read_text());statuses.append({'run':str(folder),'status':status['status'],'error':status.get('error','')})
                assert status['status']=='completed',f'Incomplete run {folder}'
                check=check_run(folder);runs[str(folder.resolve())]=check;all_metrics.append(dict(check[1],run=str(folder),phase=root.name))
        assert len(all_metrics)==108
        selected=json.loads((a.pipeline/'selection.frozen.json').read_text());groups={};keyruns={};seedmetrics=[];devices=[]
        for choice in selected['choices']:
            key=(choice['receiver'],choice['comparison_group'],choice['method']);group=[runs[str(Path(choice['screening_run']).resolve())]]
            group.extend(v for path,v in runs.items() if Path(path).parent.name=='confirmation' and (v[0]['receiver'],v[0]['comparison_group'],v[0]['method'])==key)
            assert len(group)==3 and {v[0]['seed'] for v in group}=={20261012,20261013,20261014};groups[key]=group
            for v in group:
                cfg,m,r,y,pred=v;keyruns[(cfg['receiver'],cfg['comparison_group'],cfg['method'],cfg['seed'])]=v;seedmetrics.append(m)
                cm=confusion_matrix(y,pred,labels=list(range(10)))
                for c in range(10):devices.append({'receiver':cfg['receiver'],'comparison_group':cfg['comparison_group'],'method':cfg['method'],'seed':cfg['seed'],'raw_label':c+31,'correct':int(cm[c,c]),'packets':int(cm[c].sum()),'recall':float(cm[c,c]/cm[c].sum())})
        summary=[]
        for (rx,g,method),values in sorted(groups.items()):
            row={'receiver':rx,'comparison_group':g,'method':method,'seeds':3,'parameters':values[0][1]['parameters'],'lr':values[0][0]['lr'],'weight_decay':values[0][0]['weight_decay']}
            for metric in ('packet_accuracy','macro_f1','device34_recall','training_seconds','model_packets_per_second','pipeline_packets_per_second','gpu_peak_allocated_mib'):
                v=np.array([x[1][metric] for x in values]);row.update({metric+'_mean':float(v.mean()),metric+'_sample_std':float(v.std(ddof=1)),metric+'_min':float(v.min()),metric+'_max':float(v.max())})
            summary.append(row)
        pairs=[];pair_devices=[]
        specs=[('A1_matched_representation','spec_light_train_scalar','cis_light_train_scalar'),('A5_information_complementarity','single_ib','capacity_ib_ib_concat'),('A5_information_complementarity','single_ib','capacity_ib_ib_gate'),('A5_information_complementarity','single_ib','ib_oob_concat'),('A5_information_complementarity','single_ib','ib_oob_gate'),('A5_information_complementarity','capacity_ib_ib_concat','ib_oob_concat'),('A5_information_complementarity','capacity_ib_ib_gate','ib_oob_gate'),('A6_stability','ib_stability','full_stability')]
        for rx in ('n210_1','rtl_6'):
            for g,base,comparison in specs:
                for seed in (20261012,20261013,20261014):
                    keys=((rx,g,base,seed),(rx,g,comparison,seed))
                    if any(k not in keyruns for k in keys):continue
                    baseline,method=[keyruns[k] for k in keys];total,dev=pair(baseline,method);meta={'receiver':rx,'seed':seed,'comparison_group':g,'baseline':base,'comparison':comparison,'macro_f1_delta':method[1]['macro_f1']-baseline[1]['macro_f1']}
                    pairs.append(dict(meta,**total));pair_devices.extend(dict(meta,**d) for d in dev)
        capacity=[];p1info=json.loads((ROOT/'configs/shen_p2a/p1_artifacts.json').read_text())
        for choice in p1info['selection']['choices']:
            if choice['kind']!='cis':continue
            rx=choice['receiver'];paths=[Path(choice['screening_run'])]+[Path(p1info['confirmation_root'])/f'E04_{rx}_seed{s}' for s in (20261013,20261014)]
            for path in paths:
                cfg=json.loads((path/'config.json').read_text());m=json.loads((path/'metrics.json').read_text());ra,y,pr=prediction(path/'predictions.csv');big=(cfg,m,ra,y,pr)
                light=keyruns[(rx,'A2_capacity_raw_cis','cis_light_raw',cfg['seed'])];total,dev=pair(big,light)
                capacity.append(dict(receiver=rx,seed=cfg['seed'],baseline='P1_author_raw_cis',comparison='P2_light_raw_cis',large_parameters=m['parameters'],light_parameters=light[1]['parameters'],macro_f1_delta=light[1]['macro_f1']-m['macro_f1'],**total))
        evidence=[]
        for rx in ('n210_1','rtl_6'):
            for method,control in [('ib_oob_concat','capacity_ib_ib_concat'),('ib_oob_gate','capacity_ib_ib_gate')]:
                relevant=[r for r in pairs if r['receiver']==rx and r['comparison']==method and r['baseline'] in ('single_ib',control)]
                evidence.append({'receiver':rx,'method':method,'pairs':len(relevant),'all_three_seeds_accuracy_and_f1_positive_vs_single_and_capacity':len(relevant)==6 and all(r['accuracy_delta_pp']>0 and r['macro_f1_delta']>0 for r in relevant)})
        gate=any(r['all_three_seeds_accuracy_and_f1_positive_vs_single_and_capacity'] for r in evidence)
        decision={'P2_A_completed_runs':108,'P2_B_positive_development_gate':gate,'P2_B_or_P3_automatic_start':False,'fusion_evidence':evidence,'causal_hardware_or_label_error_established':False,'independent_research_unseen_final_receivers_available':False,'frequency_metadata_verified':False}
        write_csv(out/'all_candidate_metrics.csv',all_metrics);write_csv(out/'run_status.csv',statuses);write_csv(out/'selected_seed_metrics.csv',seedmetrics);write_csv(out/'three_seed_summary.csv',summary);write_csv(out/'per_device_metrics.csv',devices);write_csv(out/'paired_differences.csv',pairs);write_csv(out/'paired_per_device.csv',pair_devices);write_csv(out/'raw_cis_capacity_pairs.csv',capacity);write_json(out/'decision.json',decision)
        diagnostics=ROOT/'outputs/shen_p2a/diagnostics_v1';plot(out,summary,diagnostics)
        table=['|Receiver|Group|Method|Acc mean ± sample SD|Macro-F1 mean|Params|','|---|---|---|---:|---:|---:|']+[f'|{s["receiver"]}|{s["comparison_group"]}|{s["method"]}|{100*s["packet_accuracy_mean"]:.2f} ± {100*s["packet_accuracy_sample_std"]:.2f}%|{s["macro_f1_mean"]:.6f}|{s["parameters"]}|' for s in summary]
        notes=['# P2-A 实验结果与决策','', '共108次正式训练：52候选+冻结28组的56个额外seed。Day2为开发集；所有候选、三seed汇总、真正十类Macro-F1、逐包配对、每设备、梯度和成本均保存。', '',*table,'','## 七个研究问题','',
          '1. RTL设备34：P1三个seed的0/200失败已核对；重新读取缓存进行作者模型预测，检查与历史预测逐元素一致。分布、低能分母、原始CFO/SNR、训练拟合聚类和特征图见独立diagnostics_v1。可观测相关性不能证明硬件机制或标签错误。',
          '2. Spec/CIS：A1控制几何、轻量卷积前端和训练预算；A2原始CIS控制容量。查看配对三seed和设备34召回；train_scalar结果不能与原始作者CIS混称纯容量实验。',
          '3. OOB环带：五个频域RI频带均做三seed；保护带/相对能量/窗为首轮探索，全部候选保留。原包Hann前窗只能检验有限记录泄漏敏感性，不建立真实硬件归因。',
          '4. Concat/Gate：纠错与新错误同时报告，按三seed相对SingleIB及同结构IB+IB控制判断。'+json.dumps(evidence,ensure_ascii=False),
          '5. Full/IB：A6同等2LR×2WD预算，30epoch和反事实patience7同时保存。早停后的救援收益属于训练政策改变；不同表示结果仍不足分离所有优化和信息影响。',
          '6. HSTU/CrossAttn：预登记工程门槛'+('满足，允许建议后续比较，但不能声称接收机泛化已证明。' if gate else '未满足，暂不支持开发复杂融合。')+' P2-B/P3均未启动。',
          '7. P3冻结：旧论文全部20台已研究使用，其中6台原盲测已公布结果，不能称研究从未使用。10/4/6仅区分本轮训练和选择角色；N210开发属于训练硬件类型外推，其余开发同类型新实例。协议配置仍是候选，需要单独确认执行。',
          '', '## 限制','', 'Fs/BW/SF与CFO/SNR单位未核实；P1未记录梯度；rank16容量控制改变线性头的秩约束；本轮未读Day3/4或六锁定接收机，未下载OSU。3seed不是接收机总体独立重复，也不是最终测试。P3 GRL/Ma代表性基线和P2-B代码只准备，未真实数据训练。']
        (out/'REPORT.md').write_text('\n\n'.join(notes)+'\n');write_json(out/'status.json',{'status':'completed','runs':108,'metrics_independently_checked':True})
        print(json.dumps(decision,ensure_ascii=False,indent=2))
    except BaseException as exc:write_json(out/'status.json',{'status':'failed','error':repr(exc)});raise
