#!/usr/bin/env python3
"""After original108 finish, execute symmetric exploratory strict-IB STFT follow-up."""
import argparse,json,time,sys,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from report_shen_p2a import check_run,pair,write_csv
from shen_p2a.cache import write_json
import numpy as np

def run(args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--primary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    state=lambda s,**kw:write_json(out/'pipeline_status.json',dict(status=s,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    try:
        state('waiting_for_primary108')
        while True:
            status=json.loads((a.primary/'pipeline_status.json').read_text())['status']
            if status=='completed':break
            if status=='failed':raise RuntimeError('Primary108 failed; follow-up cannot skip it')
            time.sleep(20)
        state('smoke_running');run(['scripts/launch_shen_p2a.py','--stage','smoke','--matrix','configs/shen_p2a_a7/matrix.json','--output',out/'smoke'])
        state('screening_running');run(['scripts/launch_shen_p2a.py','--stage','train','--matrix','configs/shen_p2a_a7/matrix.json','--output',out/'screening','--smoke-dir',out/'smoke'])
        jobs=json.loads((ROOT/'configs/shen_p2a_a7/matrix.json').read_text());choices=[];extra=[];(out/'configs').mkdir()
        for rx in ('n210_1','rtl_6'):
            for repr in ('spec','cis'):
                candidates=[(j,check_run(out/'screening'/j['run_id'])) for j in jobs if j['receiver']==rx and j['method']==f'{repr}_strict_ib']
                job,v=max(candidates,key=lambda x:(x[1][1]['macro_f1'],x[1][0]['lr']==.001));choices.append(dict(receiver=rx,method=job['method'],screening_run=str(out/'screening'/job['run_id']),config=v[0]))
                for seed in (20261013,20261014):
                    name=f'{job["run_id"]}_seed{seed}';cfg=dict(v[0],seed=seed);path=out/'configs'/f'{name}.json';path.write_text(json.dumps(cfg,indent=2)+'\n');extra.append(dict(job,run_id=name,config=str(path)))
        write_json(out/'selection.frozen.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),choices=choices,post_diagnosis_followup=True))
        write_json(out/'confirmation_matrix.json',extra);state('confirmation_running');run(['scripts/launch_shen_p2a.py','--stage','train','--matrix',out/'confirmation_matrix.json','--output',out/'confirmation','--smoke-dir',out/'smoke'])
        report=out/'report';report.mkdir();all_runs=[];summary=[];pairs=[];devices=[]
        full=json.loads((a.primary/'selection.frozen.json').read_text())['choices']
        for choice in choices:
            rx=choice['receiver'];method=choice['method'];repr=method.split('_')[0];values=[check_run(Path(choice['screening_run']))]+[check_run(d) for d in (out/'confirmation').iterdir() if d.is_dir() and json.loads((d/'config.json').read_text())['method']==method and json.loads((d/'config.json').read_text())['receiver']==rx]
            assert len(values)==3;full_choice=next(c for c in full if c['receiver']==rx and c['method']==f'{repr}_light_train_scalar')
            row=dict(receiver=rx,method=method,seeds=3,lr=values[0][0]['lr'],parameters=values[0][1]['parameters'])
            for metric in ('packet_accuracy','macro_f1','device34_recall'):
                x=np.array([v[1][metric] for v in values]);row[metric+'_mean']=float(x.mean());row[metric+'_sample_std']=float(x.std(ddof=1))
            summary.append(row)
            for v in values:
                seed=v[0]['seed'];baseline_path=Path(full_choice['screening_run']) if seed==20261012 else next(d for d in (a.primary/'confirmation').iterdir() if d.is_dir() and json.loads((d/'config.json').read_text())['receiver']==rx and json.loads((d/'config.json').read_text())['method']==full_choice['method'] and json.loads((d/'config.json').read_text())['seed']==seed)
                baseline=check_run(baseline_path);total,dev=pair(baseline,v);meta=dict(receiver=rx,seed=seed,baseline=full_choice['method'],comparison=method,macro_f1_delta=v[1]['macro_f1']-baseline[1]['macro_f1']);pairs.append(dict(meta,**total));devices.extend(dict(meta,**r) for r in dev)
        for phase in ('screening','confirmation'):
            for d in (out/phase).iterdir():
                if d.is_dir():all_runs.append(dict(check_run(d)[1],run=str(d),phase=phase))
        assert len(all_runs)==16;write_csv(report/'all_metrics.csv',all_runs);write_csv(report/'three_seed_summary.csv',summary);write_csv(report/'paired_full_STFT_vs_strict_IB_STFT.csv',pairs);write_csv(report/'paired_per_device.csv',devices)
        notes=['# 严格IB生成Spec/CIS：诊断后的追加对照','', '本轮16次训练（8候选+8新增seed），与原始108分开；追加动机是RTL34在Spec/CIS的带外STFT位置发生较大跨日期变化。投影原始FFT的|f|≤62.5kHz后才重新STFT；STFT窗口仍可生成边缘能量，但信息源只有原始IB。保持52×126几何、rank16容量、2LR预算和train-only标量缩放。所有结果均为Day2开发证据。', '',json.dumps(summary,ensure_ascii=False,indent=2),'', '逐包配对比较分别冻结的同预算Full-STFT表示。改善可支持在该训练配方下去除原始OOB信息的功能性效果；它不唯一定位物理来源，也不证明收集标签或接收机硬件原因。追加实验在观察数据后提出，不能冒称原先预登记的盲测。没有启动P2-B/P3。']
        (report/'REPORT.md').write_text('\n'.join(notes)+'\n');state('completed',formal_runs=16,P2_B_or_P3_started=False)
    except BaseException as exc:state('failed',error=repr(exc));raise
