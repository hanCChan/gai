#!/usr/bin/env python3
"""Wait for both datasets, expose replicated/failed trends, keep G2/G3 gated."""
import datetime,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a.cache import write_json
from report_shen_p2a import read_csv,write_csv

if __name__=='__main__':
    out=ROOT/'outputs/shen_p2a8/g1_joint_decision_v1'
    if out.exists():
        previous=json.loads((out/'status.json').read_text())
        if previous['status']!='waiting_for_both_datasets':raise PermissionError('Existing completed or failed joint result is immutable')
    out.mkdir(exist_ok=True);state=lambda status,**kw:write_json(out/'status.json',dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    pipelines={dataset:ROOT/f'outputs/shen_p2a8/g1_{dataset}_v{2 if dataset=="shen" else 1}' for dataset in ('shen','osu')}
    try:
        state('waiting_for_both_datasets',G2_G3_started=False)
        while True:
            states={d:json.loads((p/'pipeline_status.json').read_text())['status'] if (p/'pipeline_status.json').exists() else 'pending' for d,p in pipelines.items()}
            if any(s=='failed' for s in states.values()):raise RuntimeError('A dataset pipeline failed: '+str(states))
            if all(s=='completed' for s in states.values()):break
            osu=ROOT/'outputs/shen_p2a8/g1_osu_orchestration.json'
            if osu.exists() and json.loads(osu.read_text())['status']=='failed':raise RuntimeError('OSU preprocessing or protocol failed; do not skip the independent dataset')
            time.sleep(20)
        reg=json.loads((ROOT/'configs/shen_p2a8/g1_registration.json').read_text());trend=[]
        for dataset,pipeline in pipelines.items():
            pairs=read_csv(pipeline/'report/paired_corrections_and_regressions.csv')
            pairs=[r for r in pairs if r['raw_label']=='all' and (r['domain'] in ['shen_'+rx+'_test' for rx in reg['development_receivers']] if dataset=='shen' else r['domain']!=r['direction'])]
            for direction,domain,base,comparison in sorted({(r['direction'],r['domain'],r['baseline'],r['comparison']) for r in pairs}):
                values=[r for r in pairs if (r['direction'],r['domain'],r['baseline'],r['comparison'])==(direction,domain,base,comparison)];assert len(values)==3
                deltas=np.array([float(r['delta_accuracy_pp']) for r in values]);f1s=np.array([float(r['macro_f1_delta']) for r in values])
                trend.append(dict(dataset=dataset,direction=direction,domain=domain,baseline=base,comparison=comparison,mean_accuracy_delta_pp=float(deltas.mean()),accuracy_delta_seed_sd=float(deltas.std(ddof=1)),mean_macro_f1_delta=float(f1s.mean()),positive_seeds=int((deltas>0).sum()),negative_seeds=int((deltas<0).sum()),corrected_total=sum(int(r['corrected']) for r in values),new_errors_total=sum(int(r['new_errors']) for r in values)))
        write_csv(out/'cross_dataset_trend.csv',trend)
        table=['|Dataset|Source|Target|Strict−Full|Accuracy delta pp|Positive / negative seeds|','|---|---|---|---|---:|---:|']+[f'|{r["dataset"]}|{r["direction"]}|{r["domain"]}|{r["comparison"]}|{r["mean_accuracy_delta_pp"]:+.2f}|{r["positive_seeds"]} / {r["negative_seeds"]}|' for r in trend]
        (out/'REPORT.md').write_text('\n'.join(['# G1 跨数据集结果与阶段边界','','Shen四个开发RX与OSU两个转移方向的三seed全部逐包核验。以下趋势用于机制复核，不代表发现通用最佳频带。','','\n'.join(table),'','只有Strict改善不能充分证明频带辨识度与稳定性冲突；须同时阅读source_only_band_statistics与每RX/每TX变化，区分输入、优化、scaler及连续窗口筛选影响。OSU两个方向使用同一对物理RX，不是两个独立RX总体实验。Shen root日期未知；两数据集都存在历史研究使用，不称研究史全新盲测。','','G2/G3仍保持未启动：现有表格不是独立验证的源域失效预测器，不能据此直接设计新loss或宣布新方法。若冲突不能跨Shen开发RX和OSU复现，按最新用户方案停止围绕频带开发新方法；若有描述性复现，也需先冻结G2评估协议。额外跨日期/信道与WiSig属于后续G4，没有伪造完成。'])+'\n')
        state('G1_results_ready_for_mechanism_review',G1_general_mechanism_proven=False,G2_G3_started=False,old_P2_B_P3_started=False)
    except BaseException as e:state('failed',error=repr(e),G2_G3_started=False);raise
