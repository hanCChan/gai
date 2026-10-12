#!/usr/bin/env python3
"""Frozen A5 feature contribution audit; development-only, after formal timing."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,csv,json,time,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from shen_p2a.runner import dataset,prepare
from shen_p2a.models import make_model
from shen_p2a.cache import write_json
from shen_evidence.runner import seed_all,loader_for
from report_shen_p2a import read_csv,write_csv

@torch.no_grad()
def inspect_run(run):
    cfg=json.loads((run/'config.json').read_text());seed_all(cfg['seed'])
    assert cfg['family']=='fusion' and cfg['fusion'] in ('concat','scalar_gate')
    device=torch.device('cuda:0');model=make_model(cfg).to(device)
    checkpoint=torch.load(run/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model']);model.eval()
    source=dataset(ROOT,ROOT/cfg['protocol'],cfg,'validation')
    loader=loader_for(source,cfg,False);all_records=[]
    for x,y,rows in loader:
        x=prepare(x,cfg,device);a=model.main(x['main']);b=model.aux(x['aux'])
        if cfg['fusion']=='concat':
            gate=None;w=model.classifier.weight
            logits_main=F.linear(a,w[:,:128],model.classifier.bias)
            logits_aux=F.linear(b,w[:,128:],model.classifier.bias)
            full=model.classifier(torch.cat((a,b),1))
            # The full logits have one bias; each branch-only expression uses it once.
            main_contrib=F.linear(a,w[:,:128]);aux_contrib=F.linear(b,w[:,128:])
        else:
            gate=torch.sigmoid(model.gate(torch.cat((a,b),1)))
            w=model.classifier.weight
            main_contrib=F.linear((1-gate)*a,w);aux_contrib=F.linear(gate*b,w)
            logits_main=main_contrib+model.classifier.bias;logits_aux=aux_contrib+model.classifier.bias
            full=model.classifier((1-gate)*a+gate*b)
        label=y.numpy();full_pred=full.argmax(1).cpu().numpy();main_pred=logits_main.argmax(1).cpu().numpy();aux_pred=logits_aux.argmax(1).cpu().numpy()
        gn=gate[:,0].cpu().numpy() if gate is not None else np.full(len(label),np.nan)
        main_norm=main_contrib.norm(dim=1).cpu().numpy();aux_norm=aux_contrib.norm(dim=1).cpu().numpy()
        for i,row in enumerate(rows.tolist()):all_records.append({'sample_id':f'Test/{cfg["receiver"]}_day2_wireless_test.h5:{row}','row':row,'label':int(label[i]),'prediction':int(full_pred[i]),'main_only_logit_prediction':int(main_pred[i]),'aux_only_logit_prediction':int(aux_pred[i]),'gate':float(gn[i]),'main_logit_norm':float(main_norm[i]),'aux_logit_norm':float(aux_norm[i])})
    saved=read_csv(run/'predictions.csv');saved={r['sample_id']:int(r['prediction']) for r in saved}
    assert len(all_records)==len(saved)==2000
    assert all(r['prediction']==saved[r['sample_id']] for r in all_records)
    results=[]
    for c in range(10):
        selected=[r for r in all_records if r['label']==c]
        results.append({'run':str(run),'receiver':cfg['receiver'],'method':cfg['method'],'seed':cfg['seed'],'raw_label':c+31,'packets':len(selected),'full_correct':sum(r['prediction']==c for r in selected),'main_only_logit_correct':sum(r['main_only_logit_prediction']==c for r in selected),'aux_only_logit_correct':sum(r['aux_only_logit_prediction']==c for r in selected),'main_logit_norm_mean':float(np.mean([r['main_logit_norm'] for r in selected])),'aux_logit_norm_mean':float(np.mean([r['aux_logit_norm'] for r in selected])),'gate_mean':float(np.mean([r['gate'] for r in selected])) if cfg['fusion']=='scalar_gate' else ''})
    totals=dict(results[0],raw_label='all',packets=len(all_records),full_correct=sum(r['prediction']==r['label'] for r in all_records),main_only_logit_correct=sum(r['main_only_logit_prediction']==r['label'] for r in all_records),aux_only_logit_correct=sum(r['aux_only_logit_prediction']==r['label'] for r in all_records),main_logit_norm_mean=float(np.mean([r['main_logit_norm'] for r in all_records])),aux_logit_norm_mean=float(np.mean([r['aux_logit_norm'] for r in all_records])),gate_mean=float(np.mean([r['gate'] for r in all_records])) if cfg['fusion']=='scalar_gate' else '')
    return results+[totals]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--primary',type=Path,required=True);p.add_argument('--followup',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'status.json',{'status':'waiting_for_formal_runs'})
    while True:
        main=json.loads((a.primary/'pipeline_status.json').read_text())['status'];follow=json.loads((a.followup/'pipeline_status.json').read_text())['status']
        if main=='failed' or follow=='failed':raise RuntimeError('Formal runs failed; branch audit waits for resolution')
        if main==follow=='completed':break
        time.sleep(20)
    torch.set_num_threads(2);write_json(a.output/'status.json',{'status':'running'})
    try:
        choices=json.loads((a.primary/'selection.frozen.json').read_text())['choices'];records=[]
        for c in choices:
            if c['comparison_group']!='A5_information_complementarity' or c['method']=='single_ib':continue
            runs=[Path(c['screening_run'])]+[r for r in (a.primary/'confirmation').iterdir() if r.is_dir() and json.loads((r/'config.json').read_text())['receiver']==c['receiver'] and json.loads((r/'config.json').read_text())['method']==c['method']]
            assert len(runs)==3
            for run in runs:
                records.extend(inspect_run(run));print(run.name,flush=True)
        assert len(records)==8*3*11
        write_csv(a.output/'per_device_branch_contributions.csv',records)
        oob_gates=[r for r in records if r['raw_label']=='all' and r['method']=='ib_oob_gate']
        assert len(oob_gates)==6
        gate_range=(min(r['gate_mean'] for r in oob_gates),max(r['gate_mean'] for r in oob_gates))
        gate_aux_exact=all(r['full_correct']==r['aux_only_logit_correct'] for r in oob_gates)
        notes=['# 冻结A5模型的分支贡献诊断','', '24个已完成A5双分支模型，仅重算获准Day2验证包；预测与保存文件逐包完全一致。将实际样本的两支嵌入按已训练线性分类器精确拆成主、辅logit贡献，再分别加同一偏置计算单支logit预测。Gate值仍由两支共同计算。该代数分解使用同一真实输入，避免给模型喂零或反常嵌入；它不是单支重新训练基线，也不证明物理OOB因果。', '', '详见per_device_branch_contributions.csv；`gate_mean`接近1表示主嵌入在该模型输出中被压低。与IB+IB容量对照及逐包纠错/新增错误一起看，不能仅因整体准确率上升就声称融合互补。']
        notes.extend(['', f'六个IB+OOB Gate模型的平均门值范围为{gate_range[0]:.6f}–{gate_range[1]:.6f}；六个模型的辅支单独logit准确数均与完整模型一致：{gate_aux_exact}。这是该模型在Day2开发包上几乎只使用OOB辅支的数值证据，不能作为真正双支互补的证明。'])
        (a.output/'REPORT.md').write_text('\n'.join(notes)+'\n')
        write_json(a.output/'status.json',{'status':'completed','runs':24,'packets_per_run':2000,'saved_predictions_exact':True,'new_training_runs':0,'final_signals_opened':0})
    except BaseException as exc:write_json(a.output/'status.json',{'status':'failed','error':repr(exc)});raise
