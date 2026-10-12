#!/usr/bin/env python3
"""Source-only checkpoint analysis: distinguish dying IQ embeddings from frequency evidence."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a8.g1_data import G1Dataset,fit_scaler
from train_shen_p2a8_g1 import make,prepare
from report_shen_p2a import read_csv,write_csv
from shen_p2a.cache import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pipeline',type=Path,required=True);a=p.parse_args();out=a.pipeline/'optimization_diagnosis';out.mkdir(exist_ok=False);rows=[]
    torch.set_num_threads(2)
    for folder in sorted((a.pipeline/'runs').iterdir()):
        cfg=json.loads((folder/'config.json').read_text());status=json.loads((folder/'status.json').read_text());assert status['status']=='completed'
        history=read_csv(folder/'history.csv');last=history[-1]
        record=dict(run=folder.name,method=cfg['method'],seed=cfg['seed'],phase=cfg['phase'],last_train_accuracy=float(last['train_accuracy']),last_train_loss=float(last['train_loss']),chance_train_accuracy=1/cfg['classes'],last_epoch=int(last['epoch']),near_chance_epochs=sum(float(r['train_accuracy'])<1/cfg['classes']+.05 for r in history))
        if cfg['method'] in ('full_iq','strict_iq'):
            cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);model=make(cfg['method'],cfg['classes']).eval();model.load_state_dict(cp['model'])
            logits=[];embeddings=[]
            for key in cfg['train_keys']:
                ds=G1Dataset(ROOT,[key],cfg['method']);indices=[int(np.flatnonzero(ds.labels_all==c)[0]) for c in range(cfg['classes'])]
                x=torch.from_numpy(np.stack([ds[i][0] for i in indices]));x=prepare(x,cfg['method'],'cpu')
                with torch.no_grad():
                    features=model.features(x);embedding=model.head[:-1](features);embeddings.append(embedding.numpy());logits.append(model.head[-1](embedding).numpy())
            z=np.concatenate(embeddings);l=np.concatenate(logits)
            record.update(source_probe_samples=len(z),embedding_zero_fraction=float((z==0).mean()),embedding_between_sample_variance=float(z.var(0).mean()),logits_between_sample_variance=float(l.var(0).mean()),distinct_source_probe_predictions=int(len(np.unique(l.argmax(1)))),checkpoint_collapse_evidence=bool((z==0).mean()>.999 or (l.var(0).mean()<1e-12 and len(np.unique(l.argmax(1)))==1)))
        rows.append(record)
    write_csv(out/'source_checkpoint_optimization.csv',rows)
    write_json(out/'status.json',dict(status='completed',runs=len(rows),target_data_used=False,hyperparameters_changed=False,failed_seeds_retained=True))
    (out/'REPORT.md').write_text('# 源域优化诊断\n\n固定最终checkpoint对每个源RX每类首个窗口做无训练前向检查，并记录最后epoch训练loss/accuracy、随机水平epoch数、IQ head ReLU零值比例、嵌入/输出跨样本方差及预测类别数量。近随机训练且嵌入死亡/常数输出支持数值塌缩解释，不能将此模型性能当成有效输入信息量比较。没用开发数据选择超参，失败seed保留，不因为观察到目标性能改变训练。CIS未重现IQ head结构，不混用其零值比例。\n')
