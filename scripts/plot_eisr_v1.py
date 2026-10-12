#!/usr/bin/env python3
"""Standalone EISR evidence figures from completed exports, no selection."""
import csv,json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/eisr/v1_20261012');args=parser.parse_args()
    out=ROOT/args.output;assert json.loads((out/'status.json').read_text())['status']=='completed'
    dst=out/'figures';dst.mkdir(exist_ok=False)
    e1=rows(out/'E1/per_RX.csv');labels=[r['receiver'] for r in e1];x=np.arange(len(labels))
    has_e2=(out/'E2/decision.json').exists()
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors=['#7991a7','#195b84','#82b5a5','#bb985b']
    for i,(key,label) in enumerate([('Full_accuracy','Full'),('Strict_accuracy','Strict'),('Hard_oracle_accuracy','Hard oracle'),('Convex_soft_oracle_accuracy','Convex soft oracle')]):
        axes[0,0].bar(x+(i-1.5)*.18,[100*float(r[key]) for r in e1],width=.18,label=label,color=colors[i])
    axes[0,0].set(title='E1: paired expert Accuracy and diagnostic ceilings',ylabel='Accuracy (%)',xticks=x,xticklabels=labels)
    axes[0,0].legend(fontsize=8,ncol=2)
    gain=np.array([100*float(r['Hard_gain_over_RX_best']) for r in e1])
    axes[0,1].bar(x,gain,color='#195b84')
    axes[0,1].axhline(3,color='#b44b34',linestyle='--',label='Equal-RX mean threshold: 3 pp')
    axes[0,1].axhline(gain.mean(),color='#333333',linestyle=':',label=f'Equal-RX mean: {gain.mean():.2f} pp')
    axes[0,1].set(title='E1: hard rescue space above each RX best expert',ylabel='Accuracy gain (percentage points)',xticks=x,xticklabels=labels)
    axes[0,1].legend(fontsize=8)
    if has_e2:
        e2=rows(out/'E2/per_RX.csv');d=np.array([100*float(r['delta_macro_f1']) for r in e2])
        axes[1,0].bar(x,d,color=np.where(d>=0,'#43886c','#ba6246'))
        axes[1,0].axhline(0,color='#444444',linewidth=.7)
        axes[1,0].axhline(1,color='#777777',linestyle='--',label='Equal-RX mean threshold: +1 pp')
        axes[1,0].set(title='E2: fixed hard selector vs source-chosen baseline',ylabel='Macro-F1 change (percentage points)',xticks=x,xticklabels=[r['receiver'] for r in e2])
        axes[1,0].legend(fontsize=8)
        decision=json.loads((out/'E2/decision.json').read_text())
        tx=np.arange(31,41)
        base=100*np.array(decision['baseline_equal_RX_recall_by_TX']);selector=100*np.array(decision['selector_equal_RX_recall_by_TX'])
        axes[1,1].bar(tx-.18,base,width=.36,label='Source-chosen baseline',color='#7991a7')
        axes[1,1].bar(tx+.18,selector,width=.36,label='Fixed hard selector',color='#195b84')
        axes[1,1].set(title='E2: per-TX Recall, each RX weighted equally',ylabel='Recall (%)',xlabel='TX',xticks=tx,ylim=(0,105))
        axes[1,1].legend(fontsize=8)
    else:
        tx=rows(out/'E1/per_RX_TX.csv')
        rescue=[sum(int(r['Full_rescue']) for r in tx if int(r['TX'])==c) for c in range(31,41)]
        damage=[sum(int(r['Full_damage']) for r in tx if int(r['TX'])==c) for c in range(31,41)]
        axes[1,0].bar(np.arange(31,41)-.18,rescue,width=.36,label='Full rescues',color='#43886c')
        axes[1,0].bar(np.arange(31,41)+.18,damage,width=.36,label='Full damages',color='#ba6246');axes[1,0].legend()
        axes[1,0].set(title='E1: source TX complementarity',xlabel='TX',ylabel='Packets')
        axes[1,1].axis('off');axes[1,1].text(.5,.5,'E1 failed: E2 not executed',ha='center')
    for ax in axes.flat:ax.grid(axis='y',alpha=.15)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):plt.setp(ax.get_xticklabels(),rotation=35,ha='right',fontsize=8)
    state=json.loads((out/'decision.json').read_text())['state']
    fig.suptitle('EISR v1: preregistered CPU-only source screening — '+state)
    for ext in ('png','pdf','svg'):fig.savefig(dst/('EISR_v1_screening.'+ext),dpi=180)
    print(str(dst/'EISR_v1_screening.png'))


if __name__=='__main__':main()
