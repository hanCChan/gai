#!/usr/bin/env python3
"""Standalone research figure; dependent RX pairs get no artificial error bars."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 r=ROOT/'reports/wisig_w01/2026-10-13';rows=list(csv.DictReader((r/'w0_receiver_date_metrics.csv').open()));effects=list(csv.DictReader((r/'factorial_components.csv').open()));d=json.loads((r/'decision.json').read_text())
 fig,axes=plt.subplots(1,3,figsize=(14,4.5))
 scenarios=[('source',[0,1]),('source',[2]),('internal_rx',[0,1]),('internal_rx',[2])]
 for i,seed in enumerate([20261013,20261014]):
  values=[]
  for role,days in scenarios:
   a=[x for x in rows if int(x['seed'])==seed and x['role']==role and int(x['date_index']) in days]
   values.append(sum(float(x['accuracy'])*int(x['packets']) for x in a)/sum(int(x['packets']) for x in a)*100)
  axes[0].bar(np.arange(4)+i*.35,values,width=.35,label=str(seed))
 axes[0].set_xticks(np.arange(4)+.175);axes[0].set_xticklabels(['Seen RX\nSeen dates','Seen RX\nHeld date','Internal RX\nSeen dates','Internal RX\nHeld date'],fontsize=8);axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)');axes[0].legend(fontsize=8);axes[0].set_title('W0: eight known identities')
 names=['TX','RX','date','TX_RX','TX_date','RX_date','TX_RX_date']
 a=np.array([[float(next(x for x in effects if int(x['seed'])==s and x['effect']==n)['observed_energy_fraction'])*100 for n in names] for s in [20261013,20261014]])
 im=axes[1].imshow(a,aspect='auto',cmap='Blues',vmin=0,vmax=80)
 for i in range(2):
  for j in range(7):axes[1].text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if a[i,j]>45 else 'black')
 axes[1].set_xticks(range(7));axes[1].set_xticklabels(names,rotation=60,ha='right',fontsize=8);axes[1].set_yticks([0,1]);axes[1].set_yticklabels(['Seed13','Seed14']);axes[1].set_title('Cell-mean feature energy (%)')
 x=np.arange(4);c=d['prediction_criteria']
 axes[2].bar(x-.15,[v['mean_squared_distance_gain']*100 for v in c],width=.3,label='Mean distance');axes[2].bar(x+.15,[v['gaussian_w2_squared_gain']*100 for v in c],width=.3,label='Gaussian W2')
 axes[2].axhline(5,color='black',linestyle='--',linewidth=1);axes[2].set_xticks(x);axes[2].set_xticklabels(['13/TX8','13/TX9','14/TX8','14/TX9']);axes[2].set_ylabel('Gain over best simple control (%)');axes[2].set_title('Strict TX + date holdout');axes[2].legend(fontsize=8)
 fig.suptitle('WiSig W0/W1: reliable source learning, conditional transfer gate failed');fig.tight_layout()
 fig.savefig(r/'w0_w1_results.png',dpi=180);fig.savefig(r/'w0_w1_results.pdf');plt.close(fig)
if __name__=='__main__':main()
