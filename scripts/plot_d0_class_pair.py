#!/usr/bin/env python3
"""Publication artifacts from already computed D0 tables; no data/model load."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reports/d0_class_pair/2026-10-13_D0'
def save(fig,name):
    fig.savefig(P/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(P/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def main():
    rx=pd.read_csv(P/'per_RX.csv');tx=pd.read_csv(P/'per_RX_TX.csv')
    order=['b200_mini_1','pluto_1','rtl_2','rtl_5','rtl_7','b210_1','rtl_1','rtl_3','rtl_6','rtl_9']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for m in ('full_cis','strict_cis'):
        values=rx[rx.method==m].set_index('receiver').loc[order]
        axes[0].plot(order,values.accuracy*100,'o-',label=m)
        axes[1].plot(order,values.worst_tx_recall*100,'o-',label=m)
    for ax,title in zip(axes,['Accuracy by RX','Worst TX recall by RX']):
        ax.set_title(title);ax.set_ylim(0,100);ax.set_ylabel('%');ax.tick_params(axis='x',rotation=70);ax.grid(alpha=.2);ax.legend()
    save(fig,'accuracy_and_worst_class')
    changes=pd.read_csv(P/'strict_full_pair_changes.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for f,ax in enumerate(axes):
        table=changes[changes.fold==f].groupby(['tx_i','tx_j']).delta_confusion_risk.mean().unstack().reindex(index=range(31,41),columns=range(31,41))*100
        im=ax.imshow(table,cmap='coolwarm',vmin=-50,vmax=50);ax.set_title(f'Fold {f}: Strict - Full confusion (pp)');ax.set_xticks(range(10));ax.set_xticklabels(range(31,41));ax.set_yticks(range(10));ax.set_yticklabels(range(31,41));ax.set_xlabel('Predicted TX');ax.set_ylabel('True TX')
    fig.colorbar(im,ax=axes.ravel().tolist());save(fig,'pair_confusion_changes')
    conf=pd.read_csv(P/'confusion_counts.csv')
    for method in ('full_cis','strict_cis'):
        fig,axes=plt.subplots(2,5,figsize=(14,6))
        for ax,r in zip(axes.flat,order):
            table=conf[(conf.method==method)&(conf.receiver==r)].pivot(index='tx_i',columns='tx_j',values='conditional_probability')
            im=ax.imshow(table,vmin=0,vmax=1,cmap='Blues');ax.set_title(r);ax.set_xticks(range(10));ax.set_xticklabels(range(31,41),rotation=90);ax.set_yticks(range(10));ax.set_yticklabels(range(31,41))
        fig.suptitle(method+' RX confusion: rows true, columns predicted');fig.tight_layout();save(fig,'confusions_'+method)
    evaluation=pd.read_csv(P/'risk_prediction_evaluation.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,control in zip(axes,['same_cnn_loro','different_family_reference']):
        s=evaluation[evaluation.control==control]
        for name in s.predictor.unique():
            vals=s[s.predictor==name].groupby(['method','fold']).top10_error_capture.median()
            ax.plot([str(v) for v in vals.index],vals.values,'o-',label=name)
        ax.set_title(control);ax.set_ylim(0,1);ax.set_ylabel('Median error mass in top 10/90 pairs');ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,loc='lower right');fig.tight_layout();save(fig,'risk_predictability')
    # All pairs and both directions; no selected TX illustration.
    margins=pd.read_csv(P/'worst_RX_pairs.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,method in zip(axes,('full_cis','strict_cis')):
        table=margins[(margins.method==method)&(margins.group=='all')].pivot(index='tx_i',columns='tx_j',values='min_rx_mean_normalized_margin')
        im=ax.imshow(table,cmap='coolwarm',vmin=-3,vmax=3);ax.set_title(method+': worst RX normalized margin');ax.set_xticks(range(10));ax.set_xticklabels(range(31,41));ax.set_yticks(range(10));ax.set_yticklabels(range(31,41));ax.set_xlabel('Competing TX');ax.set_ylabel('True TX')
    fig.colorbar(im,ax=axes.ravel().tolist());save(fig,'worst_pair_margins')
    print('Created six PNG/SVG figure pairs')
if __name__=='__main__':main()
