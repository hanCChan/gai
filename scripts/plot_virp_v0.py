#!/usr/bin/env python3
"""Render standalone figures from completed, unchanged V0 exports."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def values(path, key):
    with Path(path).open() as f:
        return np.array([float(row[key]) for row in csv.DictReader(f)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/virp/v1_v0_20261012')
    args = parser.parse_args()
    out = ROOT / args.output
    assert json.loads((out / 'status.json').read_text())['status'] == 'completed'
    figures = out / 'figures'
    figures.mkdir(exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), layout='constrained')
    colors = ['#2464a0', '#d47829']
    for fold, color in enumerate(colors):
        folder = out / ('fold' + str(fold))
        delta = np.sort(values(folder / 'packets.csv', 'delta'))
        core = np.sort(values(folder / 'packets.csv', 'delta_core'))
        cdf = np.arange(1, len(delta) + 1) / len(delta)
        axes[0, 0].plot(delta, cdf, color=color, label=f'Fold {fold} (n={len(delta):,})')
        axes[0, 1].plot(core, cdf, color=color, label=f'Fold {fold}')
        rx_names = json.loads((folder / 'fit.frozen.json').read_text())['fit_receivers']
        correlations = np.sort(np.concatenate([values(folder / ('trials_' + rx + '.csv'), 'iq_correlation') for rx in rx_names]))
        axes[1, 0].plot(correlations, np.arange(1, len(correlations) + 1) / len(correlations), color=color, label=f'Fold {fold}')
    axes[0, 0].axvline(.05, color='#777777', linestyle='--', label='Coverage threshold 0.05')
    axes[0, 0].axvline(.10, color='#222222', linestyle=':', label='Median threshold 0.10')
    axes[0, 0].axhline(.20, color='#aaaaaa', linewidth=.7)
    axes[0, 0].set(title='Standardized CIS change per packet', xlabel='Median delta across four fixed responses', ylabel='Empirical CDF')
    axes[0, 0].set_xscale('log')
    axes[0, 1].axvline(.05, color='#222222', linestyle=':', label='Core median threshold 0.05')
    axes[0, 1].set(title='CIS change excluding artifact-sensitive cells', xlabel='Core delta per packet', ylabel='Empirical CDF')
    axes[0, 1].set_xscale('log')
    axes[1, 0].axvline(.98, color='#222222', linestyle=':', label='Minimum allowed 0.98')
    axes[1, 0].set(title='IQ correlation (all fixed realizations)', xlabel='Normalized complex correlation', ylabel='Empirical CDF')
    categories = []
    labels = []
    for fold in (0, 1):
        folder = out / ('fold' + str(fold))
        categories.extend([values(folder / 'natural_CIS_centroid_distances.csv', 'delta_centroid'),
                           values(folder / 'synthetic_CIS_centroid_distances.csv', 'delta_centroid')])
        labels.extend([f'F{fold}\nnatural', f'F{fold}\nsynthetic'])
    axes[1, 1].boxplot(categories, tick_labels=labels, showfliers=False)
    axes[1, 1].set(title='Same-TX CIS mean-map displacement', ylabel='Standardized centroid RMS (unpaired data)')
    for ax in axes.flat:
        ax.grid(alpha=.15)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(fontsize=7, loc='best')
    decision = json.loads((out / 'decision.json').read_text())
    fig.suptitle('VIRP v1: preregistered source-only V0 — ' + decision['status'], fontsize=13)
    fig.savefig(figures / 'V0_audit.png', dpi=180)
    fig.savefig(figures / 'V0_audit.pdf')
    fig.savefig(figures / 'V0_audit.svg')
    plt.close(fig)
    print(str(figures / 'V0_audit.png'))


if __name__ == '__main__':
    main()
