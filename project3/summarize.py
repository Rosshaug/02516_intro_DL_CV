"""Aggregates results/*.json over seeds: prints mean ± std per configuration and plots
test Dice against the number of clicks per sampling strategy (full supervision as reference).

    python summarize.py                 # PH2 UNet runs
    python summarize.py --split val
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib.clicks import STRATEGIES
from measure import METRICS

HERE = Path(__file__).resolve().parent
COLORS = dict(zip(STRATEGIES, ['#2a78d6', '#eb6834', '#1baf7a']))  # categorical slots 1-3
FULL_COLOR = '#52514e'


def key(r):
    a = r['args']
    sup = (a['clicks'], a['neg_clicks'], a['sampling']) if r['weak'] else ('full',)
    return (a['dataset'], r['model'], a['size'], a['loss'], a['augment'], a['epochs']) + sup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--dataset', default='ph2')
    parser.add_argument('--native', action='store_true', help='native-resolution metrics (inside FOV for DRIVE)')
    args = parser.parse_args()

    groups = defaultdict(list)
    for f in sorted((HERE / 'results').glob('*.json')):
        r = json.loads(f.read_text())
        if r['args']['dataset'] == args.dataset and (not args.native or r.get('metrics_native')):
            groups[key(r)].append(r)
    if not groups:
        print('No results found')
        return

    src = 'metrics_native' if args.native else 'metrics'
    print(f'{args.split} metrics ({"native resolution" if args.native else "model resolution"}), mean ± std over seeds')
    print(f'{"model / supervision":36s} {"n":>2s} ' + ' '.join(f'{m:>13s}' for m in METRICS))
    stats = {}
    order = lambda k: (k[1], k[2], k[3], k[4]) + ((0, '', 0, 0) if k[6] == 'full' else (1, k[8], k[6], k[7]))  # full first, then strategy, clicks
    for k in sorted(groups, key=order):
        runs = groups[k]
        vals = {m: np.array([r[src][args.split][m] for r in runs]) for m in METRICS}
        stats[k] = vals
        sup = 'full masks' if k[6] == 'full' else f'{k[6]}+{k[7]} clicks, {k[8]}'
        name = f'{k[1]} {k[2]}' + (f' {k[3]}' if k[3] != 'bce' else '') + (' aug' if k[4] else '') + f', {sup}'
        print(f'{name:36s} {len(runs):2d} ' + ' '.join(f'{v.mean():6.3f}±{v.std():.3f}' for v in vals.values()))

    # Plot: Dice vs clicks per strategy (PH2 UNet runs only)
    stats = {k: v for k, v in stats.items() if k[1] == 'UNet' and k[0] == 'ph2'}
    if not any(k[6] != 'full' for k in stats):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in stats.items():
        if k[6] == 'full':
            mu, sd = v['dice'].mean(), v['dice'].std()
            ax.axhline(mu, color=FULL_COLOR, ls='--', lw=1.5, label=f'full masks ({mu:.3f})')
            ax.axhspan(mu - sd, mu + sd, color=FULL_COLOR, alpha=0.12, lw=0)
    for s in STRATEGIES:
        pts = sorted((k[6], v['dice'].mean(), v['dice'].std()) for k, v in stats.items()
                     if k[6] != 'full' and k[8] == s and k[6] == k[7])
        if pts:
            n, mu, sd = map(np.array, zip(*pts))
            ax.errorbar(n, mu, yerr=sd, color=COLORS[s], lw=2, marker='o', ms=6, capsize=3, label=s)
    ax.set_xscale('log')
    xs = sorted({k[6] for k in stats if k[6] != 'full'})
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel('clicks per class per image (positive = negative)')
    ax.set_ylabel(f'{args.split} Dice')
    ax.grid(axis='y', color='#e5e4e0', lw=0.8)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title(f'PH2, point supervision vs. full masks (mean ± std over seeds)', fontsize=10)
    fig.tight_layout()
    out = HERE / 'results' / f'clicks_ablation_{args.split}.png'
    fig.savefig(out, dpi=150)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
