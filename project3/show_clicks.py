"""Figure of the simulated clicks: one row per training image, one column per sampling strategy.

    python show_clicks.py --clicks 10 --images 3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from lib.clicks import STRATEGIES, sample_clicks
from lib.dataset.PH2Dataset import PH2

POS, NEG = '#1baf7a', '#e34948'  # green = positive click, red = negative click


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clicks', type=int, default=10)
    parser.add_argument('--images', type=int, default=3)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'clicks.png'))
    args = parser.parse_args()

    data = PH2('train', size=args.size)
    fig, axes = plt.subplots(args.images, len(STRATEGIES), figsize=(3 * len(STRATEGIES), 3 * args.images),
                             squeeze=False)
    for r in range(args.images):
        x, y = data[r]
        mask = y[0] > 0.5
        for c, strategy in enumerate(STRATEGIES):
            pos, neg = sample_clicks(mask, args.clicks, args.clicks, strategy, torch.Generator().manual_seed(r))
            ax = axes[r, c]
            ax.imshow(x.permute(1, 2, 0))
            ax.contour(mask, levels=[0.5], colors='white', linewidths=0.8)
            for idx, color in [(pos, POS), (neg, NEG)]:
                ax.scatter(idx % args.size, idx // args.size, s=28, c=color, edgecolors='white', linewidths=1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(strategy)
    fig.suptitle(f'{args.clicks} positive (green) + {args.clicks} negative (red) clicks; white = true boundary')
    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
