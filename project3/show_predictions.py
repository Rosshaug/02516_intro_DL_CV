"""Figure of test-set predictions: one row per test image, columns = image, full-mask model and
one point-supervised model per sampling strategy (same click count and seed).

    python show_predictions.py --clicks 10 --seed 0 --images 4
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from lib.clicks import STRATEGIES
from lib.dataset.PH2Dataset import PH2
from lib.model.UNetModel import UNet

HERE = Path(__file__).resolve().parent
BASE = 'UNet_ph2_256_bce{}_e50_bs6_lr0.001_s{}.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clicks', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--images', type=int, default=4)
    parser.add_argument('--out', default=str(HERE / 'results' / 'predictions.png'))
    args = parser.parse_args()

    columns = {'full masks': BASE.format('', args.seed)}
    for s in STRATEGIES:
        columns[f'{args.clicks}+{args.clicks} {s}'] = BASE.format(f'_click{args.clicks}-{args.clicks}_{s}', args.seed)
    models = {}
    for name, f in columns.items():
        m = UNet()
        m.load_state_dict(torch.load(HERE / 'weights' / f, map_location='cpu'))
        models[name] = m.eval()

    data = PH2('test', size=256)
    fig, axes = plt.subplots(args.images, len(models) + 1, figsize=(2.6 * (len(models) + 1), 2.6 * args.images),
                             squeeze=False)
    for r in range(args.images):
        x, y = data[r]
        axes[r, 0].imshow(x.permute(1, 2, 0))
        axes[r, 0].contour(y[0], levels=[0.5], colors='white', linewidths=0.8)
        for c, (name, m) in enumerate(models.items(), start=1):
            with torch.no_grad():
                pred = m(x[None])[0, 0] > 0
            p, t = pred, y[0] > 0.5
            dice = 2 * (p & t).sum() / (p.sum() + t.sum())
            axes[r, c].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[r, c].contour(y[0], levels=[0.5], colors='#eb6834', linewidths=0.8)
            axes[r, c].set_xlabel(f'Dice {dice:.3f}', fontsize=9)
            if r == 0:
                axes[r, c].set_title(name, fontsize=10)
        if r == 0:
            axes[r, 0].set_title('test image', fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f'Predictions (white) vs. true boundary (orange), seed {args.seed}', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
