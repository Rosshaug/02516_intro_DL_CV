import os
import json
import random
from pathlib import Path
from time import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.model.UNetMulti import UNet as UNetMulti
from lib.losses import BCELoss, FocalLoss, WeightedBCELoss, PointBCELoss
from lib.clicks import STRATEGIES
from lib.dataset.PH2Dataset import PH2, PH2Clicks
from lib.dataset.DRIVE_Dataset import DRIVE
from lib.dataset.CMPDataset import Facade
from measure import evaluate, evaluate_native, print_table

DATASETS = {'ph2': PH2, 'drive': DRIVE, 'facade': Facade}
HERE = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default='ph2')
    parser.add_argument('--model', choices=['encdec', 'unet'], default='encdec')
    parser.add_argument('--size', type=int, default=128)  # EncDec needs 128, UNet any multiple of 16
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', choices=['bce', 'focal', 'wbce'], default='bce')
    parser.add_argument('--augment', action='store_true')  # DRIVE only
    parser.add_argument('--seed', type=int, default=0)
    # Weak (point) supervision, PH2 only. --clicks 0 means full masks (part 1).
    parser.add_argument('--clicks', type=int, default=0, help='positive clicks per training image')
    parser.add_argument('--neg_clicks', type=int, default=None, help='negative clicks (default: = --clicks)')
    parser.add_argument('--sampling', choices=STRATEGIES, default='random')
    args = parser.parse_args()
    if args.neg_clicks is None:
        args.neg_clicks = args.clicks
    weak = args.clicks > 0 or args.neg_clicks > 0
    if weak and args.dataset != 'ph2':
        parser.error('point supervision is only implemented for PH2')
    multi = args.dataset == 'facade'
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    supervision = f'{args.clicks}+{args.neg_clicks} clicks ({args.sampling})' if weak else 'full masks'
    print(f'Device: {device} | dataset: {args.dataset} | model: {args.model} | size: {args.size} | '
          f'loss: {args.loss} | augment: {args.augment} | seed: {args.seed} | supervision: {supervision}')

    # Dataset. Validation and test always use the full masks (they are only used for evaluation).
    size = args.size
    Dataset = DATASETS[args.dataset]
    aug = {'augment': True} if args.augment else {}
    if weak:
        trainset = PH2Clicks('train', size=size, n_pos=args.clicks, n_neg=args.neg_clicks,
                             strategy=args.sampling, seed=args.seed)
    else:
        trainset = Dataset('train', size=size, **aug)
    valset, testset = Dataset('val', size=size), Dataset('test', size=size)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=3, generator=g)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=3)
    print(f'Loaded {len(trainset)} training / {len(valset)} validation / {len(testset)} test images')

    # Training setup
    if multi:
        model = UNetMulti()
    else:
        model = UNet() if args.model == 'unet' else EncDec()
    model = model.to(device)
    opt = optim.Adam(model.parameters(), args.lr)
    if multi:
        loss_fn = nn.CrossEntropyLoss()
    elif weak:
        loss_fn = PointBCELoss()
    elif args.loss == 'focal':
        loss_fn = FocalLoss()
    elif args.loss == 'wbce':
        fg = torch.stack([y.mean() for _, y in Dataset('train', size=size)]).mean().item()
        print(f'Foreground fraction {fg:.3f} -> pos_weight {(1 - fg) / fg:.2f}')
        loss_fn = WeightedBCELoss((1 - fg) / fg)
    else:
        loss_fn = BCELoss()
    loss_fn = loss_fn.to(device)
    # validation loss is monitored with plain BCE on full masks when training on clicks
    val_loss_fn = BCELoss() if weak else loss_fn

    # Training loop (the final model is the last epoch; validation is only monitored)
    model.train()
    for epoch in range(args.epochs):
        tic = time()
        avg_loss = 0
        for X_batch, y_true in train_loader:
            X_batch, y_true = X_batch.to(device), y_true.to(device)
            opt.zero_grad()
            y_pred = model(X_batch)  # logits; sigmoid/softmax is inside the loss
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            opt.step()
            avg_loss += loss.item() / len(train_loader)

        val_metrics, val_loss = evaluate(model, val_loader, device, val_loss_fn)
        print(f'* Epoch {epoch+1}/{args.epochs} | train loss {avg_loss:.4f} | '
              f'val loss {val_loss:.4f} | val dice {val_metrics["dice"]:.4f} | {time()-tic:.1f}s')

    # Final metrics on all three splits, always against the full masks
    full_train = DataLoader(Dataset('train', size=size), batch_size=args.batch_size, shuffle=False, num_workers=3)
    results = {split: evaluate(model, loader, device)[0]
               for split, loader in [('train', full_train), ('val', val_loader), ('test', test_loader)]}
    for split, m in results.items():
        print_table(f'{split} metrics: {type(model).__name__} on {args.dataset} ({size}x{size}), {supervision}', m)
    # Native resolution (and inside the FOV for DRIVE), the numbers to report
    results_native = {}
    if not multi:
        for split in ['train', 'val', 'test']:
            results_native[split] = evaluate_native(model, Dataset(split, size=size), device)
            print_table(f'{split} metrics at native resolution' + (' inside FOV' if args.dataset == 'drive' else ''),
                        results_native[split])

    # Save weights and results next to this script (works regardless of the working directory)
    aug = '_aug' if args.augment else ''
    sup = f'_click{args.clicks}-{args.neg_clicks}_{args.sampling}' if weak else ''
    model_name = (f'{type(model).__name__}_{args.dataset}_{size}_{args.loss}{aug}{sup}'
                  f'_e{args.epochs}_bs{args.batch_size}_lr{args.lr:g}_s{args.seed}')
    os.makedirs(HERE / 'weights', exist_ok=True)
    os.makedirs(HERE / 'results', exist_ok=True)
    torch.save(model.state_dict(), HERE / 'weights' / f'{model_name}.pt')
    with open(HERE / 'results' / f'{model_name}.json', 'w') as f:
        json.dump({'args': vars(args), 'model': type(model).__name__, 'weak': weak,
                   'metrics': results, 'metrics_native': results_native}, f, indent=2)
    print(f'\nSaved weights and results as {model_name}')
    print('Training has finished!')


if __name__ == '__main__':
    main()
