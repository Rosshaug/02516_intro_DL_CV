import os
from pathlib import Path
from time import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet
from lib.model.UNetMulti import UNet as UNetMulti
from lib.losses import BCELoss, FocalLoss, WeightedBCELoss
from lib.dataset.PH2Dataset import PH2
from lib.dataset.DRIVE_Dataset import DRIVE
from lib.dataset.CMPDataset import Facade
from measure import evaluate, print_table

DATASETS = {'ph2': PH2, 'drive': DRIVE, 'facade': Facade}


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
    args = parser.parse_args()
    multi = args.dataset == 'facade'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device} | dataset: {args.dataset} | model: {args.model} | size: {args.size} | '
          f'loss: {args.loss} | augment: {args.augment}')

    # Dataset
    size = args.size
    Dataset = DATASETS[args.dataset]
    aug = {'augment': True} if args.augment else {}
    trainset, valset = Dataset('train', size=size, **aug), Dataset('val', size=size)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=3)
    print(f'Loaded {len(trainset)} training / {len(valset)} validation images')

    # Training setup
    if multi:
        model = UNetMulti()
    else:
        model = UNet() if args.model == 'unet' else EncDec()
    model = model.to(device)
    opt = optim.Adam(model.parameters(), args.lr)
    if multi:
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        loss_fn = FocalLoss()
    elif args.loss == 'wbce':
        fg = torch.stack([y.mean() for _, y in Dataset('train', size=size)]).mean().item()
        print(f'Foreground fraction {fg:.3f} -> pos_weight {(1 - fg) / fg:.2f}')
        loss_fn = WeightedBCELoss((1 - fg) / fg)
    else:
        loss_fn = BCELoss()
    loss_fn = loss_fn.to(device)

    # Training loop
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

        val_metrics, val_loss = evaluate(model, val_loader, device, loss_fn)
        print(f'* Epoch {epoch+1}/{args.epochs} | train loss {avg_loss:.4f} | '
              f'val loss {val_loss:.4f} | val dice {val_metrics["dice"]:.4f} | {time()-tic:.1f}s')

    # Final validation metrics
    val_metrics, _ = evaluate(model, val_loader, device)
    print_table(f'Validation metrics: {type(model).__name__} on {args.dataset} ({size}x{size}), loss {args.loss}, augment {args.augment}', val_metrics)

    # Save the model next to this script (works regardless of the working directory)
    out_path = Path(__file__).resolve().parent / 'weights'
    os.makedirs(out_path, exist_ok=True)
    aug = '_aug' if args.augment else ''
    model_name = (f'{type(model).__name__}_{args.dataset}_{size}_{args.loss}{aug}'
                  f'_e{args.epochs}_bs{args.batch_size}_lr{args.lr:g}.pt')
    torch.save(model.state_dict(), out_path / model_name)
    print(f'\nSaved weights to {out_path / model_name}')
    print('Training has finished!')


if __name__ == '__main__':
    main()
