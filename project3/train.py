import os
from pathlib import Path
from time import time
import argparse
import uuid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.model.EncDecModel import EncDec
from lib.model.UNetMulti import UNet as UNetMulti
from lib.losses import BCELoss  # , DiceLoss, FocalLoss, BCELoss_TotalVariation
from lib.dataset.PH2Dataset import PH2
from lib.dataset.DRIVE_Dataset import DRIVE
from lib.dataset.CMPDataset import Facade
from measure import evaluate, print_table

DATASETS = {'ph2': PH2, 'drive': DRIVE, 'facade': Facade}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default='ph2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    multi = args.dataset == 'facade'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device} | dataset: {args.dataset}')

    # Dataset
    size = 128  # EncDec is hardcoded to 128x128
    Dataset = DATASETS[args.dataset]
    trainset, valset = Dataset('train', size=size), Dataset('val', size=size)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=3)
    print(f'Loaded {len(trainset)} training / {len(valset)} validation images')

    # Training setup
    model = (UNetMulti() if multi else EncDec()).to(device)
    opt = optim.Adam(model.parameters(), args.lr)
    loss_fn = nn.CrossEntropyLoss() if multi else BCELoss()

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
    print_table(f'Validation metrics: {type(model).__name__} on {args.dataset}', val_metrics)

    # Save the model next to this script (works regardless of the working directory)
    out_path = Path(__file__).resolve().parent / 'weights'
    os.makedirs(out_path, exist_ok=True)
    model_name = f'{type(model).__name__}_{args.dataset}_{uuid.uuid4().hex[:8]}.pt'
    torch.save(model.state_dict(), out_path / model_name)
    print(f'\nSaved weights to {out_path / model_name}')
    print('Training has finished!')


if __name__ == '__main__':
    main()
