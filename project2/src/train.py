"""Train one model with one seed, pick the best epoch on val, evaluate on val + test.

Example:
    python src/train.py --model late_fusion --seed 0
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import FrameVideoDataset, get_transforms
from models import MODELS, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--root_dir', default='/dtu/datasets1/02516/ucf101_noleakage') # /dtu/datasets1/02516/ucf101
    p.add_argument('--out_dir', default='results')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--img_size', type=int, default=112)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_and_probs(logits, labels):
    """Handles both video logits [B, C] and per-frame logits [B, T, C]."""
    if logits.dim() == 3:
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, C), labels.repeat_interleave(T))
        probs = logits.softmax(-1).mean(1)        # average frame predictions
    else:
        loss = F.cross_entropy(logits, labels)
        probs = logits.softmax(-1)
    return loss, probs


def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss, preds, labels_all = 0.0, [], []
    with torch.set_grad_enabled(train):
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            loss, probs = loss_and_probs(model(videos), labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds.append(probs.argmax(1).cpu())
            labels_all.append(labels.cpu())
    preds, labels_all = torch.cat(preds), torch.cat(labels_all)
    acc = (preds == labels_all).float().mean().item()
    return total_loss / len(labels_all), acc, preds.tolist(), labels_all.tolist()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    run_name = f'{args.model}_seed{args.seed}'

    def loader(split, train):
        ds = FrameVideoDataset(args.root_dir, split, get_transforms(train, args.img_size))
        return DataLoader(ds, args.batch_size, shuffle=train, num_workers=args.num_workers)

    train_loader, val_loader, test_loader = loader('train', True), loader('val', False), loader('test', False)

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{run_name}: {n_params:,} parameters, device={device}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = os.path.join(args.out_dir, f'{run_name}.pt')
    best_val_acc, best_epoch, history = -1.0, -1, []
    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))
        print(f'epoch {epoch:3d}  train loss {train_loss:.3f} acc {train_acc:.3f}  '
              f'val loss {val_loss:.3f} acc {val_acc:.3f}')
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt_path)

    # Final evaluation with the best-on-val checkpoint. Test is only touched here.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, val_acc, _, _ = run_epoch(model, val_loader, device)
    _, test_acc, test_preds, test_labels = run_epoch(model, test_loader, device)
    print(f'best epoch {best_epoch}: val acc {val_acc:.3f}, test acc {test_acc:.3f}')

    result = dict(model=args.model, seed=args.seed, n_params=n_params, best_epoch=best_epoch,
                  val_acc=val_acc, test_acc=test_acc, test_preds=test_preds,
                  test_labels=test_labels, history=history, args=vars(args))
    with open(os.path.join(args.out_dir, f'{run_name}.json'), 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
