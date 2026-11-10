import torch
import torch.nn.functional as F

METRICS = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']


def confusion(logits, y_true):
    # If binary, use masks
    if logits.shape[1] == 1:
        pred, true = logits > 0, y_true > 0.5
    else:
        C = logits.shape[1]
        pred = F.one_hot(logits.argmax(1), C).permute(0, 3, 1, 2).bool()
        true = F.one_hot(y_true.long(), C).permute(0, 3, 1, 2).bool()
    dims = (0, 2, 3)
    tp = (pred & true).sum(dims)
    fp = (pred & ~true).sum(dims)
    fn = (~pred & true).sum(dims)
    tn = (~pred & ~true).sum(dims)
    return torch.stack([tp, fp, fn, tn]).cpu().double()


def metrics(counts, eps=1e-8):
    tp, fp, fn, tn = counts
    m = {
        'dice': 2 * tp / (2 * tp + fp + fn + eps),
        'iou': tp / (tp + fp + fn + eps),
        'accuracy': (tp + tn) / (tp + fp + fn + tn + eps),
        'sensitivity': tp / (tp + fn + eps),
        'specificity': tn / (tn + fp + eps),
    }
    return {k: v.mean().item() for k, v in m.items()}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    counts, loss = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        counts = counts + confusion(logits, y)
        if loss_fn is not None:
            loss += loss_fn(logits, y).item() / len(loader)
    model.train()
    return metrics(counts), loss


def print_table(title, results):
    print(f'\n{title}')
    print(' | '.join(f'{k:>11}' for k in METRICS))
    print(' | '.join(f'{results[k]:>11.4f}' for k in METRICS))
