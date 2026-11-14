"""Simulated point-click annotations for weakly supervised segmentation.

A click is a single labelled pixel: positive (inside the lesion) or negative (background).
Clicks are sampled once per image from the full mask at training resolution and then kept
fixed, like a real annotator who clicks each image once.

Sampling strategies
  random   : uniform inside / outside the mask.
  interior : annotator-like. Positives favour the lesion centre (probability grows with the
             distance to the boundary, nothing within EDGE px of it). Negatives fall on the
             skin around the lesion (between EDGE and NEAR px outside it), not in the dark
             corners outside the dermoscope that nobody would click on.
  boundary : clicks close to the boundary on both sides (within BAND px), the most
             informative but least realistic placement.
"""
import torch
import torch.nn.functional as F

STRATEGIES = ['random', 'interior', 'boundary']
EDGE, NEAR, BAND = 3, 40, 10  # pixels at the 256x256 training resolution


def distance_to_boundary(mask):
    """Chessboard distance (px) from every True pixel to the nearest False pixel (0 on False)."""
    cur = mask.float()[None, None]
    dist = torch.zeros_like(cur)
    for _ in range(max(mask.shape)):
        if not cur.any():
            break
        dist += cur
        cur = -F.max_pool2d(-cur, 3, stride=1, padding=1)  # erosion by one pixel
    return dist[0, 0]


def _weights(mask, strategy):
    d_in, d_out = distance_to_boundary(mask), distance_to_boundary(~mask)
    if strategy == 'random':
        return mask.float(), (~mask).float()
    if strategy == 'interior':
        return (d_in - EDGE).clamp(min=0), ((d_out > EDGE) & (d_out <= NEAR)).float()
    if strategy == 'boundary':
        return ((d_in > 0) & (d_in <= BAND)).float(), ((d_out > 0) & (d_out <= BAND)).float()
    raise ValueError(strategy)


def _draw(w, fallback, n, gen):
    if w.sum() == 0:  # e.g. a lesion too small for the interior margin
        w = fallback
    n = min(n, int((w > 0).sum()))
    if n == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.multinomial(w.flatten(), n, replacement=False, generator=gen)


def sample_clicks(mask, n_pos, n_neg, strategy='random', gen=None):
    """mask: (H, W) bool. Returns flat pixel indices (pos_idx, neg_idx)."""
    w_pos, w_neg = _weights(mask, strategy)
    pos = _draw(w_pos, mask.float(), n_pos, gen)
    neg = _draw(w_neg, (~mask).float(), n_neg, gen)
    return pos, neg


def click_target(pos, neg, shape):
    """Builds the (2, H, W) training target: channel 0 = label, channel 1 = 1 where clicked."""
    target = torch.zeros(shape).flatten()
    known = torch.zeros(shape).flatten()
    target[pos] = 1
    known[pos] = 1
    known[neg] = 1
    return torch.stack([target.view(shape), known.view(shape)])
