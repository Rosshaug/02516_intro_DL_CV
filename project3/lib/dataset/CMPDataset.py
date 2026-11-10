import os
import glob
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


DATA_PATH = r'/dtu/datasets1/02516/CMP_facade_DB_base'

# Label PNG pixel values are mapped to 0..num_classes-1 by subtracting this.
# CMP-style facade labels are stored as 1..12 -> set to 1. If np.unique shows 0..11, set to 0.
LABEL_OFFSET = 1


def read_label_names(root=DATA_PATH):
    with open(os.path.join(root, 'label_names.txt')) as f:
        return [line.strip() for line in f if line.strip()]


class Facade(torch.utils.data.Dataset):
    """
    Multi-class segmentation dataset.

    Returns
        X: float tensor (3, size, size), values in [0, 1]
        Y: long tensor  (size, size), class indices 0..num_classes-1
    """

    def __init__(self, split='train', size=128, root=DATA_PATH,
                 val_frac=0.15, test_frac=0.15, seed=0, augment=False):
        assert split in ('train', 'val', 'test')
        self.size = size
        self.augment = augment

        # pair every image with the label that has the same file name
        image_paths = sorted(glob.glob(os.path.join(root, 'base', '*.jpg')))
        pairs = [(p, os.path.splitext(p)[0] + '.png') for p in image_paths]
        missing = [lbl for _, lbl in pairs if not os.path.exists(lbl)]
        if missing:
            raise FileNotFoundError(f'{len(missing)} labels missing, e.g. {missing[0]}')

        # the dataset has no split -> make one, reproducible via the seed
        idx = list(range(len(pairs)))
        random.Random(seed).shuffle(idx)
        n_test = int(len(idx) * test_frac)
        n_val = int(len(idx) * val_frac)
        splits = {
            'test': idx[:n_test],
            'val': idx[n_test:n_test + n_val],
            'train': idx[n_test + n_val:],
        }
        self.pairs = [pairs[i] for i in splits[split]]

        self.label_names = read_label_names(root)
        self.num_classes = len(self.label_names)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, label_path = self.pairs[idx]

        image = Image.open(image_path).convert('RGB')
        # Palette PNG: pixel values ARE the class indices.
        # Do NOT .convert('L') or ToTensor() it - that would turn them into brightness values.
        label = Image.open(label_path)

        image = TF.resize(image, [self.size, self.size])
        label = TF.resize(label, [self.size, self.size],
                          interpolation=TF.InterpolationMode.NEAREST)  # no blended labels

        # augmentation must hit image and label identically
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        X = TF.to_tensor(image)
        Y = torch.from_numpy(np.array(label, dtype=np.int64)) - LABEL_OFFSET
        return X, Y


if __name__ == '__main__':
    # sanity check: run once and look at the printed values
    ds = Facade('train')
    print('classes:', ds.num_classes, ds.label_names)
    print('train/val/test:', len(ds), len(Facade('val')), len(Facade('test')))

    X, Y = ds[0]
    print('X', X.shape, X.dtype, 'Y', Y.shape, Y.dtype)

    values = torch.cat([ds[i][1].unique() for i in range(min(20, len(ds)))]).unique()
    print('label values (after offset):', values.tolist())
    assert values.min() >= 0 and values.max() < ds.num_classes, \
        'label range does not match label_names.txt -> fix LABEL_OFFSET'