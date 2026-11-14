import os
import glob
import random
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

DATA_PATH = '/dtu/datasets1/02516/DRIVE'
if not os.path.isdir(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../../../data/DRIVE')


class DRIVE(torch.utils.data.Dataset):
    def __init__(self, split='train', size=128, augment=False):
        self.augment = augment
        # test/ has no vessel labels, so split the 20 labelled training images
        images = sorted(glob.glob(os.path.join(DATA_PATH, 'training/images/*.tif')))
        random.Random(0).shuffle(images)
        self.images = {'train': images[:12], 'val': images[12:16], 'test': images[16:]}[split]
        self.resize = T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        sid = os.path.basename(path).split('_')[0]
        image = Image.open(path).convert('RGB')
        label = Image.open(os.path.join(DATA_PATH, f'training/1st_manual/{sid}_manual1.gif')).convert('L')
        image, label = self.resize(image), self.resize(label)
        if self.augment:
            image, label = augment(image, label)
        return TF.to_tensor(image), TF.to_tensor(label)

    def native(self, idx):
        """Full-resolution vessel label (H, W) and field-of-view mask (H, W, bool)."""
        sid = os.path.basename(self.images[idx]).split('_')[0]
        label = Image.open(os.path.join(DATA_PATH, f'training/1st_manual/{sid}_manual1.gif')).convert('L')
        fov = Image.open(os.path.join(DATA_PATH, f'training/mask/{sid}_training_mask.gif')).convert('L')
        return TF.to_tensor(label)[0], TF.to_tensor(fov)[0] > 0.5


def augment(image, label):
    # geometric: same transform for image and label (retina is round -> any rotation is valid)
    if random.random() < 0.5:
        image, label = TF.hflip(image), TF.hflip(label)
    if random.random() < 0.5:
        image, label = TF.vflip(image), TF.vflip(label)
    angle = random.uniform(-180, 180)
    image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
    label = TF.rotate(label, angle, interpolation=T.InterpolationMode.NEAREST)
    # photometric: image only
    image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
    image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
    image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
    return image, label
