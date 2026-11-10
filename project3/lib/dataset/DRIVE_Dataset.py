import os
import glob
import random
import torch
from PIL import Image
import torchvision.transforms as T

DATA_PATH = '/dtu/datasets1/02516/DRIVE'
if not os.path.isdir(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../../../data/DRIVE')


class DRIVE(torch.utils.data.Dataset):
    def __init__(self, split='train', size=128):
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
        return T.functional.to_tensor(self.resize(image)), T.functional.to_tensor(self.resize(label))
