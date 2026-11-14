import os
import glob
import random
import torch
from PIL import Image
import torchvision.transforms as T

from lib.clicks import sample_clicks, click_target

DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images'
if not os.path.isdir(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../../../data/PH2')


class PH2(torch.utils.data.Dataset):
    def __init__(self, split='train', size=128):
        cases = sorted(glob.glob(os.path.join(DATA_PATH, 'IMD*')))
        random.Random(0).shuffle(cases)
        self.cases = {'train': cases[:140], 'val': cases[140:170], 'test': cases[170:]}[split]
        self.resize = T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        name = os.path.basename(self.cases[idx])
        image = Image.open(f'{self.cases[idx]}/{name}_Dermoscopic_Image/{name}.bmp').convert('RGB')
        label = Image.open(f'{self.cases[idx]}/{name}_lesion/{name}_lesion.bmp').convert('L')
        return T.functional.to_tensor(self.resize(image)), T.functional.to_tensor(self.resize(label))

    def native(self, idx):
        """Full-resolution label (H, W) and field-of-view mask (None: whole image)."""
        name = os.path.basename(self.cases[idx])
        label = Image.open(f'{self.cases[idx]}/{name}_lesion/{name}_lesion.bmp').convert('L')
        return T.functional.to_tensor(label)[0], None


class PH2Clicks(PH2):
    """PH2 with point clicks instead of masks (for training only).

    Same images and split as PH2. The clicks are sampled once in __init__ from the full mask
    (seeded per image) and stay fixed across epochs. The label returned is a (2, H, W) tensor:
    channel 0 holds the click labels, channel 1 marks which pixels were clicked.
    """

    def __init__(self, split='train', size=128, n_pos=5, n_neg=5, strategy='random', seed=0):
        super().__init__(split, size)
        self.clicks = []
        for i in range(len(self)):
            _, y = super().__getitem__(i)
            gen = torch.Generator().manual_seed(seed * 100_003 + i)
            self.clicks.append(sample_clicks(y[0] > 0.5, n_pos, n_neg, strategy, gen))
        self.shape = (size, size)

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        return image, click_target(*self.clicks[idx], self.shape)
