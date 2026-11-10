import os
import glob
import random
import torch
from PIL import Image
import torchvision.transforms as T

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
