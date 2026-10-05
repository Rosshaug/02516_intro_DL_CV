from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

# ImageNet statistics, only used to normalize the input (no pretrained weights).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train, img_size=112):
    """Transforms operating on tensors of shape [..., C, H, W] (a frame or a stacked video)."""
    if train:
        return T.Compose([
            T.Resize(int(img_size * 8 / 7)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize(int(img_size * 8 / 7)),
        T.CenterCrop(img_size),
        T.Normalize(MEAN, STD),
    ])


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/work3/ppar/data/ucf101',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = os.path.basename(os.path.dirname(frame_path))
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = T.ToTensor()(Image.open(frame_path).convert("RGB"))
        if self.transform:
            frame = self.transform(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101', 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        class_name = os.path.basename(os.path.dirname(video_path))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = os.path.join(self.root_dir, 'frames', self.split, class_name, video_name)
        video_frames = self.load_frames(video_frames_dir)

        # Stack to [T, C, H, W] before transforming, so random crops/flips are
        # identical for every frame of the video (no fake motion between frames).
        frames = torch.stack([T.ToTensor()(frame) for frame in video_frames])
        if self.transform:
            frames = self.transform(frames)

        if self.stack_frames:
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
        else:
            frames = list(frames)

        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/work3/ppar/data/ucf101'

    transform = get_transforms(train=False, img_size=64)
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
            
