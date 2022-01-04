import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import rgb2lab

SIZE = 256
class LabDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img_lab = np.array(img)
        img_lab = rgb2lab(img_lab).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)
