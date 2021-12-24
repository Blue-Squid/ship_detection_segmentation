import os
import numpy as np
import pandas as pd
from PIL import Image

from tiny_utils import rle_decode
from torch.utils.data import Dataset



class CreateDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(Image.open("data/train_v2/" + self.images[index]).convert("RGB"))
        mask = rle_decode(self.masks[index], (768, 768))
    

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask