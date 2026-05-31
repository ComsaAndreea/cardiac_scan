import random

import torch
from torch.utils.data import Dataset

from src.utils.preprocessing import normalize_image, pad_to_size


class SAXLRVDataset(Dataset):
    def __init__(self, X, Y, size=(512, 512), augment=False):
        self.X = X
        self.Y = Y
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def augment_sample(self, image, mask):
        if random.random() < 0.5:
            brightness = random.uniform(-0.08, 0.08)
            contrast = random.uniform(0.90, 1.10)
            image = image * contrast + brightness
            image = torch.clamp(image, 0.0, 1.0)

        if random.random() < 0.3:
            noise = torch.randn_like(image) * 0.025
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image, mask

    def __getitem__(self, idx):
        image = self.X[idx]
        mask = self.Y[idx]

        image = pad_to_size(image, self.size, value=0)
        mask = pad_to_size(mask, self.size, value=0)

        image = normalize_image(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.long)

        if self.augment:
            image, mask = self.augment_sample(image, mask)

        return image, mask