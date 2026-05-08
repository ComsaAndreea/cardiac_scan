import torch
from torch.utils.data import Dataset
import numpy as np

from src.utils.preprocessing import normalize_image, pad_to_size


class ACDCDataset(Dataset):
    def __init__(self, X, Y, size=(512, 512)):
        self.X = X
        self.Y = Y
        self.size = size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        mask = self.Y[idx]

        image = pad_to_size(image, self.size, value=0)
        mask = pad_to_size(mask, self.size, value=0)

        image = normalize_image(image)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return image, mask