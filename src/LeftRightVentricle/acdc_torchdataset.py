import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class ACDCDataset(Dataset):
    def __init__(self, X, Y, size=(256, 256)):
        self.X = X
        self.Y = Y
        self.size = size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        mask = self.Y[idx]

        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return image, mask