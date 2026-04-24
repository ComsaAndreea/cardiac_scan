import torch
from torch.utils.data import Dataset
import numpy as np

class HeartDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        mask = self.Y[idx]

        # normalizare imagine (important!)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # transformare în tensor
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # adăugăm canal (1, H, W)
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return image, mask