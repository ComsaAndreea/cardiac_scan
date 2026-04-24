import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.LeftRightVentricle.acdc_torchdataset import ACDCDataset


def train_acdc(X, Y, model_name, epochs=5, batch_size=4, lr=0.001):

    dataset = ACDCDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet()

    import os
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
        print(f"Model {model_name} încărcat")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for i, (images, masks) in enumerate(dataloader):

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(f"{model_name} | Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

        print(f"{model_name} | Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), model_name)
    print(f"{model_name} salvat!")

    return model