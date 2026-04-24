# aici antrenam modelul
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.unet import UNet
from src.LeftAtrium.torchdataset import HeartDataset

def train_model(X, Y, epochs=5, batch_size=4, lr=0.001):
    # dataset
    dataset = HeartDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = UNet()
    import os
    if os.path.exists("model_leftatrium.pth"):
        model.load_state_dict(torch.load("model_leftatrium.pth"))
        print("Model încărcat pentru continuarea antrenării")

    # loss + optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, masks) in enumerate(dataloader):
            # forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 50 == 0:
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model_leftatrium.pth")
    print("Model salvat!")
    return model