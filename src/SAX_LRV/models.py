import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNetMulticlass(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))

        x3 = self.bottleneck(self.pool2(x2))

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        return self.out(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()

        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.skip_conv = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, gate, skip):
        g1 = self.gate_conv(gate)
        x1 = self.skip_conv(skip)

        attention = self.relu(g1 + x1)
        attention = self.psi(attention)

        return skip * attention


class AttentionUNetMulticlass(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        #self.bottleneck = DoubleConv(128, 256)
        self.bottleneck = nn.Sequential(
            DoubleConv(128, 256),
            nn.Dropout2d(0.2)
        )

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att1 = AttentionGate(128, 128, 64)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionGate(64, 64, 32)
        self.conv2 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))

        x3 = self.bottleneck(self.pool2(x2))

        x = self.up1(x3)
        x2_att = self.att1(gate=x, skip=x2)
        x = torch.cat([x, x2_att], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x1_att = self.att2(gate=x, skip=x1)
        x = torch.cat([x, x1_att], dim=1)
        x = self.conv2(x)

        return self.out(x)


def get_model(model_type, num_classes=4):
    if model_type == "unet":
        return UNetMulticlass(num_classes=num_classes)

    if model_type == "attention_unet":
        return AttentionUNetMulticlass(num_classes=num_classes)

    raise ValueError("model_type trebuie să fie 'unet' sau 'attention_unet'")