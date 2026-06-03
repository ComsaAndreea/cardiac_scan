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

        self.bottleneck = DoubleConv(128, 256)

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


class UNetPlusPlusMulticlass(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        filters = [64, 128, 256]

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv0_0 = DoubleConv(1, filters[0])
        self.conv1_0 = DoubleConv(filters[0], filters[1])
        self.conv2_0 = DoubleConv(filters[1], filters[2])

        self.conv0_1 = DoubleConv(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv(filters[1] + filters[2], filters[1])

        self.conv0_2 = DoubleConv(filters[0] * 2 + filters[1], filters[0])

        self.out = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x0_1 = self.conv0_1(
            torch.cat([x0_0, self.up(x1_0)], dim=1)
        )

        x1_1 = self.conv1_1(
            torch.cat([x1_0, self.up(x2_0)], dim=1)
        )

        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1)
        )

        return self.out(x0_2)


def get_model(model_type, num_classes=4):
    model_type = model_type.lower()

    if model_type == "unet":
        return UNetMulticlass(num_classes=num_classes)

    if model_type == "attention_unet":
        return AttentionUNetMulticlass(num_classes=num_classes)

    if model_type in ["unetpp", "unet++"]:
        return UNetPlusPlusMulticlass(num_classes=num_classes)

    raise ValueError(
        "model_type trebuie să fie 'unet', 'attention_unet' sau 'unetpp'"
    )