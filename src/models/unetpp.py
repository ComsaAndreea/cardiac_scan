import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        nb_filter = [64, 128, 256]

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

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

        return self.final(x0_2)