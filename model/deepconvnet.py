import math
import torch.nn as nn


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()

        self._init_hyperparams()

        self.temporal = nn.Conv2d(
            1, 25, kernel_size=[1, self.kernel_size], padding="valid"
        )
        self.spatial = nn.Conv2d(
            25, 25, kernel_size=[self.num_channels, 1], padding="valid"
        )

        self.block_1 = nn.Sequential(
            self.temporal,
            self.spatial,
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 3], stride=[1, 3]),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=[1, self.kernel_size], padding="valid"),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 3], stride=[1, 3]),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=[1, self.kernel_size], padding="valid"),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 3], stride=[1, 3]),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=[1, self.kernel_size], padding="valid"),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 3], stride=[1, 3]),
        )

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def _init_hyperparams(self):
        self.num_channels = 22
        self.kernel_size = 5
        self.input_temporal_size = 250
        self.hidden_dim = self.compute_hidden_dim(self.input_temporal_size)
        self.num_classes = 4

    def compute_hidden_dim(self, t):
        t = t - self.kernel_size + 1
        t = math.floor((t - 3) / 3) + 1
        t = math.floor((t - self.kernel_size + 1 - 3) / 3) + 1
        t = math.floor((t - self.kernel_size + 1 - 3) / 3) + 1
        t = math.floor((t - self.kernel_size + 1 - 3) / 3) + 1

        return int(4200 * t)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        x = self.cls_head(x)

        return x
