import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        stride = 1
        self.skip = nn.Sequential()

        if in_channels != out_channels:
            stride = 2
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0)
            )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.skip(x) + self.block(x)
        x = self.relu(x)

        return x

class MiniResNet_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._build_layer(16, 32)

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def _build_layer(self, in_channels, out_channels):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        x = self.avepool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)

        return x

class MiniResNet_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._build_layer(16, 16)
        self.layer2 = self._build_layer(16, 32)
        self.layer3 = self._build_layer(32, 32)

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def _build_layer(self, in_channels, out_channels):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avepool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)

        return x

class MiniResNet_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._build_layer(32, 32)
        self.layer2 = self._build_layer(32, 64)
        self.layer3 = self._build_layer(64, 64)

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def _build_layer(self, in_channels, out_channels):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avepool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)

        return x


class MiniResNet_v4(nn.Module):
    # Sora
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._build_layer(32, 32)
        self.layer2 = self._build_layer(32, 32)
        self.layer3 = self._build_layer(32, 64)
        self.layer4 = self._build_layer(64, 64)

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def _build_layer(self, in_channels, out_channels):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avepool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)


        return x


class MiniResNet_v5(nn.Module):
    # Sora
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._build_layer(16, 16)
        self.layer2 = self._build_layer(16, 16)
        self.layer3 = self._build_layer(16, 32)
        self.layer4 = self._build_layer(32, 32)

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def _build_layer(self, in_channels, out_channels):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avepool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)


        return x


class MiniResNet_v6(nn.Module):
    # Sora
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._build_layer(32, 32)
        self.layer2 = self._build_layer(32, 32)
        self.layer3 = self._build_layer(32, 64)
        self.layer4 = self._build_layer(64, 64)
        self.layer5 = self._build_layer(64, 64)

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def _build_layer(self, in_channels, out_channels):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.avepool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)


        return x


