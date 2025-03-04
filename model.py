import torch
from torch import nn
from torch.nn import functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 10)
        self.elu = nn.ELU()
        # self.dropout = nn.Dropout(0.3)  # Pode testar valores entre 0.2 e 0.5

    def forward(self, x):
        x = self.elu(self.bn1(self.conv1(x)))
        # x = self.pool(x)
        x = self.elu(self.bn2(self.conv2(x)))
        # x = self.pool(x)
        x = self.elu(self.bn3(self.conv3(x)))
        # x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.elu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    m = SimpleCNN()
    img = torch.randn(1, 3, 32, 32)
    print(m(img).shape)
