import torch
import torch.nn as nn
import torch.nn.functional as F
from bitmap import GaussianRBF, centers

def scaled_tanh(x):
    return 1.7159 * torch.tanh((2/3) * x)

class LearnedAvgPool(nn.Module): #S2 and S4
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # Average pool 2×2 stride 2
        pooled = F.avg_pool2d(x, kernel_size=2, stride=2)

        # Apply learned α and β to each map
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        return scaled_tanh(w * pooled + b)

C3_CONNECTIONS = [
    [0,1,2], [1,2,3], [2,3,4], [3,4,5], [0,4,5], [0,1,5],
    [0,1,2,3], [1,2,3,4], [2,3,4,5], [0,3,4,5], [0,1,4,5], [0,1,2,5],
    [0,1,3,4], [1,2,4,5], [0,2,3,5],
    [0,1,2,3,4,5]
]

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # C1:
        # Input: 32×32×1
        # Output: 6x28×28
        self.C1 = nn.Conv2d(1, 6, kernel_size=5)

        # S2:
        # Input: 6×28×28
        # Output: 6×14×14
        self.S2 = LearnedAvgPool(6)

        # C3:
        # Input: 6×14×14
        # Output: 16×10×10
        self.C3 = nn.ModuleList([
            nn.Conv2d(len(C3_CONNECTIONS[i]), 1, kernel_size=5)
            for i in range(16)
        ])

        # S4:
        # Input: 16×10×10
        # Output: 16×5×5
        self.S4 = LearnedAvgPool(16)

        # C5:
        # Input: 16×5×5
        # Output: 120
        self.C5 = nn.Conv2d(16, 120, kernel_size=5)

        # F6:
        # Input: 120
        # Output: 84
        self.F6 = nn.Linear(120, 84)

        # Output layer: Gaussian RBF (84 → 10)
        self.output = GaussianRBF(centers, sigma=1.0)

    def forward(self, x):

        # C1
        x = scaled_tanh(self.C1(x))     # shape: N × 6 × 28 × 28

        # S2
        x = self.S2(x)                 # shape: N × 6 × 14 × 14

        # C3 (partial connections)
        C3_maps = []
        for i, conv in enumerate(self.C3):
            selected = x[:, C3_CONNECTIONS[i], :, :]  # pick channels
            C3_maps.append(scaled_tanh(conv(selected)))

        x = torch.cat(C3_maps, dim=1)  # shape: N × 16 × 10 × 10

        # S4
        x = self.S4(x)                 # shape: N × 16 × 5 × 5

        # C5
        x = scaled_tanh(self.C5(x))     # shape: N × 120 × 1 × 1
        x = x.view(x.size(0), -1)      # N × 120

        # F6
        f6 = scaled_tanh(self.F6(x))     # N × 84

        # Output layer:
        out = self.output(f6)          # N × 10 (negative distances)

        return out