import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(64, 88)

    def forward(self, x):
        # x: (batch, time, freq)

        x = x.unsqueeze(1)   # → (batch, 1, time, freq)

        x = self.conv(x)     # → (batch, 64, time, freq)

        x = x.mean(dim=3)    # collapse freq → (batch, 64, time)

        x = x.permute(0, 2, 1)  # → (batch, time, 64)

        x = self.fc(x)       # → (batch, time, 88)

        return x