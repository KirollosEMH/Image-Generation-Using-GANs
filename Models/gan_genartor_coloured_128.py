import torch
import torch.nn as nn


class Gan_Generator_Colured(torch.nn.Module):
    def __init__(self, z_dim):
        super(Gan_Generator_Colured, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 16384),  # Output size to match 128x128 image
            nn.Tanh()
        )

    def forward(self, x):
        # Reshape to N x 1 x 128 x 128
        return self.gen(x).view(-1, 3, 128, 128)
