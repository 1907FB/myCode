import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024, bias=False),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024, bias=False),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 10, bias=False),
        )

    # 修改forward
    def forward(self, x):
        x = self.classifier(x)
        return x


def mlp():
    return MLP()
