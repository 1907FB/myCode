import torch
from torch.nn import functional as F


class SparseLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.mask = torch.nn.Parameter(torch.ones_like(self.weight), requires_grad=False)

    def forward(self, x):
        sparse_weight = self.mask * self.weight
        return F.linear(x, sparse_weight, self.bias)

    def set_mask(self, mask):
        self.mask.data = mask
