import torch


class SparseConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)
        self.mask = torch.nn.Parameter(torch.ones_like(self.weight), requires_grad=False)

    def forward(self, x):
        sparse_weight = self.mask * self.weight
        return self._conv_forward(x, sparse_weight, self.bias)

    def set_mask(self, mask):
        self.mask.data = mask
