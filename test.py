import time

import torch
from torch import nn


class Student(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age


# s = Student('xiaoming', 18)
# s.math_score = torch.rand((100, 2))
# # print(s.math_score)
#
# a = nn.Conv2d(kernel_size=(3, 3), in_channels=4, out_channels=6)
# if hasattr(a, 'weight'):
#     print(a.bias.shape)
#     print(a.weight.shape)
# b = nn.BatchNorm2d(num_features=10)
# if hasattr(b, 'weight'):
#     print(b.weight.shape)
#     print(b.bias.shape)
# c = nn.Linear(in_features=10, out_features=20)
# if hasattr(c, 'weight'):
#     print(c.weight.shape)
#     print(c.bias.shape)

# model = get_model(model="resnet18").cuda()
# weight_sparsity(model, 4)
# H, W = (16, 8)
# input = torch.rand(H, W)
# print(input)
# mask = torch.ones(H,W)
# mask[input>0.3] = 0
# print(mask)
# mask = mask.type(torch.bool)
# input[mask] =0
# print(input)
def sparsify(x, patterns=[(2, 1)]):
    # assuming x NCHW
    N, C, H, W = x.shape

    with torch.no_grad():
        # torch.ones_like can guarantee mask and x on the same device
        mask = torch.ones_like(x)

        for pattern in patterns:
            m, n = pattern
            masked_x = (mask * x).reshape(N, C // m, m, H, W).abs_()
            mask = mask.view(N, C // m, m, H, W)
            _, idx = torch.topk(masked_x, m - n, dim=2, largest=False)
            mask.scatter_(2, idx, 0.0)
            mask = mask.view(N, C, H, W)

    return mask * x


def sparsify2(x, patterns=[(2, 1)]):
    # assuming x NCHW
    N, C, H, W = x.shape

    with torch.no_grad():
        # torch.ones_like can guarantee mask and x on the same device
        mask = torch.ones_like(x)

        for pattern in patterns:
            m, n = pattern
            masked_x = (mask * x).reshape(N, C // m, m, H, W).abs_()
            mask = mask.view(N, C // m, m, H, W)
            val, _ = torch.topk(masked_x, n, dim=2, largest=True)
            val = val[:, :, n - 1:, :, :]
            idx = masked_x < val
            mask[idx] = 0
            mask = mask.view(N, C, H, W)

    return mask * x


def main():
    torch.cuda.set_device(1)
    x = torch.randn(192, 32, 8, 8).cuda()
    conv = nn.Conv2d(32, 32, 1, 1).cuda()
    start = time.time()
    sparse_x = sparsify2(x, patterns=[(4, 1)])
    print(time.time() - start)

    start = time.time()
    sparse_x2 = sparsify2(x, patterns=[(4, 1)])
    print(time.time() - start)

    start = time.time()
    sparse_x1 = sparsify(x, patterns=[(4, 1)])
    print(time.time() - start)

    print((sparse_x2 == sparse_x1).all())


if __name__ == "__main__":
    main()
