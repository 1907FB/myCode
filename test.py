import torch
from torch import nn
from apex.contrib.sparsity import ASP

from model.get_model import get_model
from sparsity.weight_sparsity import weight_sparsity


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
H, W = (16, 8)
input = torch.rand(H, W)
print(input)
mask = torch.ones(H,W)
mask[input>0.3] = 0
print(mask)
mask = mask.type(torch.bool)
input[mask] =0
print(input)
