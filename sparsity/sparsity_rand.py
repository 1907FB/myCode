import pickle

import torch
from torch.autograd import Function
from torch import nn, Tensor

forward_hooks = []
idx = 0


class SparsityFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # N C H W
        if len(input.shape) == 4:
            N, C, H, W = input.shape
            b = torch.reshape(input, (N, C // 4, 4, H, W))
            mask = torch.randint(b.shape).cuda()

            a, _ = torch.topk(torch.abs(b), 2, dim=2)
            a = a[:, :, 1:, :, :]
            tmp = torch.abs(b) < a
            mask[tmp] = 0
            mask = torch.reshape(mask, (N, C, H, W))
        else:
            N, D = input.shape
            if D % 4 != 0:
                mask = torch.ones(input.shape).cuda()
            else:
                b = torch.reshape(input, (N, D // 4, 4))
                mask = torch.ones(b.shape).cuda()
                a, _ = torch.topk(torch.abs(b), 2, dim=2)
                a = a[:, :, 1:]
                tmp = torch.abs(b) < a
                mask[tmp] = 0
                mask = torch.reshape(mask, (N, D))
        ctx.save_for_backward(mask)
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        return grad_output * mask


sparsityNor = SparsityFunction.apply


class Sparsity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return sparsity(input)
