import pickle
from collections import Counter

import numpy as np
import torch
from torch.autograd import Function
from torch import nn, Tensor

from util import constant
from util.myStatictis import Statistics


def student2dict(self):
    total = self.num[0] + self.num[1] + self.num[2] + self.num[3] + self.num[4]
    return {
        '0': self.num[0] * 1.0 / total,
        '1': self.num[1] * 1.0 / total,
        '2': self.num[2] * 1.0 / total,
        '3': self.num[3] * 1.0 / total,
        '4': self.num[4] * 1.0 / total,
        'N': self.N
    }


def wei2dict(self):
    return {
        'in 1 block': self.one,
        'in 2 block': self.two,
    }


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


class MyConvFunction(Function):
    @staticmethod
    def forward(ctx, x, w, b, stride, padding, name):
        # N C H W
        x = x.cpu().numpy()
        w = w.cpu().numpy()
        # b = b.cpu().numpy()
        stride, pad = stride, padding
        N, C, H, W = x.shape
        F, HH, WW = w.shape[0], w.shape[2], w.shape[3]
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        H_out = 1 + (H + 2 * pad - HH) // stride
        W_out = 1 + (W + 2 * pad - WW) // stride
        out = np.zeros((N, F, H_out, W_out))
        sta = [0, 0, 0, 0, 0]
        print(1e-8)
        for idx_N in range(N):
            for idx_f in range(F):
                idx_yy = 0
                for idx_y in range(H_out):
                    idx_xx = 0
                    for idx_x in range(W_out):
                        tmp = np.multiply(w[idx_f], x_pad[idx_N, :, idx_yy:idx_yy + HH, idx_xx:idx_xx + WW])
                        for i in range(HH):
                            for j in range(WW):
                                for k in range(0, C, 8):
                                    sta[np.sum(np.abs(tmp[k:k + 8, i, j]) > 1e-8)] += 1
                        out[idx_N, idx_f, idx_y, idx_x] = np.sum(tmp)
                        idx_xx += stride
                    idx_yy += stride
        out = torch.tensor(out.astype(np.float32)).cuda()
        if name in constant.valid_cal:
            constant.valid_cal[name].update(sta)
        else:
            constant.valid_cal[name] = Statistics()
            constant.valid_cal[name].update(sta)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


myConvFunc = MyConvFunction.apply


class MyConv(nn.Module):
    def __init__(self, w, b, name, stride, padding):
        super().__init__()
        self.w = w
        self.b = b
        self.name = name
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return myConvFunc(x, self.w, self.b, self.stride, self.padding, self.name)


def changeConv(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            tmp = MyConv(w=module.weight, b=module.bias, name=name, stride=module.stride[0],
                         padding=module.padding[0]).cuda()
            _set_module(model, name, tmp)
