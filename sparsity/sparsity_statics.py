import pickle
import time

import torch
from torch.autograd import Function
from torch import nn, Tensor

from util import constant
from util.myStatictis import Statistics, WeightSta
from util.print_hs import print_hist2, print_hist3

check = True
check2 = True


class SparsityFunction2(Function):
    @staticmethod
    def forward(ctx, input, changeDim=False):
        start = time.time()
        # N C H W
        global check, check2
        n, m = constant.args.sparsity.activation.n, constant.args.sparsity.activation.m
        if check:
            print("m:" + str(m))
            print("n:" + str(n))
            check = False
        if len(input.shape) == 4:
            if changeDim:
                x = input.permute(0, 3, 1, 2)
            else:
                x = input
            N, C, H, W = x.shape
            masked_x = x.reshape(N, C // m, m, H, W).abs_()
            mask = torch.ones_like(masked_x)
            val, _ = torch.topk(masked_x, n, dim=2)
            val = val[:, :, n - 1:, :, :]
            idx = masked_x < val
            mask[idx] = 0
            mask = mask.view(N, C, H, W)
            '''统计block分布'''
            # wei_sta = [0, 0]
            # wei_tmp = torch.sum(mask[:, :, 0:4, :, :], dim=2)
            # wei_sta[1] = torch.sum(wei_tmp == 1).cpu().item()
            # wei_sta[0] = N*C // m*H*W-wei_sta[1]
            # if str(constant.idx) in constant.valid_cal:
            #     constant.valid_cal[str(constant.idx)].update(wei_sta)
            # else:
            #     constant.valid_cal[str(constant.idx)] = WeightSta()
            #     constant.valid_cal[str(constant.idx)].update(wei_sta)
            ''''''
            # mask = mask.view(N, C, H, W)
            '''统计权值大小分布1'''
            # print_hist2(x[mask.type(torch.bool)], x[~mask.type(torch.bool)], constant.idx)
            # data1 = torch.tensor(x[mask.type(torch.bool)])
            if constant.args.sparsity.activation.n2:
                m2, n2 = constant.args.sparsity.activation.m2, constant.args.sparsity.activation.n2
                if check2:
                    print("m2:" + str(m2))
                    print("n2:" + str(n2))
                    check2 = False
                tmp_x = x.clone().abs_()
                tmp_x[mask.type(torch.bool)] = 0
                b = torch.reshape(tmp_x, (N, C // m2, m2, H, W))
                mask2 = torch.ones_like(b)
                a, _ = torch.topk(b, n2, dim=2)
                a = a[:, :, n2 - 1:, :, :]
                tmp = b < a
                mask2[tmp] = 0
                mask2 = mask2.view(N, C, H, W)
                '''统计权值大小分布2'''
                # data3 = torch.tensor(x[mask2.type(torch.bool)])
                ''''''
                mask += mask2
                '''统计权值大小分布3'''
                # data2 = torch.tensor(x[~mask.type(torch.bool)])
                # print_hist3(data1, data2, data3, constant.idx)
                ''''''
            if changeDim:
                mask = mask.permute(0, 2, 3, 1)
            # tmp = input * mask
            # tmp = torch.reshape(tmp, (N, C // m, m, W, H))
            # sta = torch.zeros(5)
            # for i1 in range(N):
            #     for i2 in range(C // m):
            #         for i3 in range(W):
            #             for i4 in range(H):
            #                 k = tmp[i1, i2, :, i3, i4]
            #                 sta[torch.sum(torch.abs(k) > 1e-8)] += 1
            # if str(constant.idx) in constant.valid_cal:
            #     constant.valid_cal[str(constant.idx)].update(sta)
            # else:
            #     constant.valid_cal[str(constant.idx)] = Statistics()
            #     constant.valid_cal[str(constant.idx)].update(sta)
            # print(constant.valid_cal[str(constant.idx)].num)
        else:
            N, D = input.shape
            if D % m != 0:
                mask = torch.ones(input.shape).cuda()
            else:
                b = torch.reshape(torch.abs(input), (N, D // m, m))
                mask = torch.ones_like(b)
                a, _ = torch.topk(b, n, dim=2)
                a = a[:, :, n - 1:]
                tmp = b < a
                mask[tmp] = 0
                '''统计block分布'''
                # wei_sta = [0, 0]
                # wei_tmp = torch.sum(mask[:, :, 0:4], dim=2)
                # wei_sta[1] = torch.sum(wei_tmp == 1).cpu().item()
                # wei_sta[0] = N * D // m - wei_sta[1]
                # if str(constant.idx) in constant.valid_cal:
                #     constant.valid_cal[str(constant.idx)].update(wei_sta)
                # else:
                #     constant.valid_cal[str(constant.idx)] = WeightSta()
                #     constant.valid_cal[str(constant.idx)].update(wei_sta)
                ''''''
                mask = mask.view(N, D)
                '''统计权值大小分布1'''
                # print_hist2(input[mask.type(torch.bool)], input[~mask.type(torch.bool)], constant.idx)
                # data1 = torch.tensor(input[mask.type(torch.bool)])
                if constant.args.sparsity.activation.n2:
                    m2, n2 = constant.args.sparsity.activation.m2, constant.args.sparsity.activation.n2
                    tmp_x = input.clone().abs_()
                    tmp_x[mask.type(torch.bool)] = 0
                    b = torch.reshape(tmp_x, (N, D // m2, m2))
                    mask2 = torch.ones_like(b)
                    a, _ = torch.topk(b, n2, dim=2)
                    a = a[:, :, n2 - 1:]
                    tmp = b < a
                    mask2[tmp] = 0
                    mask2 = torch.reshape(mask2, (N, D))
                    '''统计权值大小分布2'''
                    # data3 = torch.tensor(input[mask2.type(torch.bool)])
                    ''''''
                    mask += mask2
                    '''统计权值大小分布3'''
                    # data2 = torch.tensor(input[~mask.type(torch.bool)])
                    # print_hist3(data1, data2, data3, constant.idx)
                    ''''''
                # tmp = input * mask
                # tmp = torch.reshape(tmp, (N, D // m, m))
                # sta = torch.zeros(5)
                # for i1 in range(N):
                #     for i2 in range(D // m):
                #                 k = tmp[i1, i2, :]
                #                 sta[torch.sum(torch.abs(k) > 1e-8)] += 1
                # if str(constant.idx) in constant.valid_cal:
                #     constant.valid_cal[str(constant.idx)].update(sta)
                # else:
                #     constant.valid_cal[str(constant.idx)] = Statistics()
                #     constant.valid_cal[str(constant.idx)].update(sta)
        mask = mask.bool()
        ctx.save_for_backward(mask)
        print("total"+str(time.time() - start))
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        return grad_output * mask, None


class SparsityFunction(Function):
    @staticmethod
    def forward(ctx, input, changeDim=False):
        # N C H W
        n, m = constant.args.sparsity.activation.n, constant.args.sparsity.activation.m
        if len(input.shape) == 4:
            if changeDim:
                x = input.permute(0, 3, 1, 2)
            else:
                x = input
            N, C, H, W = x.shape
            mask = torch.ones_like(x)
            masked_x = (mask * x).reshape(N, C // m, m, H, W).abs_()
            mask = mask.view(N, C // m, m, H, W)
            _, idx = torch.topk(masked_x, m - n, dim=2, largest=False)
            mask.scatter_(2, idx, 0.0)
            mask = mask.view(N, C, H, W)
            if constant.args.sparsity.activation.n2:
                m2, n2 = constant.args.sparsity.activation.m2, constant.args.sparsity.activation.n2
                tmp_x = torch.tensor(x).abs_()
                tmp_x[mask.type(torch.bool)] = 0
                b = torch.reshape(tmp_x, (N, C // m2, m2, H, W))
                mask2 = torch.ones(b.shape)
                a, _ = torch.topk(torch.abs(b), n2, dim=2)
                a = a[:, :, n2 - 1:, :, :]
                tmp = torch.abs(b) < a
                mask2[tmp] = 0
                mask2 = torch.reshape(mask2, (N, C, H, W))
                mask += mask2
            if changeDim:
                mask = mask.permute(0, 2, 3, 1)
        else:
            N, D = input.shape
            if D % m != 0:
                mask = torch.ones(input.shape).cuda()
            else:
                N, D = input.shape
                mask = torch.ones_like(input)
                masked_x = (mask * input).reshape(N, D // m).abs_()
                mask = mask.view(N, D // m, m)
                _, idx = torch.topk(masked_x, m - n, dim=2, largest=False)
                mask.scatter_(2, idx, 0.0)
                mask = mask.view(N, D)
                if constant.args.sparsity.activation.n2:
                    m2, n2 = constant.args.sparsity.activation.m2, constant.args.sparsity.activation.n2
                    tmp_x = torch.tensor(input)
                    tmp_x[mask.type(torch.bool)] = torch.min(input) - 1.0
                    b = torch.reshape(tmp_x, (N, D // m2, m2))
                    mask2 = torch.ones(b.shape).cuda()
                    a, _ = torch.topk(b, n2, dim=2)
                    a = a[:, :, n2 - 1:]
                    tmp = b < a
                    mask2[tmp] = 0
                    mask2 = torch.reshape(mask2, (N, D))
                    '''统计权值大小分布2'''
                    # data3 = torch.tensor(input[mask2.type(torch.bool)])
                    ''''''
                    mask += mask2
                    '''统计权值大小分布3'''
                    # data2 = torch.tensor(input[~mask.type(torch.bool)])
                    # print_hist3(data1, data2, data3, constant.idx)
                    ''''''
                # tmp = input * mask
                # tmp = torch.reshape(tmp, (N, D // m, m))
                # sta = torch.zeros(5)
                # for i1 in range(N):
                #     for i2 in range(D // m):
                #                 k = tmp[i1, i2, :]
                #                 sta[torch.sum(torch.abs(k) > 1e-8)] += 1
                # if str(constant.idx) in constant.valid_cal:
                #     constant.valid_cal[str(constant.idx)].update(sta)
                # else:
                #     constant.valid_cal[str(constant.idx)] = Statistics()
                #     constant.valid_cal[str(constant.idx)].update(sta)
        mask = mask.bool()
        ctx.save_for_backward(mask)
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        return grad_output * mask, None


sparsityNor = SparsityFunction2.apply


class Sparsity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return sparsityNor(input)
