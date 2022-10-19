import torch
from torch import nn

from model.convnext import LayerNorm
from model.get_model import get_model
from myLayer.sparse_conv import SparseConv2d
from myLayer.sparse_linear import SparseLinear
from util import constant
from util.myStatictis import WeightSta
from util.print_hs import print_hist2, print_hist3
import ipdb


def weight_sparsity_per_layer_old(module, name, m, n):
    if not hasattr(module, 'weight') or isinstance(module, nn.BatchNorm2d) or isinstance(module,
                                                                                         LayerNorm) or isinstance(
        module, nn.LayerNorm):
        return
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            return
        # COUT, CIN, H, W
        COUT, CIN, H, W = module.weight.shape
        weight = module.weight
        b = torch.reshape(weight, (m, COUT // m, CIN, H, W))
        mask = torch.ones(b.shape).cuda()
        a, _ = torch.topk(torch.abs(b), n, dim=0)
        a = a[n - 1:, :, :, :, :]
        tmp = torch.abs(b) < a
        mask[tmp] = 0
        '''统计block分布'''
        # wei_sta = [0, 0]
        # wei_tmp = torch.sum(mask[:4, :, :, :, :], dim=0)
        # wei_sta[1] = torch.sum(wei_tmp == 1).cpu().item()
        # wei_sta[0] = COUT // m * CIN * H * W-wei_sta[1]
        # if name in constant.valid_cal:
        #     constant.valid_cal[name].update(wei_sta)
        # else:
        #     constant.valid_cal[name] = WeightSta()
        #     constant.valid_cal[name].update(wei_sta)
        ''''''
        mask = torch.reshape(mask, (COUT, CIN, H, W))
        '''统计权值大小分布'''
        # print_hist2(weight.data[mask.type(torch.bool)], weight.data[~mask.type(torch.bool)], name)
        # data1 = torch.tensor(weight.data[mask.type(torch.bool)])
        '''如果有，进行二轮稀疏'''
        if constant.args.sparsity.weight.n2:
            m2, n2 = constant.args.sparsity.weight.m2, constant.args.sparsity.weight.n2
            tmp_w = torch.tensor(module.weight.data)
            tmp_w[mask.type(torch.bool)] = 0
            weight = tmp_w
            b = torch.reshape(weight, (m2, COUT // m2, CIN, H, W))
            mask2 = torch.ones(b.shape).cuda()
            a, _ = torch.topk(torch.abs(b), n2, dim=0)
            a = a[n2 - 1:, :, :, :, :]
            tmp = torch.abs(b) < a
            mask2[tmp] = 0
            mask2 = torch.reshape(mask2, (COUT, CIN, H, W))
            # data3 = torch.tensor(weight.data[mask2.type(torch.bool)])
            mask += mask2
            # data2 = torch.tensor(weight.data[~mask.type(torch.bool)])
            # print_hist3(data1, data2, data3, name)
        mask = mask.type(torch.bool)
        # module.sparsity_mask = mask
        module.sparsity_mask = nn.Parameter(mask, requires_grad=False)
        # module.weight.data = module.weight.data * mask
        # print(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        COUT, = module.weight.shape
        weight = module.weight
        b = torch.reshape(weight, (m, COUT // m))
        mask = torch.ones(torch.abs(b).shape).cuda()
        a, _ = torch.topk(b, n, dim=0)
        a = a[n - 1:, :]
        tmp = torch.abs(b) < a
        mask[tmp] = 0
        mask = torch.reshape(mask, (COUT,))
        module.sparsity_mask = mask
        module.weight.data = module.weight.data * mask
    elif isinstance(module, nn.Linear):
        # COUT, CIN
        COUT, CIN = module.weight.shape
        weight = torch.tensor(module.weight.data[:COUT // m * m, :])
        COUT2 = COUT // m * m
        b = torch.reshape(weight, (m, COUT2 // m, CIN))
        mask = torch.ones(b.shape).cuda()
        a, _ = torch.topk(torch.abs(b), n, dim=0)
        a = a[n - 1:, :, :]
        tmp = torch.abs(b) < a
        mask[tmp] = 0
        mask = torch.reshape(mask, (COUT2, CIN))
        if constant.args.sparsity.weight.n2:
            m2, n2 = constant.args.sparsity.weight.m2, constant.args.sparsity.weight.n2
            tmp_w = torch.tensor(module.weight.data[:COUT // m * m, :])
            tmp_w[mask.type(torch.bool)] = 0
            weight = tmp_w
            b = torch.reshape(weight, (m, COUT2 // m, CIN))
            mask2 = torch.ones(b.shape).cuda()
            a, _ = torch.topk(torch.abs(b), n2, dim=0)
            a = a[n2 - 1:, :, :]
            tmp = torch.abs(b) < a
            mask2[tmp] = 0
            mask2 = torch.reshape(mask2, (COUT2, CIN))
            mask += mask2
        if COUT2 != COUT:
            mask = torch.cat((mask, torch.ones(COUT - COUT2, CIN).cuda()), 0)
        mask = mask.type(torch.bool)
        module.sparsity_mask = nn.Parameter(mask, requires_grad=False)
        # module.weight.data = module.weight.data * mask
        # module.sparsity_mask = mask.T
    else:
        print("error with :")
        print(module)
    # print(name)
    return


def weight_sparsity_per_layer(module, name, m, n):
    if isinstance(module, SparseConv2d):
        # COUT, CIN, H, W
        COUT, CIN, H, W = module.weight.shape
        weight = module.weight
        b = torch.reshape(weight, (m, COUT // m, CIN, H, W))
        mask = torch.ones(b.shape)
        a, _ = torch.topk(torch.abs(b), n, dim=0)
        a = a[n - 1:, :, :, :, :]
        tmp = torch.abs(b) < a
        mask[tmp] = 0
        '''统计block分布'''
        # wei_sta = [0, 0]
        # wei_tmp = torch.sum(mask[:4, :, :, :, :], dim=0)
        # wei_sta[1] = torch.sum(wei_tmp == 1).cpu().item()
        # wei_sta[0] = COUT // m * CIN * H * W-wei_sta[1]
        # if name in constant.valid_cal:
        #     constant.valid_cal[name].update(wei_sta)
        # else:
        #     constant.valid_cal[name] = WeightSta()
        #     constant.valid_cal[name].update(wei_sta)
        ''''''
        mask = torch.reshape(mask, (COUT, CIN, H, W))
        '''统计权值大小分布'''
        # print_hist2(weight.data[mask.type(torch.bool)], weight.data[~mask.type(torch.bool)], name)
        # data1 = torch.tensor(weight.data[mask.type(torch.bool)])
        '''如果有，进行二轮稀疏'''
        if constant.args.sparsity.weight.n2:
            m2, n2 = constant.args.sparsity.weight.m2, constant.args.sparsity.weight.n2
            tmp_w = torch.tensor(module.weight.data)
            tmp_w[mask.type(torch.bool)] = 0
            weight = tmp_w
            b = torch.reshape(weight, (m2, COUT // m2, CIN, H, W))
            mask2 = torch.ones(b.shape)
            a, _ = torch.topk(torch.abs(b), n2, dim=0)
            a = a[n2 - 1:, :, :, :, :]
            tmp = torch.abs(b) < a
            mask2[tmp] = 0
            mask2 = torch.reshape(mask2, (COUT, CIN, H, W))
            # data3 = torch.tensor(weight.data[mask2.type(torch.bool)])
            mask += mask2
            # data2 = torch.tensor(weight.data[~mask.type(torch.bool)])
            # print_hist3(data1, data2, data3, name)
        mask = mask.type(torch.bool)
        module.set_mask(mask)
        print(name)
        # module.sparsity_mask = mask
        # module.weight.data = module.weight.data * mask
        # print(module.weight.data)
    elif isinstance(module, SparseLinear):
        # COUT, CIN
        COUT, CIN = module.weight.shape
        weight = torch.tensor(module.weight.data[:COUT // m * m, :])
        COUT2 = COUT // m * m
        b = torch.reshape(weight, (m, COUT2 // m, CIN))
        mask = torch.ones(b.shape).cuda()
        a, _ = torch.topk(torch.abs(b), n, dim=0)
        a = a[n - 1:, :, :]
        tmp = torch.abs(b) < a
        mask[tmp] = 0
        mask = torch.reshape(mask, (COUT2, CIN))
        if constant.args.sparsity.weight.n2:
            m2, n2 = constant.args.sparsity.weight.m2, constant.args.sparsity.weight.n2
            tmp_w = torch.tensor(module.weight.data[:COUT // m * m, :])
            tmp_w[mask.type(torch.bool)] = 0
            weight = tmp_w
            b = torch.reshape(weight, (m, COUT2 // m, CIN))
            mask2 = torch.ones(b.shape).cuda()
            a, _ = torch.topk(torch.abs(b), n2, dim=0)
            a = a[n2 - 1:, :, :]
            tmp = torch.abs(b) < a
            mask2[tmp] = 0
            mask2 = torch.reshape(mask2, (COUT2, CIN))
            mask += mask2
        if COUT2 != COUT:
            mask = torch.cat((mask, torch.ones(COUT - COUT2, CIN).cuda()), 0)
        mask = mask.type(torch.bool)
        module.set_mask(mask)
        print(name)
    return


def backward_hook(module, gin, gout):
    if not hasattr(module, 'sparsity_mask'):
        return gin
    if isinstance(module, nn.Conv2d):
        # dx,dw
        gin0, gin1 = gin
        gin1 *= module.sparsity_mask
        new_gin = tuple([gin0, gin1])
    elif isinstance(module, nn.BatchNorm2d):
        # print(len(gin))
        # print(gin[0].shape)
        # print(gin[1].shape)
        # print(gin[2].shape)
        # print(module.sparsity_mask.shape)
        gin0, gin1, gin2 = gin
        gin1 *= module.sparsity_mask
        new_gin = tuple([gin0, gin1, gin2])
    elif isinstance(module, nn.Linear):
        # db,dx,dw
        # print(gin[0].shape)
        # print(gin[1].shape)
        # print(gin[2].shape)
        ipdb.set_trace()
        print(module)
        print(gin[0].shape)
        print(gin[1].shape)
        return gin
        gin0, gin1, gin2 = gin
        # print(module)
        # print(gin0.shape)
        # print(gin1.shape)
        # print(gin2.shape)
        print(module.sparsity_mask.shape)
        gin2 *= module.sparsity_mask
        new_gin = tuple([gin0, gin1, gin2])
    else:
        print("error with :")
        print(module)
    return new_gin


# layer_num = 1,2,3,4
def weight_sparsity(model, layer_num, m, n):
    print("INFO - Begin Weight Sparsity")
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            # for i in range(layer_num):
            #     if name.startswith('layer' + str(4 - i)):
            weight_sparsity_per_layer(module, name, m, n)
    print("INFO - End Weight Sparsity")
    return model


def register_hook(model):
    for name, module in model.named_modules():
        if hasattr(module, 'sparsity_mask'):
            module.register_backward_hook(backward_hook)
