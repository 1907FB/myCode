import torch
from torch import nn
from torchvision.models import vgg16, vgg19, resnet50, alexnet, mobilenet_v2
from model.ResNet import ResNet18, ResNet34
from model.VGG11 import vgg
from model.convnext import convnext_tiny
from util import constant
from util.myConv import _set_module


def get_model(model="vgg16", pretrained=True, mode='normal', smooth=False, num_classes=10):
    if model == "vgg11":
        return vgg(model_name=model, mode=mode, smooth=smooth, num_classes=num_classes)
    elif model == "vgg16":
        mod = vgg(model_name=model, mode=mode, smooth=smooth, num_classes=num_classes)
        if pretrained:
            tmp = vgg16(pretrained=pretrained).state_dict()
            model_dict = mod.state_dict()
            pretrained_dict = {k: v for k, v in tmp.items() if k != 'classifier.6.weight' and k != 'classifier.6.bias'}
            model_dict.update(pretrained_dict)
            mod.load_state_dict(model_dict)
        return mod
    elif model == "vgg19":
        return vgg19(pretrained=pretrained)
    elif model == "resnet18":
        ret = ResNet18(pretrained=pretrained, mode=mode)
        # if constant.args.dataloader.dataset == 'cifar10' or constant.args.dataloader.dataset == 'cifar100':
        ret.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        ret.maxpool = None
        ret.fc = torch.nn.Linear(512, num_classes)
        return ret
    elif model == "resnet34":
        ret = ResNet34(pretrained=pretrained, mode=mode)
        # if constant.args.dataloader.dataset == 'cifar10' or constant.args.dataloader.dataset == 'cifar100':
        ret.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        ret.maxpool = None
        ret.fc = torch.nn.Linear(512, num_classes)
        return ret
    elif model == "alexnet":
        return alexnet(pretrained=pretrained)
    elif model == 'mobilenetv2':
        return mobilenet_v2(pretrained=pretrained)
    elif model == 'convnet':
        ret = convnext_tiny(pretrained=pretrained)
        name = 'downsample_layers.0.0'
        tmp = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1)
        _set_module(ret, name, tmp)
        ret.head = nn.Linear(768, num_classes)
        return ret
