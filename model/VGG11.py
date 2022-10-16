import torch
from torch import nn
from sparsity.sparsity import sparsityNor
from sparsity.sparsity_fix import sparsityFix
from util import constant


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False, mode='none', smooth=False):
        super(VGG, self).__init__()
        self.features = features  # 卷积层提取特征
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=True),
        )
        self.mode = mode
        self.check = True
        self.smooth = smooth
        if init_weights:
            self._initialize_weights()  # 初始化权重
        if self.mode == 'normal':
            self.sparsity = sparsityNor
        elif self.mode == 'fix':
            self.sparsity = sparsityFix
        else:
            self.sparsity = None

    # 修改forward
    def forward(self, x):
        idx = 0
        if self.mode != 'none':
            for i in self.features:
                x = i(x)
                if isinstance(i, nn.Conv2d) or isinstance(i, nn.MaxPool2d):
                    if not self.smooth or (self.smooth and constant.epoch > constant.epo_num[idx]):
                        x = self.sparsity(x)
                    idx += 1
            x = self.avgpool(x)
            if not self.smooth or (self.smooth and constant.epoch > constant.epo_num[idx]):
                x = self.sparsity(x)
            idx += 1
            x = torch.flatten(x, 1)
            for i in self.classifier:
                x = i(x)
                if isinstance(i, nn.Linear):
                    if not self.smooth or (self.smooth and constant.epoch > constant.epo_num[idx]):
                        x = self.sparsity(x)
                    idx += 1
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x


# 卷积层提取特征
def make_features(cfg: list):  # 传入的是具体某个模型的参数列表
    layers = []
    in_channels = 3  # 输入的原始图像(rgb三通道)
    for v in cfg:
        # 如果是最大池化层，就进行池化
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 不然就是卷积层
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=1, bias=True)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
            # if v == 64:

    # layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
    # layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
    # layers += [torch.nn.Flatten()]
    # print(layers)
    return nn.Sequential(*layers)  # 单星号(*)将参数以元组(tuple)的形式导入


def vgg(model_name="vgg11", mode='none', smooth=False, num_classes=10):  # 双星号(**)将参数以字典的形式导入
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), num_classes=num_classes, mode=mode, smooth=smooth)  # **kwargs是你传入的字典数据
    return model


cfgs = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # 模型D
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}
