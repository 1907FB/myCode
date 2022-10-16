import torch
from torch import nn
from sparsity.sparsity import Sparsity


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False, mode='train'):
        super(VGG, self).__init__()
        self.features = features  # 卷积层提取特征
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=False),
        )
        self.mode = mode
        self.check = True
        if init_weights:
            self._initialize_weights()  # 初始化权重

    # 修改forward
    def forward(self, x):
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
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=1, bias=False)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
            if v == 64:
                layers += [Sparsity()]

    # layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
    # layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
    # layers += [torch.nn.Flatten()]
    # print(layers)
    return nn.Sequential(*layers)  # 单星号(*)将参数以元组(tuple)的形式导入


def vgg(model_name="vgg16", **kwargs):  # 双星号(**)将参数以字典的形式导入
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)  # **kwargs是你传入的字典数据
    return model


cfgs = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # 模型D
}
