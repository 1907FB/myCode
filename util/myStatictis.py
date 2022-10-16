import torch


class Statistics:
    def __init__(self):
        # 有效计算次数，取值只能是0~4，用下标表示
        self.num = [0, 0, 0, 0, 0]
        self.N = 0

    def update(self, x):
        self.num[0] += x[0]
        self.num[1] += x[1]
        self.num[2] += x[2]
        self.num[3] += x[3]
        self.num[4] += x[4]
        self.N += 1


class WeightSta:
    def __init__(self):
        # 有效计算次数，取值只能是0~4，用下标表示
        self.one = 0
        self.two = 0

    def update(self, x):
        self.one += x[0]
        self.two += x[1]
