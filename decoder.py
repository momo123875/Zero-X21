import torch
import torch.nn as nn
import math


# -------------------------------
# 使用简单的MLP实现的解码器
# -------------------------------
class MLP(nn.Module):
    # in_dim：特征+坐标 out_dim:灰度值 depth:深度 width:宽度
    def __init__(self, in_dim=128 + 3, out_dim=1, depth=4, width=256):
        super(MLP, self).__init__()
        stage_one = []
        stage_two = []
        for i in range(depth):
            if i == 0:
                stage_one.append(nn.Linear(in_dim, width))
                stage_two.append(nn.Linear(in_dim, width))
            elif i == depth - 1:
                stage_one.append(nn.Linear(width, in_dim))
                stage_two.append(nn.Linear(width, out_dim))
            else:
                stage_one.append(nn.Linear(width, width))
                stage_two.append(nn.Linear(width, width))
            stage_one.append(nn.ReLU())
            stage_two.append(nn.ReLU())
        self.stage_one = nn.Sequential(*stage_one)  # * 操作符用于解包列表中的元素。
        self.stage_two = nn.Sequential(*stage_two)

    def forward(self, x):
        # 结合两个阶段的输出的方法是一种残差连接
        h = self.stage_one(x)
        return self.stage_two(x + h)


# -------------------------------
# 使用SIREN实现的解码器
# -------------------------------


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()  # 初始化权重

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6 / self.in_features) / self.omega_0,
                    math.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))  # 使用正弦激活函数


class SIREN(nn.Module):
    def __init__(self, in_dim=128 + 3, out_dim=1, depth=4, width=256, omega_0=30):
        super(SIREN, self).__init__()
        layers = []
        for i in range(depth):
            is_first = True if i == 0 else False
            in_features = in_dim if i == 0 else width
            out_features = width if i < depth - 1 else out_dim
            layers.append(SineLayer(in_features, out_features, is_first, omega_0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
