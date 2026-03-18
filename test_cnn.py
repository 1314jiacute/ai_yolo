

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的CNN模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积块：输入1个通道(灰度图), 输出6个通道, 卷积核5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 第一个池化层：2x2最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积块：输入6个通道, 输出16个通道, 卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第一个全连接层：16个通道 * 4x4特征图大小 -> 120个神经元
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 第二个全连接层：120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # 输出层：84 -> 10 (对应10个数字)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播：conv1 -> ReLU -> pool
        x = self.pool(F.relu(self.conv1(x)))
        # conv2 -> ReLU -> pool
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图，以便输入到全连接层
        x = torch.flatten(x, 1) # 保留批次维度
        # fc1 -> ReLU
        x = F.relu(self.fc1(x))
        # fc2 -> ReLU
        x = F.relu(self.fc2(x))
        # fc3 (输出层，通常不加激活函数，交给损失函数处理)
        x = self.fc3(x)
        return x

# 实例化模型
net = Net()
print(net)