import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：输入1通道（灰度），输出16个特征，核大小3x3，填充1保持大小
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 池化层：2x2窗口，会让 28x28 变成 14x14
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层 (Classifier)
        # 计算过程：16个通道 * 14(高) * 14(宽) = 3136
        self.fc1 = nn.Linear(16 * 14 * 14, 128) # 先降维到 128
        self.fc2 = nn.Linear(128, 10)           # 最后输出 10 类 (数字 0-9)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        
        # 展平数据：从 [batch, 16, 14, 14] 变成 [batch, 3136]
        x = x.view(-1, 16 * 14 * 14)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x