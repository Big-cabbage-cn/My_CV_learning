import torch
import torch.nn as nn
import cv2
import numpy as np

# 1. 定义模型（还是你刚才写的那个）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # 注意：这里我们假设输入是 100x100，所以池化后是 50x50
        self.fc = nn.Linear(16 * 50 * 50, 2) 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) # 动态展平
        x = self.fc(x)
        return x

# 2. 推理函数
def run_inference():
    # 读取你预处理好的灰度图
    img = cv2.imread('processed_data/proc_test.jpg', 0)
    if img is None:
        print("错误：没找到处理后的图片，请先运行 preprocess.py")
        return

    # 缩放到 100x100，匹配我们模型的输入要求
    img_resized = cv2.resize(img, (100, 100))
    
    # 转换为 Tensor: [1, 1, 100, 100]
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0)
    img_tensor /= 255.0  # 归一化到 0-1

    # 实例化模型并运行
    model = SimpleCNN()
    model.eval() # 开启评估模式
    
    with torch.no_grad(): # 推理时不需要计算梯度，省内存
        output = model(img_tensor)
    
    print(f"模型输出的原始数值 (Logits): {output}")
    
    # 使用 Softmax 将数值转化为概率
    probabilities = torch.nn.functional.softmax(output, dim=1)
    print(f"模型判断的概率 -> [健康: {probabilities[0][0]:.2f}, 病变: {probabilities[0][1]:.2f}]")

if __name__ == "__main__":
    run_inference()
model = SimpleCNN()
print(model)