import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCNN
from utils.dataset import MNISTDataset

def train_mnist():
    # 1. 硬件配置：有 GPU 跑 GPU (Mac M1/M2 会用 mps)，没有跑 CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 2. 加载数据
    train_dataset = MNISTDataset(csv_file='data/train.csv')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. 初始化模型、损失函数和优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 训练循环
    model.train()
    for epoch in range(3): # 跑 3 轮就足够看到很高准确率了
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/3], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # 5. 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("训练完成，模型已保存！")

if __name__ == "__main__":
    train_mnist()