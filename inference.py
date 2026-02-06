import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.simple_cnn import SimpleCNN
from utils.dataset import MNISTDataset

def verify_results():
    # 1. 加载模型架构和权重
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    # 2. 读取测试集 (注意：测试集没有 Label 列，只有 784 列像素)
    test_df = pd.read_csv('data/test.csv') 
    
    plt.figure(figsize=(12, 4))
    for i in range(4):
        # 随机挑一张
        idx = np.random.randint(0, len(test_df))
        pixels = test_df.iloc[idx].values
        
        # 预处理：还原形状并归一化
        input_tensor = torch.from_numpy(pixels).float().reshape(1, 1, 28, 28).to(device) / 255.0
        
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        # 可视化
        plt.subplot(1, 4, i+1)
        plt.imshow(pixels.reshape(28, 28), cmap='gray')
        plt.title(f"Predict: {prediction}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_results()