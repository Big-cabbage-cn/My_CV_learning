import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_test=False):
        # 1. 加载 CSV 文件
        self.df = pd.read_csv(csv_file)
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 2. 提取像素数据 (从第 1 列到最后，因为第 0 列是 Label)
        if not self.is_test:
            label = self.df.iloc[idx, 0] # 训练集的第一列是数字标签
            pixels = self.df.iloc[idx, 1:].values
        else:
            label = -1 # 测试集没有标签
            pixels = self.df.iloc[idx, :].values # 测试集所有列都是像素
        
        # 3. 核心步骤：将 784 维向量还原为 1x28x28 的图像张量
        # 并归一化到 0-1 之间
        image = pixels.reshape(1, 28, 28).astype(np.float32) / 255.0
        
        return torch.from_numpy(image), torch.tensor(label)