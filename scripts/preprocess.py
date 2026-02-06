import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def check_csv_data():
    csv_path = 'data/train.csv'
    if not os.path.exists(csv_path):
        print("错误：请确保 data/train.csv 文件存在")
        return

    # 1. 加载数据
    df = pd.read_csv(csv_path)
    
    # 2. 随便看前 5 张图
    plt.figure(figsize=(10, 5))
    for i in range(5):
        label = df.iloc[i, 0]
        # 还原为 28x28
        pixels = df.iloc[i, 1:].values.reshape(28, 28)
        
        plt.subplot(1, 5, i+1)
        plt.imshow(pixels, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_csv_data()