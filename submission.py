import torch
import pandas as pd
import numpy as np
from models.simple_cnn import SimpleCNN  # 这里的路径要对应你重构后的 models 文件夹

def generate_submission():
    # 1. 硬件加速
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. 加载大脑
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval() # 开启预测模式

    # 3. 读取那张“考卷” (test.csv)
    test_path = 'data/test.csv' 
    test_df = pd.read_csv(test_path)
    
    results = []

    print(f"正在为 {len(test_df)} 张图片生成预测结果...")

    # 4. 批量预测
    with torch.no_grad():
        for i in range(len(test_df)):
            # 获取像素，test.csv 只有 784 列像素
            pixels = test_df.iloc[i].values
            # 还原形状 [Batch=1, Channel=1, H=28, W=28]
            img = torch.from_numpy(pixels).float().view(1, 1, 28, 28).to(device) / 255.0
            
            output = model(img)
            pred = torch.argmax(output, dim=1).item()
            
            # Kaggle 要求 ImageId 从 1 开始
            results.append([i + 1, pred])

    # 5. 保存答卷
    submission = pd.DataFrame(results, columns=['ImageId', 'Label'])
    submission.to_csv('final_submission.csv', index=False)
    print("✅ 成功！你的 final_submission.csv 已生成在根目录。")

if __name__ == "__main__":
    generate_submission()