# My_CV_learning: MNIST 手写数字识别项目

本项目是基于 PyTorch 实现的经典 MNIST 手写数字识别任务。通过重构，项目已实现模块化开发，支持 Apple M 系列芯片的硬件加速（MPS）。

## 📂 项目结构
- `models/`: 存放 CNN 神经网络架构 (SimpleCNN)。
- `utils/`: 存放数据处理逻辑 (MNISTDataset)。
- `scripts/`: 存放数据可视化验证脚本。
- `main.py`: 模型训练入口。
- `submission.py`: 生成 Kaggle 提交文件。
- `inference.py`: 随机抽取测试集进行推理验证。

## 🚀 技术亮点
- **模块化设计**：将数据、模型、脚本解耦，具备高度可扩展性。
- **MPS 加速**：针对 MacBook Air 优化的训练流程。
- **Kaggle 实战**：完整跑通了从原始 CSV 到最终 Submission 的全流程。

## 🛠️ 如何运行
1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
2. **下载数据**：   
   从 Kaggle 下载 MNIST 数据集，放置在 `data/` 目录下。
3. **训练模型**
   ```bash
   python main.py
4. **查看结果**：
   ```bash
   python inference.py

## 实验结果
-  经过 3 个 Epoch 训练，Loss 降低至 0.004 左右。
-  测试集预测准确率表现优秀。