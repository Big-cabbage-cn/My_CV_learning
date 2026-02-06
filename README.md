# My_CV_learning
My_CV_learning
# 项目名称：基于 [模型名] 的医学影像分类
> 这里写一句话简介，例如：本项目利用 PyTorch 实现对肺部 X 光片的自动识别。

---

## 📸 效果展示
![模型识别结果例子](./images/result.jpg)
*图 1：模型对测试集图片的预测结果（红色为病灶区域）*

---

## 🚀 项目功能
* [x] 支持多种经典模型（ResNet, EfficientNet）
* [x] 自动生成训练过程的 Loss 和 Accuracy 曲线
* [x] 适配 Kaggle 环境，支持多 GPU 训练

---

## 📊 实验结果
| 模型 | 准确率 (Accuracy) | 训练时长 | 备注 |
| :--- | :--- | :--- | :--- |
| ResNet50 | 92.5% | 2h | 基础版本 |
| EfficientNet-B0 | 95.1% | 1.5h | **最终采用版本** |
