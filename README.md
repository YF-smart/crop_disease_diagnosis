# 🌿 多作物叶片病害智能诊断系统

基于 **PyTorch + ResNet/EfficientNet 迁移学习** 的农作物叶片病害识别系统,支持 **5种作物 17类病害/健康** 的智能诊断。

## 📌 项目简介

本项目实现了从数据预处理、模型训练、评估到推理部署的完整深度学习图像分类 pipeline，适合作为机器视觉 / AI 应用岗位的面试作品。

### 核心能力展示

- ✅ **迁移学习**：基于 ImageNet 预训练权重微调 ResNet18/50、EfficientNet
- ✅ **数据处理**：处理嵌套目录结构、类别不均衡（WeightedRandomSampler）、数据增强
- ✅ **模型训练**：完整训练循环 + 学习率调度 + 早停防过拟合
- ✅ **评估分析**：混淆矩阵、Precision/Recall/F1、Top-K 准确率、易混淆类别分析
- ✅ **推理部署**：单张图片快速推理，输出 Top-3 诊断结果和置信度

## 📂 项目结构

```
crop_disease_diagnosis/
├── config/                  # 配置模块
│   └── config.py            # 所有超参数和路径集中管理
├── data/                    # 数据模块
│   └── dataset.py           # 自定义 Dataset + 数据增强 + 数据分析
├── models/                  # 模型模块
│   └── model.py             # 模型工厂 (ResNet18/34/50, EfficientNet)
├── utils/                   # 工具模块
│   ├── train_utils.py       # 训练/验证循环 + 早停
│   └── plot_utils.py        # 混淆矩阵 + 训练曲线 + 分类报告
├── train.py                 # ★ 训练主脚本
├── eval.py                  # ★ 模型评估脚本
├── inference.py             # ★ 单张图片推理脚本
├── demo/                    # 测试样例图片
├── weights/                 # 模型权重目录
│   └── README.md            # 权重获取说明
├── outputs/                 # 输出目录（训练结果、图表）
├── requirements.txt         # 依赖包列表
├── .gitignore               # Git 忽略规则
└── README.md                # 本文件
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

下载 PlantVillage 多作物病害数据集，按以下结构放置：

```
E:\project_data\PlantVillage\Crop Diseases Dataset\Crop Diseases\Crop___Disease\
├── Corn/
│   ├── Corn___Common_Rust/      (1192张)
│   ├── Corn___Gray_Leaf_Spot/   (513张)
│   ├── Corn___Healthy/          (1162张)
│   └── Corn___Northern_Leaf_Blight/ (985张)
├── Potato/
│   ├── Potato___Early_Blight/   (1000张)
│   ├── Potato___Healthy/        (152张)
│   └── Potato___Late_Blight/    (1000张)
├── Rice/
│   ├── Rice___Brown_Spot/       (613张)
│   ├── Rice___Healthy/          (1488张)
│   ├── Rice___Leaf_Blast/       (977张)
│   └── Rice___Neck_Blast/       (1000张)
├── Wheat/
│   ├── Wheat___Brown_Rust/      (902张)
│   ├── Wheat___Healthy/         (1116张)
│   └── Wheat___Yellow_Rust/     (924张)
└── sugarcane/
    ├── Bacterial Blight/        (100张)
    ├── Healthy/                 (100张)
    └── Red Rot/                 (100张)
```

> 数据集路径在 `config/config.py` 中的 `DATASET_ROOT` 修改

### 3. 训练模型

```bash
python train.py
```

训练过程会：
- 自动分析数据分布并保存图表
- 每轮输出 train/val 的 loss 和 accuracy
- 验证 loss 持续不降时自动早停
- 保存最佳模型到 `weights/best_model.pth`

### 4. 评估模型

```bash
python eval.py
```

输出：
- Top-1 / Top-3 准确率
- 混淆矩阵图
- 各类别 Precision / Recall / F1
- 最易混淆的类别对 Top-5

### 5. 单张图片推理

```bash
# 命令模式
python inference.py path/to/leaf.jpg

# 交互模式（直接回车）
python inference.py

# 保存预测结果图
python inference.py path/to/leaf.jpg --save
```

## 📊 数据集说明

| 作物 | 类别数 | 图片数 | 主要特点 |
|------|:--:|--------|------|
| 玉米 Corn | 4 | 3,852 | 灰叶斑病仅 513 张，存在不均衡 |
| 马铃薯 Potato | 3 | 2,152 | 健康类仅 152 张，严重不均衡 |
| 水稻 Rice | 4 | 4,078 | 多数据源合并，光照背景不一致 |
| 小麦 Wheat | 3 | 2,942 | 类别较均衡 |
| 甘蔗 Sugarcane | 3 | 300 | 每类仅 100 张，小样本挑战 |
| **合计** | **17** | **13,324** | |

## 🔧 技术栈

| 技术 | 用途 |
|------|------|
| PyTorch / torchvision | 模型构建、训练、推理 |
| ResNet18/50 | backbone 网络（ImageNet 预训练） |
| OpenCV / PIL | 图像读取与预处理 |
| scikit-learn | 混淆矩阵、分类报告 |
| Matplotlib / Seaborn | 训练曲线、数据可视化 |
| tqdm | 训练进度条 |

## 📈 面试讲解要点

### 我能讲清楚的事

1. **为什么用迁移学习而不是从零训练**
   - ImageNet 预训练模型已经学会了边缘、纹理等通用特征
   - 1.3 万张图片从零训练 ResNet 容易过拟合
   - 迁移学习用更少的数据、更短的时间达到更高精度

2. **怎么处理类别不均衡**
   - 用 `WeightedRandomSampler` 给少数类更高采样权重
   - 对甘蔗等小样本类别做针对性数据增强
   - 分析每类 Precision/Recall 判断模型是否偏向多数类

3. **为什么早停**
   - 验证 loss 不再下降说明模型开始背训练集（过拟合）
   - 早停在过拟合之前终止训练，保留泛化能力最好的模型

4. **混淆矩阵能看出什么**
   - 对角线越亮 = 分类越准
   - 非对角线亮斑 = 易混淆类别（如早疫病 vs 晚疫病）
   - 可以针对性收集更多易混淆类别的数据

## 📄 License

MIT License
