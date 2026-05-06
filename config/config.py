"""
全局配置文件 —— 所有超参数和路径集中管理
面试可讲点：为什么这样设置每个参数
"""
import os
import torch

class Config:
    # ==================== 路径配置 ====================
    # 数据集根目录 (嵌套结构: 作物/病害类别/图片)
    DATASET_ROOT = r"E:\project_data\PlantVillage\Crop Diseases Dataset\Crop Diseases\Crop___Disease"

    # 项目输出目录
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    WEIGHT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")

    # ==================== 模型配置 ====================
    MODEL_NAME = "resnet18"      # 可选: resnet18, resnet34, resnet50, efficientnet_b0
    NUM_CLASSES = 17             # 5种作物共17个病害/健康类别
    PRETRAINED = True            # 使用 ImageNet 预训练权重 (迁移学习核心)

    # ==================== 训练配置 ====================
    IMAGE_SIZE = 224             # 输入尺寸 (ResNet/EfficientNet 标准输入)
    BATCH_SIZE = 32              # 批大小 (根据显存调整，CPU可减到16)
    EPOCHS = 30                  # 训练轮数
    LR = 0.001                   # 初始学习率
    WEIGHT_DECAY = 1e-4          # L2 正则化系数 (防止过拟合)
    NUM_WORKERS = 2              # 数据加载线程数 (Windows建议0或2)

    # 学习率调度
    LR_STEP_SIZE = 7             # 每7个epoch学习率衰减一次
    LR_GAMMA = 0.1               # 衰减为原来的0.1倍

    # 早停策略
    EARLY_STOP_PATIENCE = 10     # 验证集loss连续10轮不降则停止
    EARLY_STOP_MIN_DELTA = 0.001 # 最小改善阈值

    # ==================== 数据增强配置 ====================
    # 训练集增强 (模拟真实拍摄场景: 不同角度/光照)
    TRAIN_AUGMENT = {
        "random_horizontal_flip": 0.5,
        "random_vertical_flip": 0.1,    # 叶片方向不确定
        "random_rotation": 30,           # ±30度旋转
        "color_jitter": {                # 颜色抖动 (光照变化)
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.3,
            "hue": 0.1
        },
        "random_affine": {              # 仿射变换
            "degrees": 0,
            "translate": (0.1, 0.1),    # 平移
            "scale": (0.9, 1.1)         # 缩放
        }
    }

    # ==================== 设备配置 ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== 类别信息 ====================
    # 按作物分组的类别 (方便后续分析和可视化)
    CROP_CLASSES = {
        "玉米 Corn": ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot",
                       "Corn___Healthy", "Corn___Northern_Leaf_Blight"],
        "马铃薯 Potato": ["Potato___Early_Blight", "Potato___Healthy",
                          "Potato___Late_Blight"],
        "水稻 Rice": ["Rice___Brown_Spot", "Rice___Healthy",
                      "Rice___Leaf_Blast", "Rice___Neck_Blast"],
        "小麦 Wheat": ["Wheat___Brown_Rust", "Wheat___Healthy",
                       "Wheat___Yellow_Rust"],
        "甘蔗 Sugarcane": ["Bacterial Blight", "Healthy", "Red Rot"],
    }

    # 病害中文对照 (用于结果展示)
    CLASS_CN = {
        "Corn___Common_Rust": "玉米普通锈病",
        "Corn___Gray_Leaf_Spot": "玉米灰叶斑病",
        "Corn___Healthy": "玉米健康",
        "Corn___Northern_Leaf_Blight": "玉米北方叶枯病",
        "Potato___Early_Blight": "马铃薯早疫病",
        "Potato___Healthy": "马铃薯健康",
        "Potato___Late_Blight": "马铃薯晚疫病",
        "Rice___Brown_Spot": "水稻褐斑病",
        "Rice___Healthy": "水稻健康",
        "Rice___Leaf_Blast": "水稻稻瘟病",
        "Rice___Neck_Blast": "水稻颈瘟病",
        "Wheat___Brown_Rust": "小麦褐锈病",
        "Wheat___Healthy": "小麦健康",
        "Wheat___Yellow_Rust": "小麦黄锈病",
        "Bacterial Blight": "甘蔗细菌性疫病",
        "Healthy": "甘蔗健康",
        "Red Rot": "甘蔗红腐病",
    }


def get_config():
    """获取配置单例"""
    return Config()
