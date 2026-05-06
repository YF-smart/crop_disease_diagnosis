"""
模型评估脚本 —— 在验证集上全面评估模型性能

用法: python eval.py

输出:
- Top-1 / Top-3 准确率
- 混淆矩阵
- 每类 Precision / Recall / F1
- 识别错误样本分析
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from models import get_model
from data.dataset import PlantVillageDataset, get_val_transforms
from utils import plot_confusion_matrix, plot_class_report
from utils.train_utils import validate_one_epoch
import torch.nn as nn
from torch.utils.data import DataLoader


def main():
    cfg = Config()

    # 设备
    device = cfg.DEVICE
    print(f"\n 使用设备: {device}")

    # ========== 加载模型 ==========
    model_path = os.path.join(cfg.WEIGHT_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f" ❌ 未找到模型权重: {model_path}")
        print(f" 请先运行 train.py 训练模型")
        return

    print(f" 加载模型: {model_path}")

    # 创建模型结构
    model = get_model(
        model_name=cfg.MODEL_NAME,
        num_classes=cfg.NUM_CLASSES,
        pretrained=False  # 不需要预训练权重，我们会加载已训练的
    )

    # 加载权重
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ========== 加载验证数据 ==========
    full_dataset = PlantVillageDataset(root_dir=cfg.DATASET_ROOT, transform=None)
    class_names = [cfg.CLASS_CN.get(cls, cls) for cls in full_dataset.classes]

    # 划分验证集 (和训练时一样 80:20，seed=42)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 构造验证集 dataloader
    from data.dataset import SubsetWithTransform
    val_dataset = SubsetWithTransform(
        full_dataset, val_indices.indices, transform=get_val_transforms(cfg)
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS)

    print(f" 验证集: {len(val_dataset):,} 张图片")

    # ========== 评估 ==========
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        val_loss, val_acc, all_preds, all_labels = validate_one_epoch(
            model, val_loader, criterion, device, epoch=0
        )

    print(f"\n {'='*50}")
    print(f" 📊 评估结果")
    print(f" {'='*50}")
    print(f" 验证 Loss: {val_loss:.4f}")
    print(f" Top-1 准确率: {val_acc:.2f}%")

    # 计算 Top-3 准确率 (预测的前3名中有一个正确就算对)
    top3_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, top3_idx = outputs.topk(3, dim=1)
            top3_correct += top3_idx.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            total += images.size(0)
    print(f" Top-3 准确率: {top3_correct/total*100:.2f}%")

    # ========== 混淆矩阵 & 分类报告 ==========
    cm_path = os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix_eval.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)

    report_path = os.path.join(cfg.OUTPUT_DIR, 'class_metrics_eval.png')
    plot_class_report(all_labels, all_preds, class_names, report_path)

    # ========== 找出最容易混淆的类别对 ==========
    from sklearn.metrics import confusion_matrix as cm_func
    cm = cm_func(all_labels, all_preds)
    # 将对角线置0，找非对角线最大值的索引
    np.fill_diagonal(cm, 0)
    flat_indices = np.argsort(cm.flatten())[::-1][:5]

    print(f"\n 🔍 最易混淆的 Top-5 类别对:")
    for idx in flat_indices:
        i, j = idx // cm.shape[0], idx % cm.shape[0]
        if cm[i, j] > 0:
            print(f"   {class_names[i]} → 被误判为 → {class_names[j]}  ({cm[i,j]} 次)")

    print(f"\n ✅ 评估完成，结果保存在: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
