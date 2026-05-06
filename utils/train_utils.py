"""
训练工具函数
- EarlyStopping: 早停策略，防止过拟合
- train_one_epoch: 训练一个 epoch
- validate_one_epoch: 验证一个 epoch
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class EarlyStopping:
    """
    早停策略 —— 验证集 loss 不再下降时自动停止训练

    为什么需要早停（面试重点）：
    训练太久会导致过拟合 —— 模型"背"住了训练数据，但泛化能力变差。
    早停在验证集性能不再提升时自动停止，避免浪费时间和过拟合。

    使用方式:
        early_stop = EarlyStopping(patience=10)
        for epoch in range(epochs):
            train(...)
            val_loss = validate(...)
            if early_stop(val_loss, model):
                print("早停触发，训练结束")
                break
    """

    def __init__(self, patience=10, min_delta=0.001, save_path='best_model.pth'):
        """
        Args:
            patience: 容忍多少个 epoch 不改善 (超过就停)
            min_delta: 最小改善阈值，小于这个值不算改善
            save_path: 最佳模型保存路径
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0          # 连续未改善的 epoch 数
        self.best_loss = float('inf')  # 当前最佳 loss
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        每轮验证后调用，检查是否需要早停

        Returns:
            True 表示应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            # 有改善 → 保存模型，重置计数器
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        else:
            # 未改善 → 计数器+1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    """
    训练一个 epoch

    Args:
        model: 模型
        loader: 训练 DataLoader
        criterion: 损失函数 (CrossEntropyLoss)
        optimizer: 优化器 (Adam)
        device: 'cuda' 或 'cpu'
        epoch: 当前轮数
        scaler: 混合精度训练的 GradScaler (可选)

    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率

    面试可讲: 我了解混合精度训练 (AMP) 可以加速训练 + 节省显存，
    但本项目数据量和模型较小，用不到。
    """
    model.train()  # 切换到训练模式 (启用 Dropout/BatchNorm 的训练行为)
    total_loss = 0.0
    correct = 0
    total = 0

    # tqdm 显示进度条
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', ncols=100)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播三连
        optimizer.zero_grad()   # 1. 清空上轮梯度
        loss.backward()         # 2. 反向传播计算梯度
        optimizer.step()        # 3. 更新参数

        # 统计
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)  # 取输出概率最大的类别
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        # 更新进度条显示
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.1f}%'
        })

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc


@torch.no_grad()  # 验证时不需要计算梯度，节省显存和时间
def validate_one_epoch(model, loader, criterion, device, epoch):
    """
    验证一个 epoch

    @torch.no_grad(): 禁用梯度计算，验证时不需要反向传播
    model.eval(): 切换到评估模式 (关闭 Dropout，BN 用全局统计)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # 收集所有预测和标签，用于后续计算混淆矩阵
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]', ncols=100)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.1f}%'
        })

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc, all_preds, all_labels
