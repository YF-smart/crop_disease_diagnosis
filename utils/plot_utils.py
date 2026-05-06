"""
可视化工具
- 混淆矩阵
- 训练曲线 (loss + accuracy)
- 每类指标柱状图
"""
import os
import numpy as np
import matplotlib
# 设置中文字体，解决图表中文乱码
_font = None
for _f in ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        _font = _f
        break
    except Exception:
        continue
if _font:
    matplotlib.rcParams['font.sans-serif'] = [_font, 'DejaVu Sans']
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path, normalize=True):
    """
    绘制混淆矩阵 —— 面试展示的利器

    混淆矩阵能看出：
    - 对角线越亮越好 (正确分类)
    - 非对角线亮斑 = 易混淆的类别对 (如早疫病 vs 晚疫病)

    Args:
        all_labels: 真实标签列表
        all_preds: 预测标签列表
        class_names: 类别中文名列表
        save_path: 保存路径
        normalize: True=按行归一化(百分比), False=绝对数量
    """
    cm = confusion_matrix(all_labels, all_preds)

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # 处理除零

    fig, ax = plt.subplots(figsize=(16, 13))

    # 用热力图展示
    sns.heatmap(
        cm, annot=True, fmt='.2f' if normalize else 'd',
        cmap='Blues', xticklabels=class_names,
        yticklabels=class_names, ax=ax,
        annot_kws={'size': 8}, vmin=0, vmax=1
    )

    ax.set_xlabel('预测标签', fontsize=13)
    ax.set_ylabel('真实标签', fontsize=13)
    ax.set_title('混淆矩阵 (归一化)' if normalize else '混淆矩阵', fontsize=15)

    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" 📊 混淆矩阵已保存: {save_path}")
    plt.close()


def plot_training_curves(history, save_path):
    """
    绘制训练曲线 —— 展示模型学习过程

    history: dict, 包含 'train_loss', 'val_loss', 'train_acc', 'val_acc' 列表

    面试可讲：
    - train_loss 持续下降但 val_loss 上升 → 过拟合
    - 两条线都平稳 → 模型收敛
    - val_acc 抖动 → batch size 太小或学习率太大
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Loss 曲线
    ax1.plot(epochs, history['train_loss'], 'b-o', label='训练损失', markersize=3)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='验证损失', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练 & 验证 Loss 曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 标注最佳 epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    ax1.annotate(f'最佳: Epoch {best_epoch}\nLoss={best_val:.4f}',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 2, best_val + 0.1),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')

    # 右图：Accuracy 曲线
    ax2.plot(epochs, history['train_acc'], 'b-o', label='训练准确率', markersize=3)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='验证准确率', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练 & 验证准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 标注最佳准确率
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    ax2.annotate(f'最佳: Epoch {best_acc_epoch}\nAcc={best_acc:.2f}%',
                xy=(best_acc_epoch, best_acc),
                xytext=(best_acc_epoch + 2, best_acc - 5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" 📈 训练曲线已保存: {save_path}")
    plt.close()


def plot_class_report(all_labels, all_preds, class_names, save_path):
    """
    绘制每个类别的精确率、召回率、F1 分数柱状图

    面试可讲：
    对于类别不均衡的数据集，只看准确率不够，
    我分析了每个类的 Precision/Recall/F1，发现小样本类别(e.g. 甘蔗)指标偏低，
    原因是训练数据太少，后续可以通过数据增强或 few-shot 方法改善。
    """
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width, precision, width, label='精确率 Precision', color='#4CAF50')
    ax.bar(x, recall, width, label='召回率 Recall', color='#2196F3')
    ax.bar(x + width, f1, width, label='F1 分数', color='#FF9800')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('各类别 Precision / Recall / F1 对比', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" 📊 分类报告图已保存: {save_path}")
    plt.close()

    # 同时打印文本报告
    print("\n" + "=" * 70)
    print(" 📋 详细分类报告 (sklearn classification_report)")
    print("=" * 70)
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4, zero_division=0
    ))
