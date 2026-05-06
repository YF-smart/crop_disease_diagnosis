"""
模型模块 —— 支持多种 backbone，迁移学习

核心思路（面试重点）：
1. 使用 torchvision 提供的预训练模型 (在 ImageNet 上训练过)
2. 把最后的全连接层 (fc) 替换成 17 类输出
3. 可以冻结 backbone 只训练分类头 (快速收敛)
   也可以全部微调 (精度更高但更慢)

什么是迁移学习？
  预训练模型已经在 ImageNet 的 100 万张图片上学过边缘、纹理、形状等通用特征。
  我们只需要把它适配到叶片病害这个特定任务上，不用从零开始训练。
  类比：一个会开车的人学骑摩托车，比完全不会的人快得多。
"""
import torch
import torch.nn as nn
import torchvision.models as vision_models


def get_model(model_name='resnet18', num_classes=17, pretrained=True, freeze_backbone=False):
    """
    模型工厂函数 —— 根据配置创建对应的模型

    Args:
        model_name: 模型名称 (resnet18 / resnet34 / resnet50 / efficientnet_b0)
        num_classes: 输出类别数 (默认 17)
        pretrained: 是否使用 ImageNet 预训练权重
        freeze_backbone: 是否冻结 backbone (只训练最后的分类层)

    Returns:
        model: PyTorch 模型

    面试可讲：
    - 为什么选 ResNet18: 轻量、训练快、适合入门，在叶片分类上准确率已足够高
    - 为什么用预训练权重: 数据量有限(1.3万张)从零训练容易过拟合，迁移学习用少量数据就能收敛
    - freeze_backbone 的使用场景: 数据很少时先冻结训练分类头，解冻后再微调整个网络
    """
    if model_name == 'resnet18':
        model = vision_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        # ResNet 的全连接层叫 fc，输入 512 维特征，输出 num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'resnet34':
        model = vision_models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'resnet50':
        model = vision_models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        model = vision_models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        # EfficientNet 的分类头叫 classifier，是一个 Sequential
        # 最后一个 Linear 层的输入通道数
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"不支持的模型: {model_name}\n"
            f"可选: resnet18, resnet34, resnet50, efficientnet_b0"
        )

    # ====== 可选：冻结 backbone ======
    if freeze_backbone:
        # 除最后一层 fc/classifier 外，其余参数不更新
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False

        # 打印可训练参数数量
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f" 🔒 已冻结 backbone，可训练参数: {trainable:,} / {total:,} "
              f"({trainable/total*100:.1f}%)")

    return model


def count_parameters(model):
    """统计模型参数量 —— 面试时经常被问到"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # 测试: 打印各模型的参数数量
    for name in ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0']:
        m = get_model(name, num_classes=17, pretrained=True)
        total, trainable = count_parameters(m)
        print(f"{name:<18s}  总参数: {total/1e6:.1f}M  "
              f"可训练: {trainable/1e6:.1f}M")
