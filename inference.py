"""
推理脚本 —— 输入一张叶片图片，输出病害诊断结果

用法:
    python inference.py <图片路径>
    python inference.py demo/corn_rust_sample.jpg
    python inference.py                     # 交互模式，输入路径

功能:
    1. 加载训练好的模型
    2. 对输入图片进行预处理
    3. 输出 Top-3 预测结果 (病害名 + 置信度)
    4. (可选) 显示预测结果图
"""
import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from models import get_model


def load_model(cfg, device):
    """
    加载训练好的模型

    面试可讲：推理时只需要加载模型权重和类别映射，
    不需要重新训练。整个推理流程 < 0.1 秒/张图。
    """
    model_paths = [
        os.path.join(cfg.WEIGHT_DIR, 'best_model.pth'),
        os.path.join(cfg.WEIGHT_DIR, 'final_model.pth'),
    ]

    checkpoint = None
    for path in model_paths:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            print(f" ✅ 加载模型权重: {path}")
            break

    if checkpoint is None:
        print(f" ❌ 未找到模型权重文件!")
        print(f" 请先运行 train.py 训练模型，或将权重文件放在:")
        for p in model_paths:
            print(f"   - {p}")
        sys.exit(1)

    # 判断加载的是完整 checkpoint 还是纯 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # train.py 保存的完整 checkpoint
        model_name = checkpoint.get('model_name', cfg.MODEL_NAME)
        num_classes = checkpoint.get('num_classes', cfg.NUM_CLASSES)
        class_names = checkpoint.get('classes', None)
        state_dict = checkpoint['model_state_dict']
    else:
        # 纯 state_dict (由早停保存)
        model_name = cfg.MODEL_NAME
        num_classes = cfg.NUM_CLASSES
        class_names = None
        state_dict = checkpoint

    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 构建类别映射
    if class_names is None:
        # 从数据集重新获取
        from data.dataset import PlantVillageDataset
        full_dataset = PlantVillageDataset(root_dir=cfg.DATASET_ROOT, transform=None)
        class_names = full_dataset.classes

    idx_to_class = {i: cls for i, cls in enumerate(class_names)}

    return model, idx_to_class, class_names


def preprocess_image(image_path, cfg):
    """
    对输入图片做和验证集相同的预处理

    面试可讲：推理时的预处理必须和训练时一致，
    否则数据分布不匹配，模型预测会失效。
    """
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度
    return input_tensor, image


def predict(model, input_tensor, device, idx_to_class, cfg, top_k=3):
    """
    对预处理后的图片进行推理

    Returns:
        results: [(类别名, 中文名, 置信度%), ...]  按置信度降序排列
    """
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        # Softmax 把原始得分转为概率 (0~1之间，总和=1)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item() * 100
        class_name = idx_to_class.get(idx, f'Unknown-{idx}')
        cn_name = cfg.CLASS_CN.get(class_name, class_name)
        results.append((class_name, cn_name, prob))

    return results


def main():
    parser = argparse.ArgumentParser(description='多作物叶片病害智能诊断')
    parser.add_argument('image', nargs='?', default=None,
                        help='输入图片路径 (不填则进入交互模式)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='显示 Top-K 个预测结果 (默认3)')
    parser.add_argument('--save', action='store_true',
                        help='保存标注了预测结果的图片')
    args = parser.parse_args()

    cfg = Config()
    device = cfg.DEVICE if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model, idx_to_class, class_names = load_model(cfg, device)
    print(f" 模型: {cfg.MODEL_NAME} | 类别数: {len(class_names)}")

    # 获取图片路径
    if args.image:
        image_paths = [args.image]
    else:
        # 交互模式
        print("\n" + "=" * 60)
        print(" 🌿 叶片病害诊断 (输入 'q' 退出)")
        print("=" * 60)
        while True:
            path = input("\n 📷 请输入图片路径: ").strip().strip('"')
            if path.lower() == 'q':
                break
            if not os.path.exists(path):
                print(f" ❌ 文件不存在: {path}")
                continue
            image_paths = [path]
            break
        if not image_paths:
            return

    # 对每张图片进行推理
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f" ❌ 文件不存在: {img_path}")
            continue

        print(f"\n {'='*60}")
        print(f" 📷 图片: {img_path}")
        print(f" {'='*60}")

        # 预处理
        input_tensor, original_image = preprocess_image(img_path, cfg)

        # 预测
        results = predict(model, input_tensor, device, idx_to_class, cfg, args.top_k)

        # 打印结果
        print(f"\n 诊断结果 (Top-{args.top_k}):")
        print(f" {'-'*45}")
        for rank, (cls_name, cn_name, prob) in enumerate(results, 1):
            bar = '█' * int(prob / 5) + '░' * (20 - int(prob / 5))
            print(f" {rank}. {cn_name:<20s}  {prob:5.1f}%  {bar}")

        # 最终判断
        best_cn = results[0][1]
        best_prob = results[0][2]
        if best_prob > 80:
            verdict = f"🟢 高置信度诊断: {best_cn}"
        elif best_prob > 50:
            verdict = f"🟡 中等置信度: {best_cn}，建议进一步确认"
        else:
            verdict = f"🔴 低置信度，建议人工鉴定"

        print(f"\n {verdict}")

        # 可选保存预测结果图
        if args.save:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(original_image)
            ax1.set_title('原始图片', fontsize=13)
            ax1.axis('off')

            # 右图: 预测结果条形图
            names = [r[1] for r in results[::-1]]
            probs = [r[2] for r in results[::-1]]
            colors = ['#4CAF50' if p > 80 else '#FF9800' if p > 50 else '#f44336'
                      for p in probs]
            ax2.barh(names, probs, color=colors)
            ax2.set_xlabel('置信度 (%)')
            ax2.set_xlim(0, 100)
            ax2.set_title('Top-3 预测结果', fontsize=13)

            for bar, prob in zip(ax2.patches, probs):
                ax2.text(bar.get_width() + 1,
                         bar.get_y() + bar.get_height()/2,
                         f'{prob:.1f}%', va='center', fontsize=10)

            save_path = os.path.join(
                cfg.OUTPUT_DIR,
                f"inference_{os.path.basename(img_path).split('.')[0]}.png"
            )
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f" 📊 结果图已保存: {save_path}")


if __name__ == "__main__":
    main()
