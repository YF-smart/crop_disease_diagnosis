# 模型权重说明

## 权重获取方式

### 方式一：自行训练（推荐）

```bash
python train.py
```

训练完成后在 `weights/` 目录下生成：
- `best_model.pth` — 验证集 loss 最低的模型（约 45MB）
- `final_model.pth` — 最终模型（包含完整 checkpoint，约 50MB）

### 方式二：从 Release 下载

> TODO: 上传到 GitHub Release 后在此添加下载链接

## 权重文件说明

| 文件 | 大小 | 包含内容 | 用途 |
|------|------|------|------|
| `best_model.pth` | ~45MB | 纯 state_dict | 推理使用 |
| `final_model.pth` | ~50MB | state_dict + 模型名 + 类别映射 | 推理 + 继续训练 |

## 当前模型性能

| 指标 | 数值 |
|------|------|
| 模型 | ResNet18 |
| 输入尺寸 | 224×224 |
| Top-1 准确率 | 96.64% |
| Top-3 准确率 | 100.00% |
| 推理速度 | ~0.05s/张 (GPU) / ~0.2s/张 (CPU) |
| 模型大小 | ~44.7 MB |

## 注意事项

- 模型权重文件较大（40-50MB），**不上传到 GitHub**（已在 `.gitignore` 中排除）
- 如要在另一台电脑使用，请拷贝 `final_model.pth` 文件
- 推理时需要保持 `config.py` 中的 `MODEL_NAME` 和 `NUM_CLASSES` 与训练时一致
