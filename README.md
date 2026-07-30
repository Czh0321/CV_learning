# 炮哥带你学CV

深度学习经典模型复现学习路线，从 LeNet 到 MobileNetV3。

## 学习路线

| 顺序 | 模型 | 核心思想 | 状态 |
|------|------|---------|------|
| 1 | [LeNet](./LeNet) | CNN 开山之作，卷积+池化+全连接 | ✅ |
| 2 | [AlexNet](./AlexNet) | ReLU + Dropout + 双 GPU | ✅ |
| 3 | [VGG](./VGG) | 小卷积核堆叠，统一 3×3 | ✅ |
| 4 | [GoogLeNet](./GoogLeNet) | Inception 多分支并行 | ✅ |
| 5 | [ResNet18](./ResNet18) | 残差连接，解决退化问题 | ✅ |
| 6 | [ResNet18_mask](./ResNet18_mask) | 实战：口罩检测（迁移学习） | ✅ |
| 7 | [MobileNetV3](./MobileNet_V3) | 轻量化：倒残差 + SE + Hardswish | 🚧 |

## 通用工具

| 脚本 | 用途 |
|------|------|
| [split_dataset.py](./ResNet18/split_dataset.py) | 数据集按类别分层拆分 train/val/test |
| [mean_std.py](./ResNet18_mask/mean_std.py) | 只对 train 文件夹计算 mean/std，避免数据泄露 |

## 环境依赖

- Python 3.10+
- PyTorch 2.x
- torchvision
- matplotlib
- pandas
- tqdm
- torchsummary

## 关键学习记录

- **数据泄露**：mean/std 只对 train 计算，val/test 不参与
- **数据划分**：磁盘上物理分开 train/val/test，不用 random_split
- **全连接层**：必须有，CrossEntropyLoss 会把缺失的 FC 当多类处理
- **早停机制**：监控 val_loss，patience 容忍轮数
- **预训练模型**：小数据集优先用 ImageNet 预训练权重
