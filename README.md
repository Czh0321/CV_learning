# CV_learning

从 LeNet 到 MobileNetV3 的深度学习学习路线

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

| 脚本 | 位置 | 用途 |
|------|------|------|
| split_dataset.py | [ResNet18/split_dataset.py](./ResNet18/split_dataset.py) | 数据集按类别分层拆分 train/val/test（直接移动，不复制） |
| mean_std.py | [ResNet18_mask/mean_std.py](./ResNet18_mask/mean_std.py) | 只对 train 文件夹计算 mean/std，避免数据泄露 |

## 环境依赖

- Python 3.10+
- PyTorch 2.x
- torchvision
- matplotlib / pandas / tqdm / torchsummary

---

## 学习笔记

### 1. 数据集划分

- **磁盘物理分开**：train/val/test 三个独立文件夹，彻底避免数据泄露，实验可复现
- **不用 `random_split`**：在代码里随机划分会导致每次运行结果不同，且 mean/std 会泄露
- **`round()` 问题**：`round(0.8*n) + round(0.2*n)` 可能不等于 `n`，应改为 `int()` + 减法
- **分层采样**：按类别比例划分，避免某类全在 train 或全在 val

标准文件结构：
```
data/
├── train/          # 70%
│   ├── with_mask/
│   └── without_mask/
├── val/            # 15%
│   ├── with_mask/
│   └── without_mask/
└── test/           # 15%
    ├── with_mask/
    └── without_mask/
```

### 2. 数据泄露

- **mean/std 只对 `data/train` 计算**，不能碰 val/test
- `random_split` + 整个文件夹算 mean/std = 有泄露（val 数据的统计量被算进去了）
- 影响虽小（80% 和 100% 的均值差异不大），但应从工程规范角度避免

### 3. 全连接层（FC）

- **必须有全连接层**：没有 FC 时 `CrossEntropyLoss` 会把 512 维输出当成 512 类处理
- 能训练能收敛，但浪费 502 维的梯度计算，且类别数 > 512 时会报错
- 卷积层 = 提取特征（眼睛），全连接层 = 做决策（大脑）
- 加 FC 不一定精度更高，但更合理、更高效

### 4. 训练优化

- **验证集 `shuffle=False`**：验证集不需要打乱，减少无谓开销
- **学习率**：`lr=0.005` 对 ResNet18 太大，val_loss 会剧烈震荡；建议 `0.0001~0.001`
- **数据增强**：`RandomHorizontalFlip` + `ColorJitter`，防过拟合
- **AdamW 替代 Adam**：带 weight_decay 正则化，防过拟合
- **早停机制**：监控 `val_loss`，连续 patience 轮不下降则停止
- **最优权重保存**：和早停统一监控 `val_loss`，不要一个看 loss 一个看 acc

### 5. 预训练模型 vs 从头训练

| | 从头训练 | ImageNet 预训练 |
|---|---|---|
| 参数初始化 | 随机 | 加载 ImageNet 权重 |
| 数据量需求 | 大（几万张） | 小（几百张即可） |
| 收敛速度 | 慢（20+ epoch） | 快（3~5 epoch） |
| 过拟合风险 | 高 | 低 |
| 科研/工程 | 特殊场景 | 默认首选 |

```python
# 预训练模型加载
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 只换最后一层
```

### 6. test 集评估

- **只在最后跑一次**，用 best 权重评估最终性能
- train 每轮跑（更新参数），val 每轮跑（选 best + 早停），test 只跑一次
- 如果每轮都看 test = 拿考试题当练习题 = 数据泄露

### 7. MobileNetV3 核心设计

- **倒残差结构（Bottleneck）**：窄→宽→窄（和 ResNet 反过来）
  - 1x1 扩展卷积（升维）→ 深度卷积（提取特征）→ SE → 1x1 压缩卷积（降维）
- **深度可分离卷积**：`groups=channels`，每个通道单独卷，参数量从 `C×C×K` 降到 `C×K`
- **SE 模块（通道注意力）**：全局平均池化 → FC → Sigmoid → 通道加权
- **混合激活函数**：前期用 ReLU（省算力），后期用 Hardswish（精度好）
- **BN（BatchNorm）**：论文表格隐含，除最后两层标 `NBN` 外每层都有
- **参数量极小**：移动端友好

### 8. 工程规范

- **固定随机种子**：`random.seed()` + `torch.manual_seed()` + `torch.cuda.manual_seed_all()`
- **`num_workers`**：Windows 下需要 `if __name__ == '__main__':` 守卫，否则多进程报错
- **进度条**：`tqdm(dataloader, desc='Train')`
- **argparse 配置化**：命令行传参，避免改代码改错版本

### 9. Git / GitHub

- `.gitignore` 排除：权重文件 `*.pth`、数据集 `data/`、日志 `logs/`、IDE 配置
- 推送步骤：`git add .` → `git commit -m "msg"` → `git push`
- Token 认证：用 Classic Token（`ghp_` 开头），勾选 `repo` 权限
- 冲突解决：`git pull --allow-unrelated-histories` → 解决冲突 → `git push`
