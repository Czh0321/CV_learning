import copy
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from tqdm import tqdm

# === 迁移学习时需替换：模型导入 ===
from model import SwinTransformer

# ============================================================
# 一、配置区（唯一需要重点修改的地方，其余代码通用）
# ============================================================
# 数据根目录：里面应包含 train/ 和 valid/ 两个子文件夹，
# 每个子文件夹下再按类别分子文件夹（0_real/ 1_fake 或 real/ fake）。
# 例：
#   data/
#   ├── train/
#   │   ├── real/
#   │   └── fake/
#   └── valid/
#       ├── real/
#       └── fake/
DATA_ROOT = './data'          # 数据集路径，迁移时改为你的数据路径

# 模型与训练超参数
IMG_SIZE   = 224              # 输入图像尺寸，需与预训练权重一致（Swin-T 用 224）
NUM_CLASS  = 2                # 类别数：real(0) / fake(1)，迁移时改为你的类别数
BATCH_SIZE = 8                # 训练批次大小（显存不足就调小，如 4、2）
NUM_EPOCHS = 20               # 训练轮数
L_RATE     = 1e-4             # 学习率
# 迁移学习时，建议用较小学习率（如 1e-4 ~ 1e-5）微调，避免破坏预训练学到的特征

# 早停：验证集准确率连续 patience 轮没有提升，自动停止训练（节省时间和费用）
EARLY_STOP_PATIENCE = 5       # 容忍多少轮不提升（如 5 轮没进步就停）

# === 预训练权重相关 ===
USE_PRETRAIN = True           # 是否使用预训练权重
PRETRAIN_PATH = './swin_tiny_patch4_window7_224.pth'  # 预训练权重文件路径
FREEZE_BACKBONE = True       # 是否冻结 backbone（PatchEmbed + 所有 Stage）
                             # True  -> 只训练分类头，速度快、训练量小、适合数据极少的场景
                             # False -> 全部层一起微调，效果通常更好，模型下载见注释下方

# 模型结构超参数（默认即 Swin-T，与官方一致）
EMBED_DIM  = 96
DEPTHS     = [2, 2, 6, 2]
NUM_HEADS  = [3, 6, 12, 24]
WINDOW_SIZE = 7
MLP_RATIO  = 4.
# 官方 Swin-T 预训练权重下载（与你的 model.py 结构完全一致，可直接加载）：
#   https://github.com/microsoft/Swin-Transformer  （官方仓库）
#   或直接命令行下载：
#   wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
#   下载后放到当前目录即可

# ============================================================
# 二、数据加载
# ============================================================
def load_data():
    # 数据增强 + 归一化。
    # 注意：均值和标准差用的是 ImageNet 统计值（Swin 预训练时的统计量），迁移新人脸数据也能直接沿用。
    # 若要重新统计自己数据集，可用 train_mean = data.mean((0,1,2,3)), train_std = data.std(...)
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),          # 缩放图像
            transforms.RandomHorizontalFlip(p=0.5),            # 随机水平翻转（数据增强）
            transforms.RandomRotation(degrees=10),             # 随机旋转（数据增强）
            transforms.ToTensor(),                             # 转张量 [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])    # 归一化
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    # ImageFolder：按文件夹名自动给类别编号（real=0, fake=1，取决于字母顺序）
    train_dataset = datasets.ImageFolder(root=DATA_ROOT + '/train', transform=data_transform['train'])
    val_dataset   = datasets.ImageFolder(root=DATA_ROOT + '/valid', transform=data_transform['val'])
    print(f"训练集类别: {train_dataset.classes}，样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 如果显存不足，建议先调小 BATCH_SIZE，而不是盲目加大 num_workers
    train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ============================================================
# 三、迁移学习：加载预训练权重（核心逻辑）
# ============================================================
def load_pretrained(model):
    print("加载预训练权重:", PRETRAIN_PATH)
    # map_location='cpu' 保证无论有无 GPU 都能加载
    checkpoint = torch.load(PRETRAIN_PATH, map_location='cpu')

    # 预训练权重可能有多种包装格式，统一取出真正包含权重的那一层
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 去掉分类头相关键，因为我们的类别数(2)和预训练(1000)不同，形状对不上
    # 这样 load_state_dict(strict=False) 时只会跳过 head，其余层（PatchEmbed + Stages）正常加载
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith('head.')}

    # strict=False：允许部分键缺失/多余（这里主要是 head 被我们删掉了）
    model.load_state_dict(state_dict, strict=False)
    print("预训练权重加载完成（分类头除外，分类头使用随机初始化）")

    # 可选：冻结 backbone，只训练分类头
    if FREEZE_BACKBONE:
        # 遍历所有参数，冻结除 head 以外的所有层
        for name, param in model.named_parameters():
            if not name.startswith('head'):
                param.requires_grad = False
        print("已冻结 backbone，仅训练分类头")


# ============================================================
# 四、训练主体
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)
    model = model.to(device)

    # 优化器：只更新 requires_grad=True 的参数（若冻结了 backbone，则跳过冻结层）
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=L_RATE, weight_decay=1e-4)

    # 损失函数：交叉熵，适用于多分类/二分类
    criterion = nn.CrossEntropyLoss()

    # 学习率调度：每 10 个 epoch 学习率乘 0.1，让后期收敛更平稳
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0  # 早停计数器：连续多少轮没提升

    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []
    since = time.time()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1} ({L_RATE if epoch==0 else ""})')
        print('-' * 40)

        # ---- 训练阶段 ----
        train_loss, train_acc, train_num = 0.0, 0.0, 0
        model.train()  # 启用训练模式（Dropout 生效）
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=True)
        for images, labels in train_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(images)                          # 前向传播
            loss = criterion(outputs, labels)                # 计算损失
            pre_lab = torch.argmax(outputs, dim=1)           # 预测类别

            optimizer.zero_grad()                            # 梯度清零，防止累加
            loss.backward()                                  # 反向传播计算梯度
            optimizer.step()                                 # 更新参数

            train_loss += loss.item() * images.size(0)       # 累加 batch 损失（乘样本数加权）
            train_acc += torch.sum(pre_lab == labels).item()
            train_num += images.size(0)

            # 实时更新进度条显示当前 loss 和 acc
            train_pbar.set_postfix({
                'loss': f'{train_loss/train_num:.4f}',
                'acc': f'{train_acc/train_num:.4f}'
            })

        # ---- 验证阶段 ----
        val_loss, val_acc, val_num = 0.0, 0.0, 0
        model.eval()  # 验证模式（关闭 Dropout 和 BN 统计更新）
        with torch.no_grad():  # 不计算梯度，节省显存
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=True)
            for images, labels in val_pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                pre_lab = torch.argmax(outputs, dim=1)

                val_loss += loss.item() * images.size(0)
                val_acc += torch.sum(pre_lab == labels).item()
                val_num += images.size(0)

                # 实时更新进度条显示当前 loss 和 acc
                val_pbar.set_postfix({
                    'loss': f'{val_loss/val_num:.4f}',
                    'acc': f'{val_acc/val_num:.4f}'
                })

        # 记录本轮指标
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_acc / train_num)
        val_acc_all.append(val_acc / val_num)

        # 学习率调度更新
        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss_all[-1]:.4f}  Train Acc: {train_acc_all[-1]:.4f}")
        print(f"Val   Loss: {val_loss_all[-1]:.4f}  Val   Acc: {val_acc_all[-1]:.4f}  (lr={cur_lr:.2e})")

        # 保存验证集准确率最高的模型
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0  # 有提升，重置计数器
            print(f"* 新的最佳准确率: {best_acc:.4f}，已更新最优权重")
        else:
            early_stop_counter += 1  # 没提升，计数器+1

        time_used = time.time() - since
        print(f"已用时: {int(time_used // 60)}m {int(time_used % 60)}s")

        # 早停判断：连续 patience 轮没提升，提前结束训练
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发：验证集准确率已连续 {EARLY_STOP_PATIENCE} 轮未提升，提前结束训练")
            break

    # 训练结束，保存最优模型
    # === 权重保存路径 ===
    torch.save(best_model_wts, './best_swin_model.pth')
    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}，最优权重已保存到 ./best_swin_model.pth")

    # 返回训练过程数据，便于画图
    return pd.DataFrame(data={"epoch": range(num_epochs),
                              "train_loss": train_loss_all,
                              "train_acc": train_acc_all,
                              "val_loss": val_loss_all,
                              "val_acc": val_acc_all})


# ============================================================
# 五、画图
# ============================================================
def plot_process(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss"], 'r-', label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss"], 'b-', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc"], 'r-', label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc"], 'b-', label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.show()


# ============================================================
# 六、主入口
# ============================================================
if __name__ == '__main__':
    # 1. 实例化模型（默认 Swin-T 结构，分类头改为 2 类）
    model = SwinTransformer(
        img_size=IMG_SIZE,
        patch_size=4,
        in_chans=3,
        num_classes=NUM_CLASS,
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True
    )

    # 2. 加载预训练权重（迁移学习核心）
    if USE_PRETRAIN:
        load_pretrained(model)
    else:
        print("未使用预训练权重，将从零开始训练")

    # 3. 加载数据
    train_loader, val_loader = load_data()

    # 4. 训练
    train_process = train_model(model, train_loader, val_loader, NUM_EPOCHS)

    # 5. 画图
    plot_process(train_process)
