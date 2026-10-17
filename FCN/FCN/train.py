import os
import time
import datetime
import subprocess
from pathlib import Path

import torch

from model import fcn_resnet50
from utils.train_and_eval import (
    train_one_epoch,
    evaluate,
    create_lr_scheduler,
)
from utils.my_dataset import VOCData
from utils.create_exp_folder import create_exp_folder
from utils.segmentation_transforms import get_transform
from utils.plot_results import plot_training_curves


# ==================== 数据集默认路径 ====================
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "VOCtrainval_11-May-2012 (1)"
    / "VOCdevkit"
)

# 第一版使用的 COCO 预训练 FCN 权重
DEFAULT_WEIGHTS_PATH = Path("weights/fcn_resnet50_coco.pth")


# ==================== 创建模型 ====================
def create_model(num_classes, weights):
    """
    按照第一版的方式创建 FCN-ResNet50，
    默认加载 COCO 预训练 FCN 权重。
    """
    model = fcn_resnet50(num_classes=num_classes)

    if not weights:
        print("警告：没有加载预训练权重，模型可能从随机初始化开始训练。")
        return model

    weights_path = Path(weights)

    # 先检查当前工作目录；如果找不到，再尝试脚本所在目录
    if not weights_path.is_absolute() and not weights_path.is_file():
        script_relative_path = Path(__file__).resolve().parent / weights_path

        if script_relative_path.is_file():
            weights_path = script_relative_path

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"找不到预训练权重文件：{weights_path}\n"
            "请检查 weights/fcn_resnet50_coco.pth 是否存在，"
            "或通过 --weights 指定正确路径。"
        )

    print(f"正在加载 COCO 预训练权重：{weights_path}")

    weights_dict = torch.load(
        str(weights_path),
        map_location="cpu",
    )

    # VOC2012 是 20 个前景类别 + 1 个背景类别。
    # 如果当前模型不是 21 类，移除最后分类层的参数。
    if num_classes != 21:
        for key in list(weights_dict.keys()):
            if "classifier.4" in key:
                del weights_dict[key]

    missing_keys, unexpected_keys = model.load_state_dict(
        weights_dict,
        strict=False,
    )

    if missing_keys:
        print(f"Missing keys: {missing_keys}")

    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    print("预训练权重加载完成。")

    return model


# ==================== 获取 GPU 显存使用量 ====================
def get_gpu_usage():
    """
    返回当前 GPU 已使用的显存，单位 MB。
    如果没有可用 GPU 或 nvidia-smi 不可用，则返回 0。
    """
    if not torch.cuda.is_available():
        return 0

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )

        used, total = map(
            int,
            result.strip().splitlines()[0].split(","),
        )

        return used

    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        ValueError,
        IndexError,
    ):
        return 0


# ==================== 训练主函数 ====================
def train(args):
    batch_size = args.batch_size
    num_classes = args.num_classes + 1  # 加上背景类别
    train_epoch = args.epochs
    num_workers = args.workers

    # 选择训练设备
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("FCN-ResNet50 VOC2012 语义分割训练")
    print(f"训练设备：{device}")
    print(f"前景类别数：{args.num_classes}")
    print(f"模型输出类别数（含背景）：{num_classes}")
    print(f"Batch size：{batch_size}")
    print(f"最大训练轮数：{args.epochs}")
    print(f"早停耐心值：{args.patience}")
    print(f"早停最小提升幅度：{args.min_delta}")
    print(f"混合精度：{args.amp and device.type == 'cuda'}")
    print("=" * 60)

    # ==================== 创建实验文件夹 ====================
    exp_folder, weights_folder = create_exp_folder()

    print(f"实验目录：{exp_folder}")
    print(f"权重目录：{weights_folder}")

    # ==================== 加载数据集 ====================
    print("\n正在加载训练集……")

    train_dataset = VOCData(
        args.data_path,
        transforms=get_transform(train=True),
        txt_name="train.txt",
    )

    print("正在加载验证集……")

    val_dataset = VOCData(
        args.data_path,
        transforms=get_transform(train=False),
        txt_name="val.txt",
    )

    print(f"训练集样本数：{len(train_dataset)}")
    print(f"验证集样本数：{len(val_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError(
            "训练集为空，请检查 data-path 和 ImageSets/Segmentation/train.txt。"
        )

    if len(val_dataset) == 0:
        raise RuntimeError(
            "验证集为空，请检查 data-path 和 ImageSets/Segmentation/val.txt。"
        )

    # ==================== DataLoader ====================
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=val_dataset.collate_fn,
    )

    # ==================== 创建模型 ====================
    model = create_model(
        num_classes=num_classes,
        weights=args.weights,
    )

    model.to(device)

    # ==================== 优化器 ====================
    # 保持第一版的 backbone + classifier 参数设置
    params_to_optimize = [
        {
            "params": [
                p for p in model.backbone.parameters()
                if p.requires_grad
            ]
        },
        {
            "params": [
                p for p in model.classifier.parameters()
                if p.requires_grad
            ]
        },
    ]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # ==================== 混合精度训练 ====================
    scaler = (
        torch.cuda.amp.GradScaler()
        if args.amp and device.type == "cuda"
        else None
    )

    if scaler is not None:
        print("已启用 CUDA 混合精度训练。")
    else:
        print("未启用混合精度训练。")

    # ==================== 学习率调度器 ====================
    lr_scheduler = create_lr_scheduler(
        optimizer,
        len(train_loader),
        args.epochs,
        warmup=True,
    )

    # ==================== 模型保存路径 ====================
    best_model_path = os.path.join(
        weights_folder,
        f"best_model_{args.num_classes}.pth",
    )

    last_model_path = os.path.join(
        weights_folder,
        f"last_model_{args.num_classes}.pth",
    )

    # ==================== 早停初始化 ====================
    # best_acc：用于保存历史最佳 Mean Accuracy 模型
    best_acc = float("-inf")

    # early_stop_best：用于判断是否达到早停要求
    early_stop_best = float("-inf")
    patience_counter = 0

    # ==================== 训练记录 ====================
    train_losses = []
    val_losses = []
    val_metrics_history = []

    start_time = time.time()

    # ==================== 开始训练 ====================
    for epoch in range(args.epochs):
        print("\n" + "=" * 60)
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print("=" * 60)

        gpu_used = get_gpu_usage()

        # ---------- 训练 ----------
        loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            train_epoch,
            gpu_used,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )

        train_losses.append(loss)

        # ---------- 验证 ----------
        metrics = evaluate(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
        )

        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        # 恢复第一版的最佳模型评价指标
        current_acc = float(metrics["Mean Accuracy"])

        print(f"训练损失：{loss:.4f}")
        print(f"验证损失：{float(metrics['Loss']):.4f}")
        print(f"验证 Mean Accuracy：{current_acc:.4f}")

        # ---------- 保存历史最佳模型 ----------
        # 与第一版一致：Mean Accuracy 只要严格提升，就保存模型。
        if current_acc > best_acc:
            best_acc = current_acc

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(f"当前最佳 Mean Accuracy：{best_acc:.4f}")
            print(f"已保存最佳模型：{best_model_path}")

        # ---------- 早停判断 ----------
        # 只有超过早停基准至少 min_delta，才重置耐心计数。
        if current_acc > early_stop_best + args.min_delta:
            early_stop_best = current_acc
            patience_counter = 0

            print(
                f"验证指标有效提升，早停计数重置为 "
                f"0/{args.patience}"
            )
        else:
            patience_counter += 1

            print(
                f"验证指标未达到早停提升标准，"
                f"计数：{patience_counter}/{args.patience}"
            )

        # ---------- 保存最后一轮模型 ----------
        torch.save(
            model.state_dict(),
            last_model_path,
        )

        print(f"已保存当前模型：{last_model_path}")

        # ---------- 触发早停 ----------
        if patience_counter >= args.patience:
            print("\n" + "=" * 60)
            print("触发早停机制，训练提前结束！")
            print(f"当前轮数：{epoch + 1}")
            print(f"最佳 Mean Accuracy：{best_acc:.4f}")
            print(f"最佳模型路径：{best_model_path}")
            print("=" * 60)
            break

    # ==================== 训练结束 ====================
    total_time = time.time() - start_time

    total_time_str = str(
        datetime.timedelta(seconds=int(total_time))
    )

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"总训练时间：{total_time_str}")
    print(f"最佳 Mean Accuracy：{best_acc:.4f}")
    print(f"最佳模型路径：{best_model_path}")
    print(f"最后模型路径：{last_model_path}")
    print("=" * 60)

    # ==================== 绘制训练曲线 ====================
    plot_training_curves(
        train_losses,
        val_losses,
        val_metrics_history,
        weights_folder,
    )


# ==================== 命令行参数 ====================
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch FCN-ResNet50 VOC2012 语义分割训练"
    )

    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        type=str,
        help="COCO 预训练 FCN 权重路径",
    )

    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="VOCdevkit 根目录",
    )

    parser.add_argument(
        "--num-classes",
        default=20,
        type=int,
        help="前景类别数，不包括背景",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="训练设备",
    )

    # 恢复第一版的默认 Batch size
    parser.add_argument(
        "--batch-size",
        default=10,
        type=int,
        help="训练批次大小",
    )

    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="最大训练轮数",
    )

    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="DataLoader 工作进程数",
    )

    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
        help="初始学习率",
    )

    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="SGD 动量",
    )

    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        dest="weight_decay",
        help="权重衰减",
    )

    # 恢复第一版的默认开启状态，同时提供关闭选项
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="启用 CUDA 混合精度训练",
    )

    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="关闭混合精度训练",
    )

    parser.set_defaults(amp=True)

    # ==================== 早停参数 ====================
    parser.add_argument(
        "--patience",
        default=8,
        type=int,
        help="连续多少轮未达到有效提升后停止训练",
    )

    parser.add_argument(
        "--min-delta",
        default=0.001,
        type=float,
        help="Mean Accuracy 的最小有效提升幅度",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs 必须大于等于 1")

    if args.patience < 1:
        parser.error("--patience 必须大于等于 1")

    if args.min_delta < 0:
        parser.error("--min-delta 不能小于 0")

    if args.num_classes < 1:
        parser.error("--num-classes 必须大于等于 1")

    if args.batch_size < 1:
        parser.error("--batch-size 必须大于等于 1")

    return args


# ==================== 程序入口 ====================
if __name__ == "__main__":
    args = parse_args()
    train(args)
