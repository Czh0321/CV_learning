import os
import shutil
import random
from pathlib import Path


def split_dataset(src_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    在源文件夹内直接将图片移动到 train/val/test 子文件夹中。

    Args:
        src_dir: 数据集根目录，结构为 src_dir/类别名/图片.jpg
                 执行后变为 src_dir/train|val|test/类别名/图片.jpg
        train_ratio: 训练集比例
        val_ratio:   验证集比例
        test_ratio:  测试集比例
        seed: 随机种子，保证可复现
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "三个比例之和必须为 1"

    src_path = Path(src_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")

    random.seed(seed)

    # 支持的图片格式
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 遍历源目录下的每个类别文件夹（先收集，避免后续创建 train/val/test 时被遍历到）
    class_dirs = [d for d in src_path.iterdir() if d.is_dir() and d.name not in ('train', 'val', 'test')]
    if not class_dirs:
        raise ValueError(f"源目录下没有类别子文件夹: {src_dir}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in img_exts]
        if not images:
            print(f"警告: {class_name} 下没有图片，跳过")
            continue

        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # 剩余全部给 test，避免 rounding 误差
        n_test = n_total - n_train - n_val

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:],
        }

        for split_name, file_list in splits.items():
            split_class_dir = src_path / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for img in file_list:
                shutil.move(str(img), str(split_class_dir / img.name))

        print(f"{class_name}: 总计 {n_total} 张 -> train {n_train}, val {n_val}, test {n_test}")

    print(f"\n完成！数据已在 {src_dir} 下分好 train/val/test")


if __name__ == '__main__':
    # === 修改路径 ===
    SRC = 'data'   # 原始数据集路径（按类别分文件夹）

    split_dataset(SRC, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
