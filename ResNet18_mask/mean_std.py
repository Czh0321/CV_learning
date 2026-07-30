import os
import numpy as np
from PIL import Image



# 只计算训练集的，否则会有数据泄露

# === 迁移学习时需替换：数据集路径 ===
folder_path = 'data/train'

# 初始化累积变量
total_pixels = 0
sum_pixel_values = np.zeros(3)      # RGB 三通道像素值之和
sum_pixel_values_sq = np.zeros(3)  # RGB 三通道像素值平方之和

# 遍历文件夹中的图片文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image, dtype=np.float64)

            # 归一化像素值到 0-1 之间
            image_array = image_array / 255.0

            # 累加每个通道的像素值和像素值平方
            sum_pixel_values += image_array.sum(axis=(0, 1))
            sum_pixel_values_sq += (image_array ** 2).sum(axis=(0, 1))
            total_pixels += image_array.shape[0] * image_array.shape[1]

# 计算均值和标准差
mean = sum_pixel_values / total_pixels
std = np.sqrt(sum_pixel_values_sq / total_pixels - mean ** 2)

print(f'mean: {mean}')
print(f'std: {std}')
