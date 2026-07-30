import os
import random
import shutil


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# === 迁移学习时需替换：原始数据集路径（猫狗图片混合放在一起的文件夹）===
file_path = 'data'

# === 迁移学习时需替换：类别列表（根据文件名前缀）===
classes = ['with_mask', 'without_mask']

# 创建 训练集train 文件夹，并由类名在其目录下创建子目录
mkfile('data/train')
for cla in classes:
    mkfile('data/train/' + cla)

# 创建 测试集test 文件夹，并由类名在其目录下创建子目录
mkfile('data/test')
for cla in classes:
    mkfile('data/test/' + cla)

# === 迁移学习时需替换：划分比例，训练集：测试集 = 9 : 1 ===
split_rate = 0.1

# 按文件名前缀归类，并按比例分成训练集和测试集
for cla in classes:
    # 获取该类别的所有图片（文件名以 cat. 或 dog. 开头）
    images = [img for img in os.listdir(file_path) if img.startswith(cla + '.')]
    num = len(images)
    test_num = int(num * split_rate)
    test_images = random.sample(images, test_num)

    for image in images:
        src_path = file_path + '/' + image
        if image in test_images:
            dst_path = 'data/test/' + cla + '/'
        else:
            dst_path = 'data/train/' + cla + '/'
        shutil.copy(src_path, dst_path + image)

    print(f'{cla}: 共 {num} 张，训练集 {num - test_num} 张，测试集 {test_num} 张')

print('数据划分完成！')
