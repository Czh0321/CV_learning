import os

import torch.utils.data as data
from PIL import Image


class VOCData(data.Dataset):
    def __init__(self, voc_root, transforms=None, txt_name: str = "train.txt"):
        super(VOCData, self).__init__()

        # 通过voc_root和year构造数据集路径
        root = os.path.join(voc_root, f"VOC2012")

        # 图像和标签的目录
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        # 获取指定txt文件的路径，这个txt文件包含了图像文件名的列表
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)

        # 确保txt文件存在
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        # 读取txt文件，获取图像文件名
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # 根据文件名构造图像和标签的路径
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

        # 确保图像数量和标签数量一致
        assert (len(self.images) == len(self.masks))

        # 保存图像预处理的转换操作（如果有的话）
        self.transforms = transforms

    def __getitem__(self, index):
        """
        获取一个样本（图像和标签）

        Args:
            index (int): 索引

        Returns:
            tuple: (图像, 标签) 其中标签是图像的语义分割标注
        """
        # 打开图像文件并转换为RGB模式
        img = Image.open(self.images[index]).convert('RGB')

        # 打开对应的标签文件（通常是png格式的分割标签）
        target = Image.open(self.masks[index])

        # 如果定义了数据预处理（例如数据增强），则应用它
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # 返回图像和标签
        return img, target

    def __len__(self):
        # 返回数据集的大小（图像的数量）
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """
                这个方法用于合并一个batch的样本
                这里使用了cat_list函数将图像和标签打包成一个批次

                Args:
                    batch (list): 从数据集中获取的样本列表，包含了图像和标签

                Returns:
                    tuple: (batched_images, batched_targets)，一个批次的图像和标签
        """
        images, targets = list(zip(*batch))  # 拆解批次中的图像和标签
        batched_imgs = cat_list(images, fill_value=0)  # 使用cat_list函数合并图像，填充值为0
        batched_targets = cat_list(targets, fill_value=255)  # 使用cat_list函数合并标签，填充值为255
        return batched_imgs, batched_targets



def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size  # 计算最终批次的形状，(batch_size, channel, height, width)

    # 创建一个新的空张量，填充为fill_value（默认是0），大小为batch_shape
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)

    # 将每个图像的内容复制到批次张量中的相应位置，确保所有图像的尺寸都一致
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

    # 将每个图像的内容复制到批次张量中的相应位置，确保所有图像的尺寸都一致
    return batched_imgs

