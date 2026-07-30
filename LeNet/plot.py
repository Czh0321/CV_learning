from click.core import batch
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(24), transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

for step, (images, labels) in enumerate(train_loader):
    if step > 0:
        break
batch_x = images.squeeze().numpy() #形状通常是 (BatchSize, 通道数, 高, 宽),加上这个.squeeze()将一维的batchsize去掉
batch_y = labels.numpy()  #将张量转化成numpy数组
class_labels = train_data.classes
print(class_labels)


# 可视化一个batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii+1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_labels[batch_y[ii]], size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()