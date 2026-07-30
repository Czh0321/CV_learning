import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
from safetensors.torch import load_model
from torch.distributed.checkpoint import load_state_dict
from torchvision import transforms
# === 迁移学习时需替换：数据集 ===
from torchvision.datasets import ImageFolder
# === 迁移学习时需替换：模型导入 ===
from model import GoogLeNet, inception


#处理训练集和验证集
def train_val_data_process():
    # === 迁移学习时需替换：数据集路径 + 变换 ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48600565, 0.45329682, 0.41553797],
                             std=[0.26266328, 0.25601205, 0.25864194])
    ])
    train_data = ImageFolder(root='./data/train', transform=transform)
    val_data = ImageFolder(root='./data/test', transform=transform)

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=8,
                                       pin_memory=True)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=8,
                                       pin_memory=True)

    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 优化器Adam，来优化梯度下降
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 定义损失函数（交叉熵损失函数）
    criterion = nn.CrossEntropyLoss()
    #将模型放入设备中
    model = model.to(device)
    #复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

# 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失函数列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

    #初始化参数
        #训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_acc = 0.0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集损失函数
        val_acc = 0.0
        # 训练集样本数
        train_num = 0
        # 验证集样本数
        val_num = 0

        for step, (images, labels) in enumerate(train_dataloader):
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            # 将模型设置为训练模式
            model.train()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            outputs = model(images)

            pre_lab = torch.argmax(outputs, dim=1)
            #outputs 是模型训练出来的值，和真实值算损失
            loss = criterion(outputs, labels)

            # 将梯度初始化为0(防止将前面轮次的梯度计算累加)
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据反向传播的梯度信息更新网络参数，以起到降低loss函数计算值的作用
            optimizer.step()

            #该批次样本数量
            train_loss += loss.item()*images.size(0) #第一个维度的大小，即为样本的数量
            train_acc += torch.sum(pre_lab == labels)

            train_num += images.size(0)

        # 验证集
        for step, (images, labels) in enumerate(val_dataloader):
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            # 将模型设置为验证模式
            model.eval()
            # 上下文管理器关闭梯度计算
            with torch.no_grad():
                # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                outputs = model(images)
                # 查找每一行中最大值对应的行标
                pre_lab = torch.argmax(outputs, dim=1)
                #outputs 是模型训练出来的值，和真实值算损失
                loss = criterion(outputs, labels)

                # 验证没有反向传播，没有更新参数的过程
                val_loss += loss.item()*images.size(0)
                val_acc += torch.sum(pre_lab == labels)

                val_num += images.size(0)

        # （每一轮）计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)

        train_acc_all.append(train_acc.double().item()/train_num)
        val_acc_all.append(val_acc.double().item()/val_num)

        print("{} Train Loss: {:.4f} Train Acc {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val Loss: {:.4f} val Acc {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        # 训练耗时
        time_used = time.time() - since
        print("训练耗费的时间：{:0f}m, {:0}s".format(time_used // 60, time_used % 60))

    # 选择最优参数
    # 加载最高准确率下的模型参数

    torch.save(best_model_wts, './best_GoogLeNet_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                      "train_loss": train_loss_all,
                                      "train_acc": train_acc_all,
                                      "val_loss": val_loss_all,
                                      "val_acc": val_acc_all})
    return train_process

def matplot_acc_loss(train_process):
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

if __name__ == '__main__':
    # === 迁移学习时需替换：模型实例化 ===
    GoogLeNet = GoogLeNet(inception)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(GoogLeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)


