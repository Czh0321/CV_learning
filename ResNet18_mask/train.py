import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
# 进度条
from tqdm import tqdm
# === 迁移学习时需替换：数据集类 ===
from torchvision.datasets import ImageFolder
# === 迁移学习时需替换：模型导入 ===
from model import Residual, ResNet18


#处理训练集和验证集
def train_val_data_process():
    # === 迁移学习时需替换：数据集路径 ===
    ROOT_TRAIN = r'data/train'   # === 迁移学习时需替换 ===
    ROOT_VAL   = r'data/val'     # === 迁移学习时需替换 ===

    # === 迁移学习时需替换：mean/std（只对 train 文件夹计算，避免数据泄露）===
    normalized = transforms.Normalize(mean=[0.597, 0.554, 0.539], std=[0.317 ,0.318 ,0.321 ])

    # === 迁移学习时需替换：train_transform 可加数据增强 ===
    train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalized])
    val_transform   = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalized])

    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_data   = ImageFolder(ROOT_VAL,   transform=val_transform)

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=8,
                                       pin_memory=True)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=64,
                                       shuffle=False,
                                       num_workers=8,
                                       pin_memory=True)

    return train_dataloader, val_dataloader

def test_data_process():
    # === 迁移学习时需替换：数据集路径 ===
    ROOT_TEST = r'data/test'    # === 迁移学习时需替换 ===

    # === 迁移学习时需替换：mean/std（与 train 保持一致）===
    normalized = transforms.Normalize(mean=[0.597, 0.554, 0.539], std=[0.317, 0.318, 0.321])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalized])

    test_data = ImageFolder(ROOT_TEST, transform=test_transform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=64,
                                      shuffle=False,
                                      num_workers=8,
                                      pin_memory=True)

    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    test_correct = 0.0
    test_num = 0

    model.eval()
    with torch.no_grad():
        for test_data_x, test_data_y in tqdm(test_dataloader, desc='Test'):
            test_data_x = test_data_x.to(device, non_blocking=True)
            test_data_y = test_data_y.to(device, non_blocking=True)

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)

            test_correct += torch.sum(pre_lab == test_data_y)
            test_num += test_data_y.size(0)

    test_acc = test_correct.item() / test_num
    print('\n' + '=' * 50)
    print('Test Acc: {:.4f}'.format(test_acc))
    print('=' * 50)
    return test_acc

def train_model_process(model, train_dataloader, val_dataloader, num_epochs, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # === 迁移学习时需替换：优化器 + 学习率 ===
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 定义损失函数（交叉熵损失函数）
    criterion = nn.CrossEntropyLoss()
    #将模型放入设备中
    model = model.to(device)
    #复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

# 初始化参数
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
    # === 早停机制参数 ===
    early_stop_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print('\n' + '=' * 50)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('=' * 50)

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

        for step, (images, labels) in enumerate(tqdm(train_dataloader, desc='Train')):
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
        for step, (images, labels) in enumerate(tqdm(val_dataloader, desc='Val')):
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

        print('\n[Epoch {}] Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1], val_loss_all[-1], val_acc_all[-1]))

        # === 早停判断：验证集损失连续 patience 轮未下降则停止 ===
        if val_loss_all[-1] < best_val_loss:
            best_val_loss = val_loss_all[-1]
            early_stop_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
            print('早停计数: {}/{}'.format(early_stop_counter, patience))
            if early_stop_counter >= patience:
                print('早停触发！连续 {} 轮验证集损失未下降。'.format(patience))
                break

        # 训练耗时
        time_used = time.time() - since
        print('累计耗时: {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))

    # 选择最优参数
    # 加载最高准确率下的模型参数

    # === 迁移学习时需替换：权重保存路径 ===
    torch.save(best_model_wts, 'best_resnet18mask_model.pth')

    actual_epochs = len(train_loss_all)
    train_process = pd.DataFrame(data={"epoch": range(actual_epochs),
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
    model = ResNet18(Residual)
    train_dataloader, val_dataloader = train_val_data_process()
    # === 迁移学习时需替换：训练轮数 + 早停耐心值 ===
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=20, patience=5)

    # === 加载最优权重，跑一遍 test ===
    model.load_state_dict(torch.load('best_resnet18mask_model.pth', weights_only=True))
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    matplot_acc_loss(train_process)


