import torch
from torch import nn
from torchsummary import summary

#神经网络约定好的写法(前三行)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 模型的初始化
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)  #第一个卷积，定义参数输入通道为1，输出通道为6，卷积核大小5
        self.sigmoid = nn.Sigmoid()  #激活函数
        self.s2 = nn.AvgPool2d(2, 2) #图像2维池化
        self.c3 = nn.Conv2d(6, 16, 5)
        self.s4 = nn.AvgPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)
    #前向传播
    def forward(self, x):
        x = self.sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)

        # 中间全连接增加sigmoid激活
        x = self.sigmoid(self.f5(x))
        x = self.sigmoid(self.f6(x))
        # 输出层不加激活
        x = self.f7(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #模型上面已经搭建完成，将模型塞到设备中
    model = LeNet().to(device)
    print(summary(model,(1,28,28)))



