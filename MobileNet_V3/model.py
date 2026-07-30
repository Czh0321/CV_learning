import torch
import torch.nn as nn
from torchsummary import summary


class SEModule(nn.Module):
    """挤压激励模块：学习每个通道的重要程度，抑制无用的通道"""
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)       # 全局平均池化 → [B, C]
        y = self.fc(y).view(b, c, 1, 1)       # 学习通道权重 → [B, C, 1, 1]
        return x * y                            # 用权重加权原始特征


class Bottleneck(nn.Module):
    """MobileNetV3 倒残差块：1x1扩展 → 深度卷积 → SE → 1x1压缩 + 残差连接"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_size, use_se=True, act='hs'):
        super(Bottleneck, self).__init__()
        self.use_se = use_se
        # 残差连接的条件：stride=1 且输入输出通道相同
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # 激活函数：前期用 ReLU，后期用 Hardswish（更省计算）
        self.act = nn.Hardswish(inplace=True) if act == 'hs' else nn.ReLU(inplace=True)

        # 1x1 扩展卷积（升维，增加特征表达力）
        self.expand_conv = nn.Conv2d(in_channels, expand_size, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expand_size)

        # 深度可分离卷积（逐通道卷积，参数量极少）
        self.depthwise_conv = nn.Conv2d(
            expand_size, expand_size, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, groups=expand_size, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expand_size)

        # SE 模块（通道注意力）
        if use_se:
            self.se = SEModule(expand_size)

        # 1x1 压缩卷积（降维回低通道数，不加激活函数）
        self.project_conv = nn.Conv2d(expand_size, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.act(self.expand_bn(self.expand_conv(x)))
        y = self.act(self.depthwise_bn(self.depthwise_conv(y)))
        if self.use_se:
            y = self.se(y)
        y = self.project_bn(self.project_conv(y))

        if self.use_residual:
            y = y + x

        return y


class MobileNetV3(nn.Module):
    def __init__(self, Bottleneck):
        super(MobileNetV3, self).__init__()
        # 特征提取头部：3x3 卷积 + BN + Hardswish
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        # 阶段1：无 SE，ReLU 激活
        self.b2 = nn.Sequential(
            Bottleneck(16, 16, kernel_size=3, stride=1, expand_size=16, use_se=False, act='re'),
        )

        # 阶段2：无 SE，ReLU 激活
        self.b3 = nn.Sequential(
            Bottleneck(16, 24, kernel_size=3, stride=2, expand_size=64, use_se=False, act='re'),
            Bottleneck(24, 24, kernel_size=3, stride=1, expand_size=72, use_se=False, act='re'),
        )

        # 阶段3：有 SE，ReLU 激活
        self.b4 = nn.Sequential(
            Bottleneck(24, 40, kernel_size=5, stride=2, expand_size=72, use_se=True, act='re'),
            Bottleneck(40, 40, kernel_size=5, stride=1, expand_size=120, use_se=True, act='re'),
        )

        # 阶段4：无 SE，Hardswish 激活
        self.b5 = nn.Sequential(
            Bottleneck(40, 80, kernel_size=3, stride=2, expand_size=240, use_se=False, act='hs'),
            Bottleneck(80, 80, kernel_size=3, stride=1, expand_size=200, use_se=False, act='hs'),
            Bottleneck(80, 80, kernel_size=3, stride=1, expand_size=184, use_se=False, act='hs'),
            Bottleneck(80, 80, kernel_size=3, stride=1, expand_size=184, use_se=False, act='hs'),
        )

        # 阶段5：有 SE，Hardswish 激活
        self.b6 = nn.Sequential(
            Bottleneck(80, 112, kernel_size=3, stride=1, expand_size=480, use_se=True, act='hs'),
            Bottleneck(112, 112, kernel_size=3, stride=1, expand_size=672, use_se=True, act='hs'),
            Bottleneck(112, 160, kernel_size=5, stride=2, expand_size=672, use_se=True, act='hs'),
            Bottleneck(160, 160, kernel_size=5, stride=1, expand_size=960, use_se=True, act='hs'),
        )

        # 1x1 卷积升维到 960
        self.b7 = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish(inplace=True),
        )

        # 全局平均池化 + 1x1 卷积升维到 1280
        self.b8 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(960, 1280, kernel_size=1),
            nn.Hardswish(inplace=True),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, 10),
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.b7(x)
        x = self.b8(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV3(Bottleneck).to(device)
    print(summary(model, (3, 224, 224)))
