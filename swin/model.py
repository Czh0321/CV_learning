import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# ============================ 窗口划分与还原 ============================
# Swin Transformer 的核心思想之一：将图像划分为不重叠的局部窗口，
# 注意力只在窗口内计算，大幅降低了计算量（从全局 O((MN)^2) 降到窗口级 O(M^2 * N)）。
#
# 例如输入 [B, 56, 56, 96]，window_size=7，则每个窗口 7×7=49 个 token，
# 共 (56/7)×(56/7) = 64 个窗口，输出 [B*64, 49, 96]。

def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口。

    参数:
        x: [B, H, W, C]
        window_size: 窗口边长（如 7）
    返回:
        windows: [B * num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape

    # 1. reshape: [B, H, W, C] -> [B, H//ws, ws, W//ws, ws, C]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # 2. permute: -> [B, H//ws, W//ws, ws, ws, C]
    #    把窗口编号维度放到前面，窗口内行列维度放在一起
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # 3. flatten 前三维: [B * num_H * num_W, ws, ws, C]
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    window_partition 的逆操作：把窗口拼回完整特征图。

    参数:
        windows: [B * num_windows, window_size, window_size, C]
        window_size: 窗口边长
        H, W: 原始特征图高宽
    返回:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]

    # 1. reshape: -> [B, H//ws, W//ws, ws, ws, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)

    # 2. permute: -> [B, H//ws, ws, W//ws, ws, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # 3. reshape: -> [B, H, W, C]
    x = x.view(B, H, W, C)
    return x


# ============================ 图像分块嵌入 (Patch Embedding) ============================
# 与 ViT 类似，把图像切成 patch 并做线性投影。
# 区别：Swin 的 patch_size=4（ViT 是 16），所以特征图分辨率更高，保留了更多空间细节。
# 输入: [B, 3, 224, 224] -> 输出: [B, 56, 56, 96]（注意 Swin 用 channels-last 格式）
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)

        # 网格大小：224 / 4 = 56，即 56 × 56 的 patch 网格
        self.patches_resolution = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]  # 3136

        # 用一个卷积同时完成 "分块 + 线性投影"
        # Conv2d(kernel=4, stride=4): [B, 3, 224, 224] -> [B, 96, 56, 56]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                 # [B, 96, 56, 56]
        x = x.flatten(2).transpose(1, 2)  # [B, 56*56, 96] -> 先 flatten 再转置
        x = self.norm(x)                  # LayerNorm
        return x


# ============================ 窗口注意力 (W-MSA / SW-MSA) ============================
# 这是 Swin 的灵魂组件：注意力只在局部窗口内计算，而不是全局。
# 相比 ViT 的全局注意力，窗口注意力将计算复杂度从 O((HW)^2) 降到 O(HW * M^2)，
# 其中 M 是窗口大小（如 7），计算量随图像分辨率呈线性增长而非平方增长。
#
# 额外引入了 "相对位置偏置 (relative position bias)"，让模型感知 patch 之间的空间关系。
# 相对位置偏置是一个可学习的查找表，大小为 (2*ws-1) × (2*ws-1) = 13 × 13 = 169。
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = (window_size, window_size)  # (7, 7)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子

        # ---- 相对位置偏置表 ----
        # 窗口内每个 token 相对其他 token 的偏移范围是 [-(ws-1), ws-1]，
        # 共 2*ws-1 = 13 个可能值，所以查找表大小 = 13 × 13 = 169
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )  # [169, num_heads]

        # ---- 计算每个 token 对的相对位置索引 ----
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, ws, ws]
        coords_flatten = coords.view(2, -1)  # [2, ws*ws]

        # 两两做差: [2, ws*ws, ws*ws] -> 每个 token 对所有 token 的相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]

        # 把相对坐标从 [-(ws-1), ws-1] 平移到 [0, 2*ws-2]（非负，才能做索引）
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1  # 把行坐标乘以列数，展平成一维索引

        # 最终的相对位置索引: [ws*ws, ws*ws]，每个值是查找表中的行索引
        relative_position_index = relative_coords.sum(-1)  # [49, 49]
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV 投影（和 ViT 一样，用一个线性层同时生成 Q、K、V）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 截断正态分布初始化偏置表
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        参数:
            x: [num_windows * B, ws*ws, C]  — 一个窗口内的所有 token
            mask: 注意力掩码，仅 SW-MSA 使用，用于屏蔽跨窗口区域的注意力
        """
        B_, N, C = x.shape  # B_ = B * num_windows, N = ws * ws = 49

        # 1. 生成 Q、K、V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 各: [B_, num_heads, N, head_dim]

        # 2. 计算注意力分数: Q × K^T / scale
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_, num_heads, N, N]

        # 3. 加上相对位置偏置
        #    relative_position_index: [49, 49] -> 从偏置表中索引出对应的偏置值
        #    -> [49, 49, num_heads] -> [1, num_heads, 49, 49]（广播到所有 batch 和 head）
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # 4. 如果有 mask（SW-MSA 使用），将需要屏蔽的位置设为 -inf
        if mask is not None:
            nW = mask.shape[0]  # 窗口数
            # attn: [B, nW, num_heads, N, N] -> 加上 mask [1, nW, 1, N, N]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # 5. Softmax 归一化 + Dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 6. 用注意力权重对 V 加权求和
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # [B_, N, C]

        # 7. 线性投影 + Dropout
        out = self.proj_drop(self.proj(out))
        return out


# ============================ 前馈神经网络 (MLP) ============================
# 与 ViT 完全一样：Linear -> GELU -> Dropout -> Linear -> Dropout
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


# ============================ Swin Transformer Block ============================
# 一个 Block 包含两种注意力之一：
#   - W-MSA  (Window Multi-Head Self-Attention):  在固定窗口内做注意力
#   - SW-MSA (Shifted Window MSA): 窗口偏移后再做注意力，让信息跨窗口流动
#
# 两种 Block 交替使用：第 0, 2, 4... 层用 W-MSA，第 1, 3, 5... 层用 SW-MSA。
# SW-MSA 通过将特征图循环移位 (shift_size = ws/2)，让原本不在一起的 patch
# 进入同一个窗口，从而实现跨窗口的信息交互。
#
# 结构（Pre-Norm）:
#   x = x + WMSA/MSA(LayerNorm(x))
#   x = x + MLP(LayerNorm(x))
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  # 0 表示 W-MSA，>0 表示 SW-MSA
        self.mlp_ratio = mlp_ratio

        # shift_size 必须在 [0, window_size) 范围内
        assert 0 <= self.shift_size < self.window_size, "shift_size 必须在 [0, window_size) 范围内"

        # --- 自注意力部分 ---
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        # --- 前馈网络部分 ---
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )

    def forward(self, x, attn_mask=None):
        """
        参数:
            x: [B, L, C]  (L = H * W)
            attn_mask: SW-MSA 的注意力掩码 [num_windows, ws*ws, ws*ws]
        """
        B, L, C = x.shape
        H = W = int(L ** 0.5)  # 从序列长度恢复 H, W（假设方形特征图）

        shortcut = x  # 残差连接的捷径

        # 1. reshape 成 [B, H, W, C] 方便做空间操作
        x = x.view(B, H, W, C)

        # 2. 如果是 SW-MSA，做循环移位（cyclic shift）
        if self.shift_size > 0:
            # 向左上方向循环移位 shift_size 个像素
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 3. 窗口划分: [B, H, W, C] -> [B*num_windows, ws*ws, C]
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 4. 窗口内注意力计算
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 5. 窗口还原: [B*num_windows, ws*ws, C] -> [B, H, W, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 6. 如果做了循环移位，再移回来
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 7. reshape 回 [B, L, C] + 残差 + LayerNorm
        x = x.view(B, L, C)
        x = shortcut + self.norm1(x)

        # 8. MLP + 残差 + LayerNorm
        x = x + self.norm2(self.mlp(x))
        return x


# ============================ 补丁合并层 (Patch Merging) ============================
# Swin 的层级结构核心：在 stage 之间做 2× 下采样。
# 方式：取相邻 2×2 个 patch 拼接，通道数从 C 变 4C，再用线性层降到 2C。
# 输入: [B, H, W, C] -> 输出: [B, H/2, W/2, 2C]
#
# 类似 ViT 的 patch 太粗糙，Swin 通过逐层合并保留多尺度信息，
# 类似 CNN 中的 pooling / stride 操作。
class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 4C -> 2C
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        参数:
            x: [B, L, C]  (L = H * W)
        返回:
            [B, H/2 * W/2, 2C]
        """
        B, L, C = x.shape
        H = W = int(L ** 0.5)

        x = x.view(B, H, W, C)

        # 取 2×2 相邻区域：左上、右上、左下、右下
        # 每隔一个像素取，共 4 组，每组大小 [B, H/2, W/2, C]
        x0 = x[:, 0::2, 0::2, :]  # 左上
        x1 = x[:, 1::2, 0::2, :]  # 右上
        x2 = x[:, 0::2, 1::2, :]  # 左下
        x3 = x[:, 1::2, 1::2, :]  # 右下

        # 拼接: [B, H/2, W/2, 4C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)  # [B, (H/2)*(W/2), 4C]

        # LayerNorm + 线性降维: 4C -> 2C
        x = self.norm(x)
        x = self.reduction(x)  # [B, (H/2)*(W/2), 2C]
        return x


# ============================ 基础层 (BasicLayer / Stage) ============================
# 一个 Stage = 若干个 SwinTransformerBlock + 一个 PatchMerging（最后一个 stage 不合并）
#
# Swin-T 的 4 个 Stage:
#   Stage 1: 2 个 Block (dim=96,  heads=3,  分辨率 56×56)
#   Stage 2: 2 个 Block (dim=192, heads=6,  分辨率 28×28)
#   Stage 3: 6 个 Block (dim=384, heads=12, 分辨率 14×14)
#   Stage 4: 2 个 Block (dim=768, heads=24, 分辨率  7×7)
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, downsample=None):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        # 构建 blocks：偶数索引用 W-MSA (shift=0)，奇数索引用 SW-MSA (shift=ws//2)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # 交替 W-MSA / SW-MSA
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # 下采样层（PatchMerging），最后一个 stage 通常为 None
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ============================ Swin Transformer ============================
# 完整的 Swin Transformer 模型，流程如下：
#
#   1. Patch Embedding: [B, 3, 224, 224] -> [B, 3136, 96]  (56×56 个 patch)
#   2. 加绝对位置嵌入
#   3. Stage 1: 2 Blocks + Merging -> [B, 784, 192]   (28×28)
#   4. Stage 2: 2 Blocks + Merging -> [B, 196, 384]   (14×14)
#   5. Stage 3: 6 Blocks + Merging -> [B, 49, 768]     (7×7)
#   6. Stage 4: 2 Blocks (无 Merging) -> [B, 49, 768]
#   7. 全局平均池化 + LayerNorm + 分类头 -> [B, num_classes]
#
# 对比 ViT:
#   - ViT 是单尺度（只有一种分辨率），Swin 是多尺度层级结构
#   - ViT 全局注意力（O(N^2)），Swin 窗口注意力（O(N)），更高效
#   - Swin 通过窗口移位实现跨窗口交互，不需要全局注意力也能建模长距离依赖
class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)       # stage 数量 = 4
        self.embed_dim = embed_dim
        self.num_features = embed_dim * (2 ** (self.num_layers - 1))  # 最后一个 stage 的维度 = 96 * 8 = 768

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches  # 3136

        # 2. 绝对位置嵌入（可学习），与 ViT 类似
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. 构建 4 个 Stage
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),              # 每个 stage 维度翻倍: 96 -> 192 -> 384 -> 768
                depth=depths[i_layer],                           # 每个 stage 的 block 数: [2, 2, 6, 2]
                num_heads=num_heads[i_layer],                   # 每个 stage 的注意力头数: [3, 6, 12, 24]
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,                                  # 每个 block 使用相同的 dropout 概率
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                # 最后一个 stage 不做 PatchMerging
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        # 4. 最终的 LayerNorm
        self.norm = norm_layer(self.num_features)

        # 5. 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化：线性层和归一化层用截断正态分布"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 1. Patch Embedding: [B, 3, 224, 224] -> [B, 3136, 96]
        x = self.patch_embed(x)

        # 2. Dropout
        x = self.pos_drop(x)

        # 3. 依次通过 4 个 Stage
        for layer in self.layers:
            x = layer(x)  # 每个 Stage 后分辨率减半、通道翻倍

        # 4. LayerNorm
        x = self.norm(x)  # [B, 49, 768]

        # 5. 全局平均池化: 对序列维度取平均 -> [B, 768]
        x = x.mean(dim=1)

        # 6. 分类头
        x = self.head(x)  # [B, num_classes]
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Swin-T (Tiny) 配置 —— 最小版本，适合学习和小数据集训练
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True
    ).to(device)
    print(summary(model, (3, 224, 224)))
