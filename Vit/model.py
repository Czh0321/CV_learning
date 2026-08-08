import torch
from torch import nn
from torchsummary import summary

# ============================ 图像分块嵌入 ============================
# 将输入图像切分为固定大小的 patch，并通过卷积将其映射为嵌入向量。
# 这是 ViT 的第一步：把 "图像" 转换成 "序列"，类似于 NLP 中的词嵌入。
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()

        # 统一成（H, W） 和 （P, P）
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        # 计算网格大小：224 // 16 = 14，即 14 × 14 的网格
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        # 切成块的数量：14 × 14 = 196 个 patch
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 用一个卷积层来等价完成 "分块 + 线性投影"
        # Conv2d(kernel=16, stride=16) 会将每个 16×16 的 patch 投影为 1 个 embed_dim 维向量
        # 输出形状：[B, embed_dim, 14, 14]  -> 展平后 [B, 196, embed_dim]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 可选的归一化层，对每个 patch 的嵌入向量做 LayerNorm
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # 强制要求输入图像的长宽，必须严格等于模型初始化时设定的长宽
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸 ({H}×{W}) 与设定尺寸 ({self.img_size[0]}×{self.img_size[1]}) 不匹配"

        # 1. 卷积投影: [B, 3, 224, 224] -> [B, 768, 14, 14]
        # 2. flatten(2): 将 H、W 两个维度展平 -> [B, 768, 196]
        # 3. transpose(1, 2): 交换通道和序列维度 -> [B, 196, 768]，得到序列形式
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# ============================ 多头自注意力 (MSA) ============================
# 核心组件：让序列中的每个 patch 都能 "看到" 其他所有 patch，
# 从而捕捉全局依赖关系。这是 Transformer 的灵魂。
class Attention(nn.Module):
    def __init__(self,
                 dim, # 768
                 num_heads=12,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super(Attention, self).__init__()

        self.num_heads = num_heads        # 注意力头数
        self.head_dim = dim // num_heads  # 每个头的维度: 768 / 12 = 64
        self.scale = self.head_dim ** -0.5  # 缩放因子，防止点积过大导致梯度消失

        # 用一个线性层同时生成 Q、K、V（权重共享，更高效）
        # 输入 dim -> 输出 dim * 3（Q、K、V 各一份）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)  # 对注意力权重做 dropout

        # 将多头注意力的输出投影回原始维度
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)   # 对最终输出做 dropout

    def forward(self, x):
        # x 形状: [B, N, dim]，N 是序列长度（196 + 1 个 class token = 197）
        B, N, C = x.shape

        # 1. 生成 Q、K、V: [B, N, dim] -> [B, N, 3*dim] -> [B, N, 3, num_heads, head_dim]
        #    再 reshape 为 [3, B, num_heads, N, head_dim]，方便后续按头处理
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各为 [B, num_heads, N, head_dim]

        # 2. 计算注意力分数: Q × K^T / scale
        #    [B, num_heads, N, head_dim] × [B, num_heads, head_dim, N] -> [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 3. Softmax 归一化，得到注意力权重（每个 patch 对其他 patch 的关注度）
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 4. 用注意力权重对 V 加权求和: [B, num_heads, N, N] × [B, num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
        out = attn @ v

        # 5. 将多头拼接回去: [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, dim]
        out = out.transpose(1, 2).reshape(B, N, C)

        # 6. 线性投影 + dropout
        out = self.proj_drop(self.proj(out))
        return out


# ============================ 前馈神经网络 (MLP) ============================
# 每个 Transformer Block 中，在自注意力之后接一个两层 MLP。
# 结构：Linear -> GELU -> Dropout -> Linear -> Dropout
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0.):
        super(MLP, self).__init__()

        out_features = out_features or in_features  # 默认输出维度 = 输入维度

        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层：升维（×4）
        self.act = nn.GELU()                                 # GELU 激活函数（比 ReLU 更平滑）
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层：降回原始维度
        self.drop = nn.Dropout(drop)                         # dropout 防止过拟合

    def forward(self, x):
        x = self.fc1(x)     # 升维
        x = self.act(x)     # 激活
        x = self.drop(x)    # dropout
        x = self.fc2(x)     # 降维
        x = self.drop(x)    # dropout
        return x


# ============================ Transformer Encoder Block ============================
# 一个完整的编码器块，结构为（Pre-Norm 架构）：
#   x = x + Attention(LayerNorm(x))        # 多头自注意力 + 残差连接
#   x = x + MLP(LayerNorm(x))              # 前馈网络 + 残差连接
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super(Block, self).__init__()

        # --- 自注意力部分 ---
        self.norm1 = norm_layer(dim)  # LayerNorm（在注意力之前，即 Pre-Norm）
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        # --- 前馈网络部分 ---
        self.norm2 = norm_layer(dim)  # LayerNorm（在 MLP 之前）
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),  # 隐藏层维度 = dim × 4
            drop=drop
        )

    def forward(self, x):
        # 残差连接1：输入 + Attention(LayerNorm(输入))
        x = x + self.attn(self.norm1(x))
        # 残差连接2：输入 + MLP(LayerNorm(输入))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================ Vision Transformer (ViT) ============================
# 完整的 ViT 模型，流程如下：
#   1. 图像分块嵌入 (Patch Embedding)
#   2. 拼接 Class Token（用于最终分类）
#   3. 加上位置编码 (Positional Encoding)
#   4. 经过 L 层 Transformer Encoder Block
#   5. 取 Class Token 的输出，经过 LayerNorm + Linear 得到分类结果
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. 图像分块嵌入层
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches  # 196

        # 2. Class Token：一个可学习的向量，拼接到序列最前面，用于汇总全局信息做分类
        #    形状: [1, 1, 768]，训练时会自动广播到 batch 中每个样本
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. 位置编码：给每个 patch 加上位置信息（因为自注意力本身没有位置感 知能力）
        #    形状: [1, 197, 768]（196 个 patch + 1 个 class token）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 4. Transformer Encoder：堆叠 depth 个 Block
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        # 5. 最终的 LayerNorm
        self.norm = norm_layer(embed_dim)

        # 6. 分类头：Linear(768 -> num_classes)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]

        # --- Step 1: 图像分块嵌入 ---
        # [B, 3, 224, 224] -> [B, 196, 768]
        x = self.patch_embed(x)

        # --- Step 2: 拼接 Class Token ---
        # cls_token: [1, 1, 768] -> [B, 1, 768]  Batch 维度（第0维）上复制 B 份
        # 拼接后: [B, 196, 768] -> [B,   197, 768]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # --- Step 3: 加上位置编码 ---
        # [B, 197, 768] + [1, 197, 768] -> [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)

        # --- Step 4: 经过 Transformer Encoder ---
        # [B, 197, 768] -> [B, 197, 768]
        x = self.blocks(x)

        # --- Step 5: LayerNorm ---
        x = self.norm(x)

        # --- Step 6: 取 Class Token 的输出做分类 ---
        # x[:, 0] 取序列第 0 个位置（即 class token）: [B, 768]
        x = self.head(x[:, 0])  # [B, num_classes]
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True
    ).to(device)
    print(summary(model, (3, 224, 224)))
