import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """产生 N 个相同的层（深度拷贝，确保参数不共享）"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embedding(nn.Module):
    """词嵌入层：将词表中的ID映射为 d_model 维的连续向量"""
    # d_model 每个词用多大的d_model构成 d_model 论文中为512
    def __init__(self, vocab, d_model):
        # 初始化方法，传入模型的维度（d_model）和词表大小（vocab）
        super(Embedding, self).__init__()
        # Embedding 层，将词表大小映射为 d_model 维的向量
        self.lut = nn.Embedding(vocab, d_model)
        # 存储模型的维度 d_model
        self.d_model = d_model

    def forward(self, x):
        # 乘以 sqrt(d_model) 是为了缩放，使位置编码和词嵌入在同一数量级
        return self.lut(x) * math.sqrt(self.d_model)


# 位置编码
class PositionalEncoding(nn.Module):
    """位置编码：让模型感知序列中每个词的位置信息"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个 size 为 max_len（设定的句子最大长度）× embedding 维度 的全零矩阵
        # 位置编码矩阵 pe 最多能存多少行
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引列向量（[max_len, 1]），用于与频率项广播相乘，一次性计算出所有位置的角度值
        position = torch.arange(0, max_len).unsqueeze(1)
        # 公式中的分母：10000^(2i/d_model)，用 exp-log 技巧避免大数溢出
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 根据公式求出来值，放 pe 矩阵
        # 偶数列用 sin，奇数列用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 加一个维度，使得 pe 维度变成：1×max_len×d_model
        # 方便与 x（[batch_size, seq_len, d_model]）进行相加
        pe = pe.unsqueeze(0)
        # 将 pe 矩阵以持久 buffer 状态存下（不会作为要训练的参数）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # pe[:, :x.size(1)] 取前 seq_len 行，与 x 相加
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 到此编码器的输入的结束


# 注意力机制(类似模板)  计算q 和 k 之间的点积，并根据该相似度对value进行加权求和
def attention(query, key, value, mask=None, dropout=None):
    """缩放点积注意力：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V"""
    # 将 Q 的最后一个维度值作为 d_k
    d_k = query.size(-1)
    # 将 K 的最后两个维度互换（转置），才能与 Q 相乘，乘完之后除以 根号d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果存在 mask，则将那些需要遮挡的部分替换成一个很大的负数
    # softmax 后这些位置会趋近于 0，相当于"看不到"
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 将 mask 后的 attention 矩阵按照最后一个维度进行 softmax，得到注意力权重
    p_attn = F.softmax(scores, dim=-1)

    # 如果 dropout 参数设置为非空，则进行 dropout 操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵和 V 的乘积，以及注意力权重矩阵
    return torch.matmul(p_attn, value), p_attn


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    """多头注意力：将 Q、K、V 拆成 h 个头分别计算注意力，最后拼接"""
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 保证 d_model 可以被头数 h 整除
        assert d_model % h == 0
        # 每个头的维度
        self.d_k = d_model // h
        self.h = h
        # 4个线性层：WQ, WK, WV, Wo(输出投影，将h个头拼回去)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # mask: [batch_size, 1, 1, seq_len]，方便在所有头之间广播
            mask = mask.unsqueeze(1)
        # Q 的第一个维度值是 batch_size
        nbatches = query.size(0)

        # 1. 分别对 Q、K、V 做线性变换，然后拆分成 h 个头
        #    view: [batch_size, seq_len, d_model] -> [batch_size, seq_len, h, d_k]
        #    transpose: [batch_size, seq_len, h, d_k] -> [batch_size, h, seq_len, d_k]
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2. 调用注意力函数，得到每个头的输出和注意力权重
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. 将 h 个头的结果拼接回来
        #    transpose: [batch_size, h, seq_len, d_k] -> [batch_size, seq_len, h, d_k]
        #    contiguous: 让内存连续（transpose 后不连续，view 前需要调用）
        #    view: [batch_size, seq_len, h, d_k] -> [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 4. 通过最后一个线性层 Wo 做输出变换
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """前馈神经网络：两层线性变换 + ReLU 激活"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 先升维到 d_ff，经过 ReLU 和 Dropout，再降回 d_model
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    """层归一化：对每个样本的特征维度做归一化，稳定训练"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 可学习的缩放参数和偏移参数
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 归一化后做缩放和偏移
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """残差连接 + 层归一化：Add & Norm"""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # 先对子层的输出做 dropout，然后与输入做残差相加，最后做层归一化
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """编码器的一层：自注意力 + 残差归一化，前馈网络 + 残差归一化"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 两个 SublayerConnection：一个用于注意力，一个用于前馈
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 第一个子层：自注意力（Q=K=V=x），然后残差+归一化
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个子层：前馈网络，然后残差+归一化
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """编码器：N 个 EncoderLayer 堆叠"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # 依次通过每一层
        for layer in self.layers:
            x = layer(x, mask)
        # 最后做一次层归一化
        return self.norm(x)


class DecoderLayer(nn.Module):
    """解码器的一层：掩码自注意力 + 交叉注意力 + 前馈网络"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 三个 SublayerConnection
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # m 是编码器的输出（memory）
        m = memory

        # 第一个子层：带掩码的自注意力（Q=K=V=x）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个子层：交叉注意力（Q=x, K=V=编码器输出m）
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 第三个子层：前馈网络
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """解码器：N 个 DecoderLayer 堆叠"""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 依次通过每一层
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # 最后做一次层归一化
        return self.norm(x)


class Generator(nn.Module):
    """输出层：线性映射 + softmax，将 d_model 维向量映射到词表大小的概率分布"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # log_softmax 更数值稳定，配合 NLLLoss 使用
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """完整的 Transformer 模型：编码器 + 解码器 + 嵌入层 + 生成器"""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed        # 源语言（英文）嵌入层
        self.tgt_embed = tgt_embed        # 目标语言（中文）嵌入层
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 编码器处理源语言
        memory = self.encode(src, src_mask)
        # 解码器处理目标语言，结合编码器的输出
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """构建完整 Transformer 模型的工厂函数

    Args:
        src_vocab: 源语言词表大小
        tgt_vocab: 目标语言词表大小
        N: 编码器/解码器的层数（论文中为6）
        d_model: 模型维度（论文中为512）
        d_ff: 前馈网络中间层维度（论文中为2048）
        h: 注意力头数（论文中为8）
        dropout: dropout 概率
    """
    c = copy.deepcopy

    # 实例化各个组件
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # 构建完整模型
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed=nn.Sequential(Embedding(src_vocab, d_model), c(position)),
        tgt_embed=nn.Sequential(Embedding(tgt_vocab, d_model), c(position)),
        generator=Generator(d_model, tgt_vocab)
    )

    # 参数初始化：使用 Xavier 均匀初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


if __name__ == '__main__':
    # 测试模型是否能正常前向传播
    # 源语言词表大小 10000，目标语言词表大小 10000
    model = make_model(src_vocab=10000, tgt_vocab=10000, N=6, d_model=512, d_ff=2048, h=8)

    # 模拟输入：batch_size=2, seq_len=10
    src = torch.randint(1, 10000, (2, 10))
    tgt = torch.randint(1, 10000, (2, 10))

    # mask 全为1（不遮挡）
    src_mask = torch.ones(1, 1, 10)
    tgt_mask = torch.ones(1, 1, 10)

    out = model(src, tgt, src_mask, tgt_mask)
    print("输出形状:", out.shape)  # 期望: [2, 10, 10000]

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("总参数量: {}".format(total_params))
