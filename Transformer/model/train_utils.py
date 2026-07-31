import os
import math
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tf_model import make_model


# ============================================================
# 1. 数据集
# ============================================================
class TranslationDataset(Dataset):
    """英中翻译数据集：读取 .en 和 .zh 文件，用 SentencePiece 分词后转为 ID 序列"""

    def __init__(self, en_file, zh_file, sp_en, sp_zh, max_len=128):
        """
        Args:
            en_file: 英文文件路径
            zh_file: 中文文件路径
            sp_en: 英文 SentencePiece 分词器
            sp_zh: 中文 SentencePiece 分词器
            max_len: 最大序列长度，超长截断
        """
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_lines = f.readlines()
        with open(zh_file, 'r', encoding='utf-8') as f:
            self.zh_lines = f.readlines()

        self.sp_en = sp_en
        self.sp_zh = sp_zh
        self.max_len = max_len

    def __len__(self):
        return len(self.en_lines)

    def __getitem__(self, idx):
        # 用 SentencePiece 编码为 ID 序列
        # 中文（目标语言）需要加上 <sos> 和 <eos>，SentencePiece 训练时默认有这些特殊标记
        en_ids = self.sp_en.encode(self.en_lines[idx].strip(), out_type=int)
        zh_ids = self.sp_zh.encode(self.zh_lines[idx].strip(), out_type=int)

        # 截断到 max_len（留出 <sos>/<eos> 的位置）
        en_ids = en_ids[:self.max_len - 2]
        zh_ids = zh_ids[:self.max_len - 2]

        # 目标语言加上 <sos>(1) 和 <eos>(2)
        zh_ids = [1] + zh_ids + [2]  # 1=<sos>, 2=<eos>

        return torch.tensor(en_ids, dtype=torch.long), torch.tensor(zh_ids, dtype=torch.long)


def collate_fn(batch):
    """将一个 batch 的不等长序列进行 padding 对齐"""
    en_batch, zh_batch = zip(*batch)

    # 每个 batch 中最长的序列长度
    en_max_len = max(len(seq) for seq in en_batch)
    zh_max_len = max(len(seq) for seq in zh_batch)

    # 用 0（<pad>）填充
    en_padded = torch.zeros(len(batch), en_max_len, dtype=torch.long)
    zh_padded = torch.zeros(len(batch), zh_max_len, dtype=torch.long)

    for i, (en_seq, zh_seq) in enumerate(batch):
        en_padded[i, :len(en_seq)] = en_seq
        zh_padded[i, :len(zh_seq)] = zh_seq

    return en_padded, zh_padded


# ============================================================
# 2. Mask 生成
# ============================================================
def make_src_mask(src, pad_id=0):
    """生成源语言 mask：pad 位置为 0，非 pad 位置为 1

    返回: [batch_size, 1, 1, src_len]
    """
    # src: [batch_size, src_len]
    # src != pad_id -> [batch_size, src_len]，True/False
    # unsqueeze(1).unsqueeze(2) -> [batch_size, 1, 1, src_len]
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_id=0):
    """生成目标语言 mask：包含 padding mask + 因果 mask（下三角）

    返回: [batch_size, 1, tgt_len, tgt_len]
    """
    # tgt: [batch_size, tgt_len]
    tgt_len = tgt.size(1)

    # 1. padding mask: [batch_size, 1, 1, tgt_len]
    tgt_pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)

    # 2. 因果 mask（下三角矩阵）: [tgt_len, tgt_len]
    #    位置 i 只能看到位置 0~i，未来的位置被遮挡
    tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()

    # 3. 两个 mask 做与运算，合并为一个
    #    [batch_size, 1, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask & tgt_sub_mask

    return tgt_mask


# ============================================================
# 3. 损失函数（Label Smoothing）
# ============================================================
class LabelSmoothingLoss(nn.Module):
    """标签平滑损失：将 one-hot 的硬标签软化为概率分布，防止模型过度自信

    例如 label_smoothing=0.1 时：
      原始标签 [0, 1, 0, 0] -> [0.033, 0.9, 0.033, 0.033]
    """

    def __init__(self, vocab_size, smoothing=0.1, pad_id=0):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_id = pad_id

    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size * seq_len, vocab_size] 模型预测的 log 概率
            target: [batch_size * seq_len] 真实标签
        """
        # 将目标标签转为 one-hot，然后做平滑
        # 原始 one-hot: 正确类别=1，其余=0
        # 平滑后: 正确类别=1-smoothing，其余=smoothing/(vocab_size-1)
        confidence = 1.0 - self.smoothing
        smooth = self.smoothing / (self.vocab_size - 2)  # -2 因为要去掉 <pad> 和正确类别

        # 构造平滑后的目标分布
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smooth)
        true_dist.scatter_(1, target.unsqueeze(1), confidence)
        true_dist[:, self.pad_id] = 0  # pad 位置的损失不算

        # 计算 KL 散度（等价于交叉熵）
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# ============================================================
# 4. 学习率调度（Noam Schedule，论文中的 warmup 策略）
# ============================================================
class NoamScheduler:
    """Noam 学习率调度：先 warmup 上升，再按步数的倒数平方根衰减

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        # warmup 阶段线性增长，之后按步数倒数平方根衰减
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ============================================================
# 5. 数据加载入口
# ============================================================
def get_dataloader(en_file, zh_file, sp_en, sp_zh, batch_size=32, max_len=128, shuffle=True):
    """构建 DataLoader"""
    dataset = TranslationDataset(en_file, zh_file, sp_en, sp_zh, max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0
    )
    return dataloader


def load_tokenizers(tokenizer_dir='./tokenizer'):
    """加载 SentencePiece 分词器"""
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(os.path.join(tokenizer_dir, 'eng.model'))

    sp_zh = spm.SentencePieceProcessor()
    sp_zh.load(os.path.join(tokenizer_dir, 'chn.model'))

    return sp_en, sp_zh
