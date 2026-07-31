"""模型训练脚本：加载数据 -> 构建模型 -> 训练 -> 保存权重"""

import os
import time
import torch
import sentencepiece as spm
from torch.utils.data import DataLoader

from config import Config
from model.tf_model import make_model
from model.train_utils import (
    TranslationDataset,
    collate_fn,
    make_src_mask,
    make_tgt_mask,
    LabelSmoothingLoss,
    NoamScheduler,
    load_tokenizers,
)


def train_epoch(model, dataloader, criterion, scheduler, optimizer, cfg):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(cfg.device)
        tgt = tgt.to(cfg.device)

        # tgt[:, :-1] 是输入（去掉最后一个 token），tgt[:, 1:] 是目标（去掉第一个 <sos>）
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # 生成 mask
        src_mask = make_src_mask(src, cfg.pad_id).to(cfg.device)
        tgt_mask = make_tgt_mask(tgt_in, cfg.pad_id).to(cfg.device)

        # 前向传播
        out = model(src, tgt_in, src_mask, tgt_mask)

        # 计算损失
        # out: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
        # tgt_out: [batch, seq_len] -> [batch * seq_len]
        loss = criterion(
            out.contiguous().view(-1, out.size(-1)),
            tgt_out.contiguous().view(-1)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 统计
        n_tokens = (tgt_out != cfg.pad_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if (batch_idx + 1) % 50 == 0:
            print('  Batch {:4d} | Loss: {:.4f} | LR: {:.2e}'.format(
                batch_idx + 1, loss.item(), scheduler.optimizer.param_groups[0]['lr']
            ))

    return total_loss / total_tokens


def eval_epoch(model, dataloader, criterion, cfg):
    """验证一个 epoch"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(cfg.device)
            tgt = tgt.to(cfg.device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, cfg.pad_id).to(cfg.device)
            tgt_mask = make_tgt_mask(tgt_in, cfg.pad_id).to(cfg.device)

            out = model(src, tgt_in, src_mask, tgt_mask)

            loss = criterion(
                out.contiguous().view(-1, out.size(-1)),
                tgt_out.contiguous().view(-1)
            )

            n_tokens = (tgt_out != cfg.pad_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / total_tokens


def main():
    cfg = Config()

    # 创建输出目录
    os.makedirs(cfg.weights_dir, exist_ok=True)
    os.makedirs(cfg.run_dir, exist_ok=True)

    # 加载分词器
    print("加载分词器...")
    sp_en, sp_zh = load_tokenizers(cfg.tokenizer_dir)
    en_vocab_size = sp_en.get_piece_size()
    zh_vocab_size = sp_zh.get_piece_size()
    print("英文词表大小: {} | 中文词表大小: {}".format(en_vocab_size, zh_vocab_size))

    # 构建数据集
    print("加载数据...")
    train_dataset = TranslationDataset(cfg.train_en, cfg.train_zh, sp_en, sp_zh, cfg.max_len)
    dev_dataset = TranslationDataset(cfg.dev_en, cfg.dev_zh, sp_en, sp_zh, cfg.max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    print("训练集: {} 条 | 验证集: {} 条".format(len(train_dataset), len(dev_dataset)))

    # 构建模型
    print("构建模型...")
    model = make_model(
        src_vocab=en_vocab_size,
        tgt_vocab=zh_vocab_size,
        N=cfg.n_layer,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        h=cfg.n_head,
        dropout=cfg.dropout
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print("模型参数量: {}".format(total_params))

    # 损失函数 + 优化器 + 学习率调度
    criterion = LabelSmoothingLoss(zh_vocab_size, cfg.label_smoothing, cfg.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, cfg.d_model, cfg.warmup_steps)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        start = time.time()
        print('\nEpoch {}/{}'.format(epoch + 1, cfg.epochs))
        print('-' * 40)

        train_loss = train_epoch(model, train_loader, criterion, scheduler, optimizer, cfg)
        dev_loss = eval_epoch(model, dev_loader, criterion, cfg)

        print('Train Loss: {:.4f} | Dev Loss: {:.4f} | Time: {:.1f}s'.format(
            train_loss, dev_loss, time.time() - start
        ))

        # 保存最优模型
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(cfg.weights_dir, 'best_model.pth'))
            print('>>> 保存最优模型 (loss={:.4f})'.format(best_loss))

        # 每个 epoch 都保存一次
        torch.save(model.state_dict(), os.path.join(cfg.weights_dir, 'last_model.pth'))

    print('\n训练完成！最优 loss: {:.4f}'.format(best_loss))


if __name__ == '__main__':
    main()
