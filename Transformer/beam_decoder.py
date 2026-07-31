"""束搜索（Beam Search）解码器"""

import torch
import torch.nn.functional as F


class Beam:
    """单条 beam 的状态：记录已生成的 token 序列和累积对数概率"""

    def __init__(self, tokens, score):
        """
        Args:
            tokens: 已生成的 token ID 列表（含 <sos>）
            score: 累积对数概率
        """
        self.tokens = tokens    # list[int]
        self.score = score       # float


def beam_search(model, src, sp_zh, cfg, src_mask=None):
    """对单条句子进行 beam search 解码

    Args:
        model: Transformer 模型
        src: [1, src_len] 源语言 ID 序列
        sp_zh: 中文 SentencePiece 分词器（用于获取 vocab_size）
        cfg: 配置对象
        src_mask: 源语言 mask（可选）

    Returns:
        tokens: 生成的 token ID 列表（不含 <sos>）
    """
    model.eval()
    device = cfg.device
    max_len = cfg.max_decode_len
    beam_size = cfg.beam_size
    length_penalty = cfg.length_penalty
    vocab_size = sp_zh.get_piece_size()

    src = src.to(device)
    if src_mask is None:
        src_mask = (src != cfg.pad_id).unsqueeze(1).unsqueeze(2).to(device)

    # 1. 编码器前向传播，得到 memory
    with torch.no_grad():
        memory = model.encode(src, src_mask)

    # 2. 初始化 beam：以 <sos> 开头
    beams = [Beam(tokens=[cfg.sos_id], score=0.0)]
    completed = []

    # 3. 逐步解码
    for step in range(max_len):
        candidates = []

        for beam in beams:
            # 如果已经生成了 <eos>，移入完成列表
            if beam.tokens[-1] == cfg.eos_id:
                completed.append(beam)
                continue

            # 解码器输入：当前已生成的序列
            tgt = torch.tensor(beam.tokens, dtype=torch.long).unsqueeze(0).to(device)
            tgt_mask = torch.tril(torch.ones(len(beam.tokens), len(beam.tokens))).unsqueeze(0).unsqueeze(0).to(device)

            # 解码器前向传播
            with torch.no_grad():
                out = model.decode(memory, src_mask, tgt, tgt_mask)

            # 取最后一个位置的输出
            log_probs = F.log_softmax(out[:, -1, :], dim=-1)  # [1, vocab_size]

            # 取 beam_size 个最高概率的 token
            topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for i in range(beam_size):
                new_tokens = beam.tokens + [topk_ids[0][i].item()]
                # 长度惩罚：鼓励适当长度的翻译，避免过短
                length = len(new_tokens) - 1  # 去掉 <sos>
                new_score = beam.score + topk_log_probs[0][i].item() / (length ** length_penalty)
                candidates.append(Beam(tokens=new_tokens, score=new_score))

        # 如果所有 beam 都已完成，提前结束
        if not candidates:
            break

        # 选取 beam_size 个得分最高的候选
        candidates.sort(key=lambda b: b.score, reverse=True)
        beams = candidates[:beam_size]

    # 4. 合并已完成的和未完成的，选最优
    all_beams = completed + beams
    all_beams.sort(key=lambda b: b.score, reverse=True)
    best = all_beams[0]

    # 返回去掉 <sos> 的 token 序列
    if best.tokens[0] == cfg.sos_id:
        return best.tokens[1:]
    return best.tokens
