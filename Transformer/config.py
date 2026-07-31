"""全局配置：模型结构、训练参数、推理参数"""


class Config:
    # =====================
    # 数据路径
    # =====================
    data_dir = './data'
    tokenizer_dir = './tokenizer'
    weights_dir = './weights'
    run_dir = './run'

    # 训练集 / 验证集 / 测试集
    train_en = './data/train.en'
    train_zh = './data/train.zh'
    dev_en = './data/dev.en'
    dev_zh = './data/dev.zh'
    test_en = './data/test.en'
    test_zh = './data/test.zh'

    # =====================
    # 模型结构（论文默认值）
    # =====================
    d_model = 512          # 模型维度
    d_ff = 2048            # 前馈网络中间层维度
    n_layer = 6            # 编码器/解码器层数
    n_head = 8             # 注意力头数
    dropout = 0.1          # dropout 概率

    # =====================
    # 训练参数
    # =====================
    batch_size = 32        # 批次大小
    max_len = 128          # 最大序列长度
    epochs = 20            # 训练轮数
    warmup_steps = 4000    # Noam 调度的 warmup 步数
    label_smoothing = 0.1  # 标签平滑系数
    pad_id = 0             # <pad> 的 ID
    sos_id = 1             # <sos> 的 ID
    eos_id = 2             # <eos> 的 ID

    # =====================
    # 推理参数
    # =====================
    beam_size = 5          # beam search 的束宽
    max_decode_len = 128   # 解码时最大生成长度
    length_penalty = 0.6   # 长度惩罚系数

    # =====================
    # 设备
    # =====================
    # 自动选择 GPU 或 CPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
