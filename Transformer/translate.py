"""模型推理脚本：加载训练好的模型，对输入句子进行翻译"""

import torch
import sentencepiece as spm

from config import Config
from model.tf_model import make_model
from model.train_utils import load_tokenizers
from beam_decoder import beam_search


def translate(model, sp_en, sp_zh, text, cfg):
    """翻译单条句子

    Args:
        model: Transformer 模型
        sp_en: 英文分词器
        sp_zh: 中文分词器
        text: 待翻译的英文句子
        cfg: 配置对象

    Returns:
        翻译后的中文字符串
    """
    # 1. 英文分词 -> ID 序列
    en_ids = sp_en.encode(text.strip(), out_type=int)
    # 截断
    en_ids = en_ids[:cfg.max_len - 2]
    src = torch.tensor([en_ids], dtype=torch.long)

    # 2. beam search 解码
    tokens = beam_search(model, src, sp_zh, cfg)

    # 3. 去掉 <eos>，解码为中文
    if cfg.eos_id in tokens:
        tokens = tokens[:tokens.index(cfg.eos_id)]

    zh_text = sp_zh.decode(tokens)

    return zh_text


def translate_file(model, sp_en, sp_zh, input_file, output_file, cfg):
    """批量翻译文件中的句子

    Args:
        input_file: 输入文件（每行一句英文）
        output_file: 输出文件（每行一句中文翻译）
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            result = translate(model, sp_en, sp_zh, line, cfg)
            f.write(result + '\n')
            print('[{:4d}] {}'.format(i + 1, result))


def main():
    cfg = Config()

    # 加载分词器
    print("加载分词器...")
    sp_en, sp_zh = load_tokenizers(cfg.tokenizer_dir)
    en_vocab_size = sp_en.get_piece_size()
    zh_vocab_size = sp_zh.get_piece_size()

    # 构建模型并加载权重
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

    # 加载训练好的权重
    weight_path = '{}/best_model.pth'.format(cfg.weights_dir)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weight_path))
    else:
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    print("已加载权重: {}".format(weight_path))

    # 交互式翻译
    print("\n" + "=" * 40)
    print("翻译模式（输入 quit 退出）")
    print("=" * 40)

    while True:
        text = input("\n英文输入> ")
        if text.strip().lower() == 'quit':
            break
        if not text.strip():
            continue
        result = translate(model, sp_en, sp_zh, text, cfg)
        print("中文翻译> {}".format(result))


if __name__ == '__main__':
    main()
