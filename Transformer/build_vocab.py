import json
import os
from collections import Counter

# 特殊标记
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# 特殊标记对应的ID
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def tokenize_en(text):
    """英文分词：按空格切分"""
    return text.strip().split()


def tokenize_zh(text):
    """中文分词：按字切分"""
    return list(text.strip())


def build_vocab(token_counter, min_freq=1):
    """根据词频统计构建词表，返回 word2id 和 id2word"""
    word2id = {}
    # 先添加特殊标记
    for token in SPECIAL_TOKENS:
        word2id[token] = len(word2id)

    # 按词频从高到低添加
    for word, freq in token_counter.most_common():
        if freq >= min_freq:
            word2id[word] = len(word2id)

    id2word = {idx: word for word, idx in word2id.items()}
    return word2id, id2word


def build_vocab_from_file(filepath, tokenize_fn, min_freq=1):
    """读取文件，统计词频并构建词表"""
    counter = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = tokenize_fn(line)
            counter.update(tokens)

    word2id, id2word = build_vocab(counter, min_freq)
    return word2id, id2word


def save_vocab(word2id, filepath):
    """将词表保存为JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    data_dir = './data'

    # 从训练集构建词表
    print("正在构建英文词表...")
    en_word2id, en_id2word = build_vocab_from_file(
        os.path.join(data_dir, 'train.en'), tokenize_en, min_freq=1
    )
    print("英文词表大小: {}".format(len(en_word2id)))

    print("正在构建中文词表...")
    zh_word2id, zh_id2word = build_vocab_from_file(
        os.path.join(data_dir, 'train.zh'), tokenize_zh, min_freq=1
    )
    print("中文词表大小: {}".format(len(zh_word2id)))

    # 保存词表
    save_vocab(en_word2id, os.path.join(data_dir, 'vocab.en.json'))
    save_vocab(zh_word2id, os.path.join(data_dir, 'vocab.zh.json'))

    print("词表已保存到 {} 目录下".format(data_dir))
    print("vocab.en.json - 英文词表")
    print("vocab.zh.json - 中文词表")
