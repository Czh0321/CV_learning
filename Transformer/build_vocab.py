import sentencepiece as spm
import os

# 词表大小
EN_VOCAB_SIZE = 32000
ZH_VOCAB_SIZE = 32000

# 数据路径
data_dir = './data'
tokenizer_dir = './tokenizer'

# 如果tokenizer目录不存在则创建
if not os.path.exists(tokenizer_dir):
    os.makedirs(tokenizer_dir)


def train_en_tokenizer():
    """训练英文BPE分词器"""
    cmd = '--input={} --model_prefix={}/eng --vocab_size={} --character_coverage=1.0 --model_type=bpe'.format(
        os.path.join(data_dir, 'train.en'),
        tokenizer_dir,
        EN_VOCAB_SIZE
    )
    spm.SentencePieceTrainer.Train(cmd)
    print("英文分词器训练完成: eng.model, eng.vocab")


def train_zh_tokenizer():
    """训练中文BPE分词器"""
    cmd = '--input={} --model_prefix={}/chn --vocab_size={} --character_coverage=0.9995 --model_type=bpe'.format(
        os.path.join(data_dir, 'train.zh'),
        tokenizer_dir,
        ZH_VOCAB_SIZE
    )
    spm.SentencePieceTrainer.Train(cmd)
    print("中文分词器训练完成: chn.model, chn.vocab")


if __name__ == '__main__':
    print("开始训练英文分词器...")
    train_en_tokenizer()

    print("开始训练中文分词器...")
    train_zh_tokenizer()

    print("分词器已保存到 {} 目录下".format(tokenizer_dir))
