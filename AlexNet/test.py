import torch
import torch.utils.data as Data
from torchvision import transforms
# === 迁移学习时需替换：数据集 ===
from torchvision.datasets import FashionMNIST
# === 迁移学习时需替换：模型导入 ===
from model import AlexNet


def test_data_process():
    # === 迁移学习时需替换：数据集 + 输入尺寸 ===
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(227), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_dataloader

def test_model_process(model, test_dataloader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将模型放到训练数据中
    model = model.to(device)
    test_correct = 0.0
    test_num = 0
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()
            # 输出每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值行标
            pre_lab= torch.argmax(output, dim=1)

            test_correct += torch.sum(pre_lab == test_data_y)
            test_num += test_data_y.size(0)

    test_acc = test_correct.item() / test_num
    print("测试准确率", test_acc)
    return test_acc


if __name__ == '__main__':
    # === 迁移学习时需替换：模型实例化 ===
    model = AlexNet()
    # === 迁移学习时需替换：预训练权重路径 ===
    model.load_state_dict(torch.load('best_AlexNet_model.pth', weights_only=True))

    test_dataloader = test_data_process()
    test_acc = test_model_process(model, test_dataloader)
    print(test_acc)












