import torch
import torch.utils.data as Data
from torchvision import transforms
# === 迁移学习时需替换：数据集 ===
from torchvision.datasets import FashionMNIST
from torchvision.transforms.v2.functional import normalize

# === 迁移学习时需替换：模型导入 ===
from model import GoogLeNet,inception
from torchvision.datasets import ImageFolder
from PIL import Image

def test_data_process():
    # === 迁移学习时需替换：数据集 + 输入尺寸 ===
    # 数据集路径
    ROOT_TRAIN = r'data/test'

    # Normalize 做的是“标准化”（把 0~1 的数据转成均值为 0 的分布）。
    normalize = transforms.Normalize([0.486, 0.453, 0.415], [0.262, 0.256, 0.259])
    # 定义数据集处理方法
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

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
    model = GoogLeNet(inception)
    # === 迁移学习时需替换：预训练权重路径 ===
    model.load_state_dict(torch.load('best_dog_cat_model.pth', weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    classes = ['猫', '狗']
    # test_dataloader = test_data_process()
    # test_acc = test_model_process(model, test_dataloader)
    # print(test_acc)


    # 测试网上的图
    image = Image.open('10621685_024402421124_2.jpg')
    normalize = transforms.Normalize([0.486,0.453,0.415],[0.262,0.256, 0.259])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab= torch.argmax(output, dim=1)
        result = pre_lab.item()
        print(classes[result])





