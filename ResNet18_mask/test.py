import torch
import torch.utils.data as Data
from torchvision import transforms
# === 迁移学习时需替换：数据集 ===
from torchvision.datasets import FashionMNIST
# === 迁移学习时需替换：模型导入 ===
from model import ResNet18,Residual
from PIL import Image


if __name__ == '__main__':
    # === 迁移学习时需替换：模型实例化 ===
    model = ResNet18(Residual)
    # === 迁移学习时需替换：预训练权重路径 ===
    model.load_state_dict(torch.load('best_resnet18mask_model.pth', weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    classes = ['戴口罩','没带口罩']

    image = Image.open('VCG211564978077.jpg')
    normalized = transforms.Normalize(mean=[0.597, 0.554, 0.539], std=[0.317 ,0.318 ,0.321 ])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalized])
    image = test_transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
        print(classes[result])













