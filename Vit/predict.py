import torch
from torchvision import transforms
from PIL import Image

# === 迁移学习时需替换：模型导入 ===
from model import VisionTransformer
# ============================================================
# 一、配置区（迁移时只需改这里）
# ============================================================
# 训练好的最优权重路径（由 train.py 保存的 best_vit_model.pth）
# === 迁移学习时需替换：权重路径 ===
MODEL_WEIGHTS = './best_vit_model.pth'

# 要进行预测的单张图片路径
# === 迁移学习时需替换：测试图片路径 ===
IMAGE_PATH = './test_images/sample.jpg'

# 类别列表：顺序必须和训练时 ImageFolder 自动生成的编号一致！
# 训练时 ImageFolder 按文件夹名字母顺序编号，例如 real 在前、fake 在后 -> [real, fake]
# 对应标签：real=0, fake=1
# === 迁移学习时需替换：你的类别名列表 ===
CLASS_NAMES = ['real', 'fake']

# 模型与输入配置（需与训练时完全一致）
IMG_SIZE = 224     # 输入图像尺寸
NUM_CLASS = 2      # 类别数

# 是否是二分类（是则打印概率，无需类别名也能直观理解）
BINARY = True


# ============================================================
# 二、加载模型 + 权重
# ============================================================
def load_model():
    # 实例化模型，结构与训练时保持一致
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=16,
        in_chans=3,
        num_classes=NUM_CLASS,
        qkv_bias=True
    )

    # 加载训练好的权重（weights_only=True 仅加载张量，更安全）
    state_dict = torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    print(f"权重加载成功: {MODEL_WEIGHTS}")
    return model


# ============================================================
# 三、单张图片推理
# ============================================================
def predict(model, image_path):
    # 推理时切到 CPU（若无 GPU）并关闭梯度，节省显存
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 图片预处理：必须和训练时的 val 预处理一致（尤其是 Resize 和 Normalize）
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),   # 缩放到模型要求的尺寸
        transforms.ToTensor(),                     # PIL 图片 [0,255] -> 张量 [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # 用与训练相同的 ImageNet 统计量归一化
    ])

    # 读取图片并预处理，加一个 batch 维度 [1, 3, 224, 224]
    img = Image.open(image_path).convert('RGB')   # 统一转成 3 通道 RGB
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)       # [1, num_classes] 各类别 logits
        # softmax 转成概率（各类别概率之和为 1）
        probs = torch.softmax(output, dim=1).squeeze(0)   # [num_classes]
        # 概率最高的类别索引
        pred_idx = torch.argmax(probs).item()
        pred_prob = probs[pred_idx].item()

    # 打印结果
    print("\n" + "=" * 40)
    print(f"图片: {image_path}")
    print(f"预测类别索引: {pred_idx}")
    if BINARY:
        # 二分类：直接以 0.5 为阈值判定（real 的概率越接近 1 越像真实）
        print(f"  真实(real) 概率: {probs[0].item():.4f}")
        print(f"  AI生成(fake)概率: {probs[1].item():.4f}")
        label = '真实人脸' if pred_idx == 0 else 'AI 生成人脸'
        print(f"==> 判定结果: {label} (置信度 {pred_prob:.2%})")
    else:
        # 多分类：打印各类别对应概率
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {probs[i].item():.4f}")
        print(f"==> 判定结果: {CLASS_NAMES[pred_idx]} (置信度 {pred_prob:.2%})")
    print('=' * 40)


# ============================================================
# 四、主入口
# ============================================================
if __name__ == '__main__':
    model = load_model()          # 加载模型和权重
    predict(model, IMAGE_PATH)    # 对单张图片做预测