import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ====================== 单类别训练配置（答辩演示专用） ======================
SMALL_DATA_ROOT = r"/workspace/examples/PythonProject1/data_dectectenhance"
FINAL_AUG_ROOT = r"/workspace/examples/PythonProject1/data_dectectxenhance_single_class"
IMG_SIZE = 128  # 恢复128×128，单类别能学出清晰图
GAN_BATCH_SIZE = 8
GAN_LR = 1e-4
GAN_Z_DIM = 128
SEED = 42
TRAIN_EPOCHS = 20  # 20轮足够单类别收敛
GENERATE_NUM = 30  # 生成30张清晰图，够答辩演示

# 随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 1. 筛选样本数最多的单类别害虫 ======================
def get_max_samples_class():
    """统计所有类别样本数，返回样本最多的类别及其所有样本路径"""
    cls_img_map = defaultdict(list)  # 类别→样本路径
    
    # 遍历所有标注文件
    train_ann_paths = glob.glob(os.path.join(SMALL_DATA_ROOT, "labels", "train", "*.txt"))
    print(f"正在统计所有类别样本数...（共{len(train_ann_paths)}个标注文件）")
    
    for ann_path in tqdm(train_ann_paths):
        img_name_base = os.path.splitext(os.path.basename(ann_path))[0]
        img_dir = os.path.join(SMALL_DATA_ROOT, "images", "train")
        img_path = None
        # 找对应图片
        for ext in ["jpg", "png", "jpeg"]:
            candidate = os.path.join(img_dir, f"{img_name_base}.{ext}")
            if os.path.exists(candidate) and cv2.imread(candidate) is not None:
                img_path = candidate
                break
        if img_path is None:
            continue
        
        # 解析标注，按类别统计
        with open(ann_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    if img_path not in cls_img_map[cls]:
                        cls_img_map[cls].append(img_path)
                except:
                    continue
    
    # 找样本数最多的类别
    max_cls = -1
    max_paths = []
    for cls, paths in cls_img_map.items():
        if len(paths) > len(max_paths):
            max_cls = cls
            max_paths = paths
    
    # 去重+过滤无效图
    max_paths = list(set(max_paths))
    max_paths = [p for p in max_paths if cv2.imread(p) is not None]
    
    print(f"\n  找到样本数最多的类别：类别{max_cls}")
    print(f" 该类别样本数：{len(max_paths)}张（足够GAN学习）")
    return max_cls, max_paths

# ====================== 2. 单类别数据集 ======================
class SingleClassDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=img)
        return augmented["image"]

# ====================== 3. 轻量GAN模型（适配单类别） ======================
class SingleClassGenerator(nn.Module):
    """适配单类别的轻量生成器，重点学纹理/轮廓"""
    def __init__(self, z_dim=128, channels=3):
        super().__init__()
        self.main = nn.Sequential(
            # z_dim×1×1 → 4×4
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4×4 → 8×8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 64×64 → 128×128
            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class SingleClassDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.main = nn.Sequential(
            # 128×128 → 64×64
            nn.Conv2d(channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64×64 → 32×32
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 16×16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 8×8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8 → 4×4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → 1×1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).reshape(-1)

# ====================== 4. 数据增强（温和，保留单类别特征） ======================
def get_single_class_transform():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

# ====================== 5. 单类别训练（重点学清晰特征） ======================
def train_single_class_gan(img_paths):
    transform = get_single_class_transform()
    dataset = SingleClassDataset(img_paths, transform)
    dataloader = DataLoader(dataset, batch_size=GAN_BATCH_SIZE, shuffle=True, drop_last=True)

    # 初始化模型
    gen = SingleClassGenerator(z_dim=GAN_Z_DIM).to(device)
    disc = SingleClassDiscriminator().to(device)

    # 优化器
    criterion = nn.BCELoss()
    opt_gen = optim.Adam(gen.parameters(), lr=GAN_LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=GAN_LR, betas=(0.5, 0.999))

    # 训练（单类别收敛快）
    print(f"\n开始训练样本最多的单类别（共{len(img_paths)}张样本，{TRAIN_EPOCHS}轮）...")
    for epoch in range(TRAIN_EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS}")
        for batch in loop:
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # 训练判别器
            noise = torch.randn(batch_size, GAN_Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            # 真实样本损失
            loss_disc_real = criterion(disc(batch), torch.ones_like(disc(batch)) * 0.9)
            # 假样本损失
            loss_disc_fake = criterion(disc(fake.detach()), torch.zeros_like(disc(fake)) + 0.1)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # 训练生成器
            loss_gen = criterion(disc(fake), torch.ones_like(disc(fake)))
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            loop.set_postfix(D_loss=loss_disc.item(), G_loss=loss_gen.item())
    
    print("\n  单类别GAN训练完成！")
    return gen

# ====================== 6. 生成清晰的单类别害虫图像 ======================
def generate_single_class_images(gen, cls_id):
    gen.eval()
    save_dir = os.path.join(FINAL_AUG_ROOT, f"single_class_{cls_id}_images")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始生成类别{cls_id}的清晰害虫图像（共{GENERATE_NUM}张）...")
    with torch.no_grad():
        for i in range(GENERATE_NUM):
            # 生成多样化噪声
            noise = torch.randn(1, GAN_Z_DIM, 1, 1).to(device) * 0.95
            fake_img = gen(noise)
            
            # 反归一化，转成可显示的图像
            fake_img = fake_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            fake_img = (fake_img + 1) / 2.0  # [-1,1] → [0,1]
            fake_img = np.clip(fake_img * 255, 0, 255).astype(np.uint8)
            
            # 保存高清图像（PNG格式，无压缩）
            save_path = os.path.join(save_dir, f"single_class_pest_{i+1:02d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    print(f"\n  单类别害虫图像生成完成！")
    print(f"  保存路径：{save_dir}")
    print(f" 生成的图像是128×128高清版，可直接用于毕设答辩演示")

# ====================== 主函数（一键运行） ======================
def main():
    # 1. 找样本最多的类别
    max_cls, max_paths = get_max_samples_class()
    if len(max_paths) < 100:
        print("警告：该类别样本数仍较少，但仍可训练出演示图")
    
    # 2. 训练单类别GAN
    gen = train_single_class_gan(max_paths)
    
    # 3. 生成清晰图像
    generate_single_class_images(gen, max_cls)

if __name__ == "__main__":
    main()