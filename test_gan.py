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
import warnings

warnings.filterwarnings('ignore')

# ====================== 卡通头像GAN训练配置（答辩演示专用） ======================
# 卡通头像数据集路径（仅图片，无标注）
ANIME_FACE_ROOT = r"/workspace/examples/PythonProject1/anime-faces"
# 生成结果保存路径
GENERATE_SAVE_ROOT = r"/workspace/examples/PythonProject1/anime_gan_results"
IMG_SIZE = 128  # 128×128足够生成清晰卡通头像
GAN_BATCH_SIZE = 8
GAN_LR = 1e-4
GAN_Z_DIM = 128
SEED = 42
TRAIN_EPOCHS = 10  # 卡通头像收敛快，10轮足够生成清晰图
GENERATE_NUM = 30  # 生成30张清晰图，满足答辩演示需求

# 随机种子（保证结果可复现）
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 1. 加载卡通头像数据集（仅图片） ======================
def load_anime_face_dataset():
    """加载卡通头像图片路径，过滤无效文件"""
    # 支持的图片格式
    img_exts = ["jpg", "png", "jpeg", "bmp", "webp"]
    img_paths = []
    
    print(f"正在加载卡通头像数据集：{ANIME_FACE_ROOT}")
    # 遍历文件夹下所有图片
    for ext in img_exts:
        paths = glob.glob(os.path.join(ANIME_FACE_ROOT, f"*.{ext}"))
        paths += glob.glob(os.path.join(ANIME_FACE_ROOT, f"**/*.{ext}"), recursive=True)
        img_paths.extend(paths)
    
    # 去重+过滤无效图片
    img_paths = list(set(img_paths))
    valid_paths = []
    for path in tqdm(img_paths, desc="过滤无效图片"):
        try:
            img = cv2.imread(path)
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                valid_paths.append(path)
        except:
            continue
    
    print(f"\n  加载完成！")
    print(f" 有效卡通头像数量：{len(valid_paths)}张")
    if len(valid_paths) < 100:
        print("  警告：有效图片数量较少，但仍可训练出演示效果")
    return valid_paths

# ====================== 2. 卡通头像数据集类 ======================
class AnimeFaceDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取图片并转换为RGB
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 应用数据增强
        augmented = self.transform(image=img)
        return augmented["image"]

# ====================== 3. 轻量GAN模型（适配卡通头像生成） ======================
class AnimeGenerator(nn.Module):
    """轻量生成器，专为卡通头像优化，侧重生成清晰轮廓和风格"""
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
            nn.Tanh()  # 输出归一化到[-1,1]
        )

    def forward(self, x):
        return self.main(x)

class AnimeDiscriminator(nn.Module):
    """轻量判别器，适配卡通头像特征判别"""
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
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x):
        return self.main(x).reshape(-1)

# ====================== 4. 数据增强（温和，保留卡通头像特征） ======================
def get_anime_transform():
    """卡通头像专用数据增强，避免过度变换导致风格丢失"""
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC),  # 高质量缩放
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),  # 温和亮度调整
        A.HorizontalFlip(p=0.5),  # 水平翻转增加多样性
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到[-1,1]
        ToTensorV2()  # 转换为Tensor
    ])

# ====================== 5. 训练GAN生成卡通头像 ======================
def train_anime_gan(img_paths):
    """训练GAN模型，生成卡通头像"""
    # 初始化数据增强和数据加载器
    transform = get_anime_transform()
    dataset = AnimeFaceDataset(img_paths, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=GAN_BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True  # 加速GPU加载
    )

    # 初始化生成器和判别器
    gen = AnimeGenerator(z_dim=GAN_Z_DIM).to(device)
    disc = AnimeDiscriminator().to(device)

    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    opt_gen = optim.Adam(gen.parameters(), lr=GAN_LR, betas=(0.5, 0.999))  # 经典GAN优化器
    opt_disc = optim.Adam(disc.parameters(), lr=GAN_LR, betas=(0.5, 0.999))

    # 开始训练
    print(f"\n  开始训练卡通头像GAN（共{TRAIN_EPOCHS}轮）...")
    for epoch in range(TRAIN_EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS}")
        for batch in loop:
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # --------------------- 训练判别器 ---------------------
            # 生成噪声和假样本
            noise = torch.randn(batch_size, GAN_Z_DIM, 1, 1).to(device)
            fake_imgs = gen(noise)
            
            # 真实样本损失（标签平滑，提升稳定性）
            real_labels = torch.ones_like(disc(batch)) * 0.9
            loss_disc_real = criterion(disc(batch), real_labels)
            
            # 假样本损失
            fake_labels = torch.zeros_like(disc(fake_imgs)) + 0.1
            loss_disc_fake = criterion(disc(fake_imgs.detach()), fake_labels)
            
            # 总判别器损失
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            # 反向传播
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # --------------------- 训练生成器 ---------------------
            loss_gen = criterion(disc(fake_imgs), torch.ones_like(disc(fake_imgs)))
            
            # 反向传播
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # 更新进度条
            loop.set_postfix(
                D_loss=round(loss_disc.item(), 4),
                G_loss=round(loss_gen.item(), 4)
            )
    
    print("\n  卡通头像GAN训练完成！")
    return gen

# ====================== 6. 生成清晰的卡通头像 ======================
def generate_anime_faces(gen):
    """使用训练好的生成器生成卡通头像并保存"""
    # 创建保存目录
    os.makedirs(GENERATE_SAVE_ROOT, exist_ok=True)
    gen.eval()  # 切换到评估模式
    
    print(f"\n 开始生成卡通头像（共{GENERATE_NUM}张）...")
    with torch.no_grad():  # 禁用梯度计算，节省显存
        for i in range(GENERATE_NUM):
            # 生成随机噪声（增加多样性）
            noise = torch.randn(1, GAN_Z_DIM, 1, 1).to(device) * 0.95
            fake_img = gen(noise)
            
            # 反归一化：从[-1,1]转换为[0,255]
            fake_img = fake_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            fake_img = (fake_img + 1) / 2.0
            fake_img = np.clip(fake_img * 255, 0, 255).astype(np.uint8)
            
            # 保存图片（PNG格式，无压缩）
            save_path = os.path.join(GENERATE_SAVE_ROOT, f"anime_face_{i+1:02d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    print(f"\n  卡通头像生成完成！")
    print(f" 生成结果保存路径：{GENERATE_SAVE_ROOT}")
    print(f" 生成的图像尺寸：{IMG_SIZE}×{IMG_SIZE}（高清，可直接用于答辩演示）")

# ====================== 主函数（一键运行） ======================
def main():
    # 1. 加载卡通头像数据集
    img_paths = load_anime_face_dataset()
    if len(img_paths) == 0:
        print("  错误：未找到有效卡通头像图片，请检查数据集路径！")
        return
    
    # 2. 训练GAN模型
    gen = train_anime_gan(img_paths)
    
    # 3. 生成卡通头像
    generate_anime_faces(gen)

if __name__ == "__main__":
    main()