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
import shutil
import warnings

warnings.filterwarnings('ignore')

# ====================== 128×128稳定版配置 ======================
SMALL_DATA_ROOT = r"/workspace/examples/PythonProject1/data_dectectenhance"
FINAL_AUG_ROOT = r"/workspace/examples/PythonProject1/data_dectectxenhance_gan"
# 核心：稳定版分辨率128×128
IMG_SIZE = 128  
SMALL_OBJ_THRESH = 0.05
LONGTAIL_THRESH = 20
# 稳定版训练参数（降低学习率+合理批次）
PRETRAIN_EPOCHS = 25  # 全量预训练轮数
FINE_TUNE_EPOCHS = 15 # 小目标微调轮数
GAN_BATCH_SIZE = 8    # 128分辨率适配批次
GAN_LR = 5e-5         # 超低学习率保证稳定
GAN_Z_DIM = 128
SEED = 42

# 随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# 开启CUDA优化（提升训练速度）
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# ====================== 1. 数据集分析（放宽过滤，保留更多样本） ======================
def analyze_dataset():
    cls_count = defaultdict(int)
    small_obj_img_paths = []
    all_pest_img_paths = []

    train_ann_paths = glob.glob(os.path.join(SMALL_DATA_ROOT, "labels", "train", "*.txt"))
    print(f"找到标注文件数量：{len(train_ann_paths)}")

    for ann_path in train_ann_paths:
        img_name_base = os.path.splitext(os.path.basename(ann_path))[0]
        img_dir = os.path.join(SMALL_DATA_ROOT, "images", "train")
        img_paths_candidate = []
        for ext in ["jpg", "png", "jpeg"]:
            candidate = os.path.join(img_dir, f"{img_name_base}.{ext}")
            if os.path.exists(candidate):
                img_paths_candidate.append(candidate)
        
        if not img_paths_candidate:
            print(f"警告：标注文件{ann_path}未找到对应图片")
            continue
        img_path = img_paths_candidate[0]

        # 仅过滤损坏图片，取消尺寸限制（保留更多样本）
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无效图片：{img_path}")
            continue
        
        # 放宽模糊过滤阈值（从50→20），保留小目标样本
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.var(gray) < 20:
            print(f"跳过极度模糊图片：{img_path}")
            continue

        # 所有有效样本加入全量列表
        if img_path not in all_pest_img_paths:
            all_pest_img_paths.append(img_path)

        # 解析标注（保留小目标逻辑）
        with open(ann_path, "r") as f:
            lines = f.readlines()
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:])
                    if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                        cls_count[cls] += 1
                        if w * h < SMALL_OBJ_THRESH and img_path not in small_obj_img_paths:
                            small_obj_img_paths.append(img_path)
                except Exception as e:
                    print(f"跳过无效标注行（{ann_path}第{line_idx+1}行）：{e}")
                    continue

    longtail_cls = [cls for cls, count in cls_count.items() if count < LONGTAIL_THRESH]
    # 去重
    small_obj_img_paths = list(set(small_obj_img_paths))
    all_pest_img_paths = list(set(all_pest_img_paths))

    print(f"\n数据集分析完成：")
    print(f"  总害虫类别数：{len(cls_count)}")
    print(f"  长尾害虫类别数：{len(longtail_cls)} (样本数 < {LONGTAIL_THRESH})")
    print(f"  全量有效害虫样本数：{len(all_pest_img_paths)}")
    print(f"  放宽阈值后小目标害虫样本数：{len(small_obj_img_paths)}")

    # 最终过滤无效样本
    small_obj_img_paths = [p for p in small_obj_img_paths if cv2.imread(p) is not None]
    all_pest_img_paths = [p for p in all_pest_img_paths if cv2.imread(p) is not None]
    return cls_count, longtail_cls, small_obj_img_paths, all_pest_img_paths

# ====================== 2. 128×128轻量稳定版GAN模型 ======================
class ResBlock(nn.Module):
    """轻量残差块：保留细节，避免过拟合"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class Generator(nn.Module):
    """128×128稳定版生成器：简化结构，避免梯度爆炸"""
    def __init__(self, z_dim=128, channels=3):
        super().__init__()
        # 输入: z_dim x 1 x 1 → 4x4
        self.init = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # 4x4 → 8x8
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ResBlock(256)
        )

        # 8x8 → 16x16
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResBlock(128)
        )

        # 16x16 → 32x32
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResBlock(64)
        )

        # 32x32 → 64x64
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ResBlock(32)
        )

        # 64x64 → 128x128（最终稳定分辨率）
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x

class Discriminator(nn.Module):
    """128×128稳定版判别器：匹配生成器结构"""
    def __init__(self, channels=3):
        super().__init__()
        # 128x128 → 64x64
        self.down1 = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 64x64 → 32x32
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64)
        )

        # 32x32 → 16x16
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128)
        )

        # 16x16 → 8x8
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256)
        )

        # 8x8 → 4x4
        self.down5 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 4x4 → 1x1
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.final(x)
        return x

# ====================== 3. 稳定版数据增强（温和策略） ======================
def get_stable_transform():
    """温和增强：保留小目标细节，避免过度变换"""
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=3, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
        # 标准归一化
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ])

# ====================== 4. 稳定版训练策略（严格梯度控制） ======================
def train_gan_stage(img_paths, stage_name, epochs, gen=None, disc=None, is_finetune=False):
    transform = get_stable_transform()
    
    valid_img_paths = [p for p in img_paths if cv2.imread(p) is not None]
    print(f"\n{stage_name} - 有效样本数：{len(valid_img_paths)}")

    if len(valid_img_paths) < 30:
        print(f"{stage_name}：样本数过少，跳过")
        return gen, disc

    dataset = PestDataset(valid_img_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=GAN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    # 初始化模型
    if gen is None or disc is None:
        gen = Generator(z_dim=GAN_Z_DIM).to(device)
        disc = Discriminator().to(device)
        # 稳定权重初始化
        def init_weights(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
            if classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        gen.apply(init_weights)
        disc.apply(init_weights)

    # 稳定版优化器（超低学习率+权重衰减）
    lr = GAN_LR * 0.1 if is_finetune else GAN_LR
    criterion = nn.BCELoss()
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=2e-5)
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=2e-5)

    # 训练（严格梯度控制）
    gen.train()
    disc.train()
    grad_accum_steps = 2  # 梯度累积2步，等效批次=16
    print(f"\n开始{stage_name}（{epochs} epochs，梯度累积{grad_accum_steps}步）...")

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"{stage_name} Epoch {epoch+1}/{epochs}")
        epoch_d_loss = []
        epoch_g_loss = []
        step = 0

        for batch in loop:
            step += 1
            if batch is None or batch.shape[0] == 0:
                continue
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # 训练判别器（梯度累积）
            noise = torch.randn(batch_size, GAN_Z_DIM, 1, 1).to(device)  # 标准噪声，无偏移
            fake = gen(noise)

            disc_real = disc(batch).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real) * 0.9)
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake) + 0.1)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2 / grad_accum_steps

            loss_disc.backward()
            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=0.1)  # 严格梯度裁剪
                opt_disc.step()
                disc.zero_grad()

            # 训练生成器（梯度累积）
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output)) / grad_accum_steps

            loss_gen.backward()
            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=0.1)
                opt_gen.step()
                gen.zero_grad()

            # 记录损失
            if step % grad_accum_steps == 0:
                epoch_d_loss.append(loss_disc.item() * grad_accum_steps)
                epoch_g_loss.append(loss_gen.item() * grad_accum_steps)
                loop.set_postfix(D_loss=np.mean(epoch_d_loss), G_loss=np.mean(epoch_g_loss))

    return gen, disc

# ====================== 5. 128×128稳定生成（无压缩） ======================
def generate_stable_samples(gen, num_generate=100):
    """稳定生成：保证色彩正常，无条纹"""
    gen.eval()
    generated_img_paths = []
    gen_save_dir = os.path.join(FINAL_AUG_ROOT, "gan_generated", "stable_128_pest")
    os.makedirs(gen_save_dir, exist_ok=True)

    print(f"\n开始生成 {num_generate} 张128×128稳定版小目标害虫样本...")
    with torch.no_grad():
        for i in tqdm(range(num_generate), desc="生成稳定样本"):
            # 标准噪声
            noise = torch.randn(1, GAN_Z_DIM, 1, 1).to(device)
            fake_img = gen(noise)

            # 正确反归一化，避免精度误差
            fake_img = fake_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            fake_img = (fake_img + 1) / 2.0
            fake_img = np.clip(fake_img * 255.0, 0, 255).astype(np.uint8)
            
            # 转BGR+无压缩保存
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
            img_name = f"stable_small_pest_gan_{i:04d}.png"
            save_path = os.path.join(gen_save_dir, img_name)
            cv2.imwrite(save_path, fake_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # 轻度压缩，平衡体积和质量
            generated_img_paths.append(save_path)

    return generated_img_paths

# ====================== 6. 复用类和辅助函数（稳定版） ======================
class PestDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            try:
                augmented = self.transform(image=img)
                img = augmented["image"]
            except Exception as e:
                print(f"增强失败 {img_path}: {e}")
                return None

        return img

def filter_low_quality_images(img_paths):
    """极宽松过滤：只删完全无效的图"""
    high_quality_paths = []
    print(f"\n稳定版过滤低质量图像...")
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            os.remove(img_path)
            continue
        
        # 仅过滤纯白板（拉普拉斯方差<3）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        if np.var(laplacian) < 3:
            os.remove(img_path)
            continue
        
        high_quality_paths.append(img_path)
    
    # 兜底：避免空样本
    if len(high_quality_paths) == 0 and len(img_paths) > 0:
        print("警告：过滤后无样本，恢复前30张图")
        for img_path in img_paths[:30]:
            if os.path.exists(img_path):
                high_quality_paths.append(img_path)
    
    print(f"过滤后剩余：{len(high_quality_paths)} / {len(img_paths)}")
    return high_quality_paths

def augment_generated_samples(generated_img_paths):
    if not generated_img_paths:
        return []
    
    aug = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    aug_save_dir = os.path.join(FINAL_AUG_ROOT, "images", "train")
    os.makedirs(aug_save_dir, exist_ok=True)
    augmented_paths = []

    for img_path in tqdm(generated_img_paths, desc="增强稳定样本"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            augmented = aug(image=img)
            aug_img = augmented["image"]
            aug_img_np = aug_img.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            aug_img_np = (aug_img_np * std + mean) * 255
            aug_img_np = np.clip(aug_img_np, 0, 255).astype(np.uint8)

            save_name = f"aug_{os.path.basename(img_path)}"
            save_path = os.path.join(aug_save_dir, save_name)
            cv2.imwrite(save_path, cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            augmented_paths.append(save_path)
        except Exception as e:
            print(f"增强失败 {img_path}: {e}")
            continue

    return augmented_paths

def copy_original_aug_samples():
    original_train_imgs = glob.glob(os.path.join(SMALL_DATA_ROOT, "images", "train", "*.[jp][pn]g"))
    original_train_labels = glob.glob(os.path.join(SMALL_DATA_ROOT, "labels", "train", "*.txt"))

    final_train_img_dir = os.path.join(FINAL_AUG_ROOT, "images", "train")
    final_train_label_dir = os.path.join(FINAL_AUG_ROOT, "labels", "train")
    os.makedirs(final_train_img_dir, exist_ok=True)
    os.makedirs(final_train_label_dir, exist_ok=True)

    print(f"\n复制原有样本...")
    for img_path in tqdm(original_train_imgs):
        shutil.copy(img_path, final_train_img_dir)
    for label_path in tqdm(original_train_labels):
        shutil.copy(label_path, final_train_label_dir)

    try:
        val_img_src = os.path.join(SMALL_DATA_ROOT, "images", "val")
        val_img_dst = os.path.join(FINAL_AUG_ROOT, "images", "val")
        if os.path.exists(val_img_src):
            shutil.copytree(val_img_src, val_img_dst, dirs_exist_ok=True)
        
        val_label_src = os.path.join(SMALL_DATA_ROOT, "labels", "val")
        val_label_dst = os.path.join(FINAL_AUG_ROOT, "labels", "val")
        if os.path.exists(val_label_src):
            shutil.copytree(val_label_src, val_label_dst, dirs_exist_ok=True)
    except Exception as e:
        print(f"复制验证集失败：{e}")

def generate_final_yaml():
    original_yaml_path = os.path.join(SMALL_DATA_ROOT, "data.yaml")
    class_names = []
    num_classes = 0
    if os.path.exists(original_yaml_path):
        try:
            with open(original_yaml_path, "r", encoding='utf-8') as f:
                data = yaml.safe_load(f)
                class_names = data.get("names", [])
                num_classes = data.get("nc", len(class_names))
        except Exception as e:
            print(f"读取yaml失败: {e}")

    new_data = {
        "path": FINAL_AUG_ROOT,
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": class_names if class_names else [f"pest_{i}" for i in range(num_classes)]
    }

    with open(os.path.join(FINAL_AUG_ROOT, "data.yaml"), "w", encoding='utf-8') as f:
        yaml.dump(new_data, f, default_flow_style=False, allow_unicode=True)

# ====================== 主函数 ======================
def main():
    if not os.path.exists(SMALL_DATA_ROOT):
        print(f"错误：路径不存在 {SMALL_DATA_ROOT}")
        return
    
    # 1. 分析数据集
    cls_count, longtail_cls, small_obj_img_paths, all_pest_img_paths = analyze_dataset()

    # 2. 分阶段训练稳定版GAN
    gen, disc = train_gan_stage(all_pest_img_paths, "全量数据预训练（稳定版）", PRETRAIN_EPOCHS)
    gen, disc = train_gan_stage(small_obj_img_paths, "小目标样本微调（稳定版）", FINE_TUNE_EPOCHS, gen, disc, is_finetune=True)

    # 3. 生成稳定样本
    generated_img_paths = generate_stable_samples(gen, num_generate=100)

    # 4. 过滤
    if generated_img_paths:
        generated_img_paths = filter_low_quality_images(generated_img_paths)

    # 5. 增强
    if generated_img_paths:
        augment_generated_samples(generated_img_paths)

    # 6. 合并
    copy_original_aug_samples()

    # 7. 生成配置
    generate_final_yaml()

    # 统计
    final_train_num = len(glob.glob(os.path.join(FINAL_AUG_ROOT, "images", "train", "*.[jp][pn]g")))
    print(f"\n=== 128×128稳定版训练完成 ===")
    print(f"最终训练集样本数：{final_train_num}")
    print(f"生成的稳定版小目标害虫样本数：{len(generated_img_paths)}")
    print(f"输出路径：{FINAL_AUG_ROOT}")

if __name__ == "__main__":
    main()