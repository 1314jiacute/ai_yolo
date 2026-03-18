# 1. 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os

# 2. 定义超参数和配置
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数
latent_dim = 100  # 生成器输入噪声向量的维度
batch_size = 128
num_epochs = 50
learning_rate = 0.0002

# 图像配置
image_size = 28  # MNIST图像尺寸为 28x28
channels = 1  # MNIST为灰度图，通道数为 1

# 创建保存生成图像的目录
sample_dir = 'gan_samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 3. 数据预处理与加载
# MNIST 数据集的预处理步骤
# 注意：我们将图像像素值归一化到 [-1, 1] 范围，这是因为生成器最后使用了 Tanh 激活函数
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
])

# 下载并加载训练集 (GAN 只需要训练集来学习数据分布)
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

# 使用 DataLoader 进行批量处理
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 4. 定义生成器 (Generator)
# 生成器的作用是从随机噪声 z 生成逼真的图像
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 定义一个简单的神经网络，将 latent_dim 维的噪声映射到 1x28x28 的图像
        self.model = nn.Sequential(
            # 输入: (batch_size, latent_dim)
            nn.Linear(latent_dim, 128 * 7 * 7),  # 全连接层，输出维度为 128*7*7
            nn.BatchNorm1d(128 * 7 * 7),  # 批归一化，稳定训练
            nn.ReLU(True),  # ReLU 激活函数

            # 重塑张量以便进行转置卷积操作
            # 输出: (batch_size, 128, 7, 7)
            nn.Unflatten(1, (128, 7, 7)),

            # 转置卷积层 1: (batch_size, 128, 7, 7) -> (batch_size, 64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 转置卷积层 2: (batch_size, 64, 14, 14) -> (batch_size, 1, 28, 28)
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Tanh 激活函数，将输出像素值限制在 [-1, 1] 范围内
        )

    def forward(self, x):
        # 定义前向传播
        img = self.model(x)
        return img


# 5. 定义判别器 (Discriminator)
# 判别器的作用是判断一张图像是真实的 (来自 MNIST) 还是虚假的 (来自生成器)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 定义一个简单的 CNN 分类器
        self.model = nn.Sequential(
            # 输入: (batch_size, 1, 28, 28)
            # 卷积层 1: (batch_size, 1, 28, 28) -> (batch_size, 64, 14, 14)
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU 激活函数

            # 卷积层 2: (batch_size, 64, 14, 14) -> (batch_size, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 展平操作
            # 输出: (batch_size, 128 * 7 * 7)
            nn.Flatten(),

            # 全连接层，输出一个单一的概率值
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()  # Sigmoid 激活函数，将输出限制在 [0, 1] 范围内，表示真假概率
        )

    def forward(self, x):
        # 定义前向传播
        validity = self.model(x)
        return validity


# 6. 初始化模型、损失函数和优化器
# 创建生成器和判别器实例，并移动到设备
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定义损失函数 (二元交叉熵损失)
criterion = nn.BCELoss()

# 定义优化器 (Adam 优化器)
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 7. 训练 GAN
# 用于可视化训练过程的固定噪声
fixed_noise = torch.randn(64, latent_dim, device=device)

print("Starting training...")
for epoch in range(num_epochs):
    # 遍历训练数据
    for i, (real_imgs, _) in enumerate(train_loader):

        # 获取批次大小，并将真实图像移动到设备
        batch_size_actual = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # ---------------------
        #  训练判别器
        # ---------------------
        optimizer_D.zero_grad()

        # 定义真实和虚假图像的标签
        real_labels = torch.ones(batch_size_actual, 1, device=device)
        fake_labels = torch.zeros(batch_size_actual, 1, device=device)

        # 前向传播真实图像
        real_pred = discriminator(real_imgs)
        # 计算真实图像的损失
        d_loss_real = criterion(real_pred, real_labels)

        # 生成虚假图像
        noise = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_imgs = generator(noise)

        # 前向传播虚假图像，.detach() 会切断梯度流向生成器
        fake_pred = discriminator(fake_imgs.detach())
        # 计算虚假图像的损失
        d_loss_fake = criterion(fake_pred, fake_labels)

        # 总损失
        d_loss = d_loss_real + d_loss_fake

        # 反向传播和优化
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  训练生成器
        # ---------------------
        optimizer_G.zero_grad()

        # 我们希望生成器生成的图像能被判别器判断为真实的
        # 所以将虚假图像再次输入判别器，但这次不切断梯度
        fake_pred = discriminator(fake_imgs)
        # 计算生成器的损失 (基于判别器的判断结果)
        g_loss = criterion(fake_pred, real_labels)

        # 反向传播和优化
        g_loss.backward()
        optimizer_G.step()

        # 打印训练信息
        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # 每个 Epoch 结束后，保存一张由固定噪声生成的图像，以观察生成效果的变化
    generator.eval()  # 切换到评估模式
    with torch.no_grad():
        fake_imgs = generator(fixed_noise).detach().cpu()
    save_image(fake_imgs, f'{sample_dir}/epoch_{epoch + 1}.png', nrow=8, normalize=True)
    generator.train()  # 切换回训练模式

# 8. 保存训练好的模型
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
print("Training finished. Models and generated images saved.")

# 9. 可视化最终生成的图像
# 加载一张最后生成的图像并显示
img_path = f'{sample_dir}/epoch_{num_epochs}.png'
img = plt.imread(img_path)
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title(f'Generated MNIST Images after {num_epochs} Epochs')
plt.show()