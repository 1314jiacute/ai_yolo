# 1. 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 2. 定义超参数
batch_size = 64  # 每次训练的样本数量
learning_rate = 0.001  # 学习率
num_epochs = 10  # 训练的轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先使用GPU

# 3. 数据预处理与加载
# MNIST 数据集的预处理步骤
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化，均值和标准差是MNIST数据集的常见值
])

# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 使用DataLoader进行批量处理
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 4. 定义CNN网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 经过两次池化，图像尺寸从 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加dropout防止过拟合
        self.fc2 = nn.Linear(128, 10)  # 10个输出对应10个数字

    def forward(self, x):
        # 输入: (batch_size, 1, 28, 28)

        # 第一个卷积块
        out = self.conv1(x)  # 输出: (batch_size, 32, 28, 28)
        out = self.relu1(out)
        out = self.pool1(out)  # 输出: (batch_size, 32, 14, 14)

        # 第二个卷积块
        out = self.conv2(out)  # 输出: (batch_size, 64, 14, 14)
        out = self.relu2(out)
        out = self.pool2(out)  # 输出: (batch_size, 64, 7, 7)

        # 展平操作，为全连接层做准备
        out = out.view(out.size(0), -1)  # 输出: (batch_size, 64*7*7)

        # 全连接层
        out = self.fc1(out)  # 输出: (batch_size, 128)
        out = self.relu3(out)
        out = self.dropout(out)
        out = self.fc2(out)  # 输出: (batch_size, 10)

        return out


# 5. 初始化模型、损失函数和优化器
model = CNN().to(device)  # 将模型移动到指定设备（GPU/CPU）
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适用于分类问题
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器


# 6. 训练模型
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # 将模型设置为训练模式
    total_loss = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        # 计算并打印每个epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        total_loss += epoch_loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 计算并返回平均损失
    average_loss = total_loss / num_epochs
    return average_loss


# 调用训练函数
train_loss = train(model, train_loader, criterion, optimizer, num_epochs)


# 7. 测试模型
def test(model, test_loader):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0

    # 不计算梯度，节省计算资源
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算并打印准确率
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


# 调用测试函数
test_accuracy = test(model, test_loader)

# 8. 结果可视化
# 绘制训练损失曲线
plt.plot(range(1, num_epochs + 1), [train_loss / num_epochs] * num_epochs, label='Average Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# 打印最终结果
print(f'\nTraining complete!')
print(f'Average Training Loss: {train_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.2f}%')