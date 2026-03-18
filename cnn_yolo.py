import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import cv2
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# ====================== 基础配置 ======================
DATA_ROOT = r"/workspace/examples/PythonProject1/data_dectectenhance"
DATA_YAML_PATH = r"/workspace/examples/PythonProject1/data_dectectenhance/data.yaml"
RESULT_ROOT = r"/workspace/examples/PythonProject1/pest_experiment_results"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 超参数优化
TRAIN_EPOCHS = 16
BATCH_SIZE = 24
LEARNING_RATE = 1e-3
IMG_SIZE = 640

# 创建目录
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "classification_models"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "metrics"), exist_ok=True)

# 固定种子
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ====================== 解析类别信息（极简版，直接解决匹配问题） ======================
def parse_yaml_simple():
    """仅解析yaml中的类别名，后续用labels参数强制匹配"""
    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError(f"data.yaml不存在：{DATA_YAML_PATH}")
    
    import yaml as pyyaml
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        yaml_data = pyyaml.safe_load(f)
    
    total_class_names = yaml_data.get("names", [f"class_{i}" for i in range(102)])
    return total_class_names

# ====================== 数据集类（简化版） ======================
class PestClsDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取图片
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 数据增强
        if self.transform:
            img = self.transform(img)
        
        # 标签处理（直接返回原始标签，后续统一过滤）
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

# ====================== 加载数据集（过滤无效类别） ======================
def load_dataset():
    """加载数据集并返回：加载器、实际类别列表、类别数"""
    print("\n====================== 加载分类数据集 ======================")
    
    # 路径定义
    train_img_dir = os.path.join(DATA_ROOT, "images", "train")
    train_label_dir = os.path.join(DATA_ROOT, "labels", "train")
    val_img_dir = os.path.join(DATA_ROOT, "images", "val")
    val_label_dir = os.path.join(DATA_ROOT, "labels", "val")
    
    # 加载训练集
    train_imgs, train_labels = [], []
    train_img_files = glob.glob(os.path.join(train_img_dir, "*"))
    
    for img_path in tqdm(train_img_files, desc="处理训练集"):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(train_label_dir, label_name)
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    parts = lines[0].strip().split()
                    if len(parts) >= 1:
                        try:
                            cls = int(float(parts[0]))
                            # 过滤无效类别（只保留0~101）
                            if 0 <= cls <= 101:
                                train_imgs.append(img_path)
                                train_labels.append(cls)
                        except:
                            continue
    
    # 加载验证集
    val_imgs, val_labels = [], []
    val_img_files = glob.glob(os.path.join(val_img_dir, "*"))
    
    for img_path in tqdm(val_img_files, desc="处理验证集"):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(val_label_dir, label_name)
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    parts = lines[0].strip().split()
                    if len(parts) >= 1:
                        try:
                            cls = int(float(parts[0]))
                            if 0 <= cls <= 101:
                                val_imgs.append(img_path)
                                val_labels.append(cls)
                        except:
                            continue
    
    # 统计实际出现的类别
    all_labels = train_labels + val_labels
    actual_classes = sorted(list(set(all_labels)))
    num_classes = len(actual_classes)
    
    # 构建类别映射（原始ID→连续ID）
    class_map = {old: new for new, old in enumerate(actual_classes)}
    train_labels_mapped = [class_map[cls] for cls in train_labels]
    val_labels_mapped = [class_map[cls] for cls in val_labels]
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 构建DataLoader
    train_dataset = PestClsDataset(train_imgs, train_labels_mapped, train_transform)
    val_dataset = PestClsDataset(val_imgs, val_labels_mapped, val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f" 数据集加载完成！")
    print(f"训练集：{len(train_imgs)}张 | 验证集：{len(val_imgs)}张")
    print(f" 实际类别数：{num_classes} | 类别范围：{actual_classes[:5]}...{actual_classes[-5:]}")
    
    return train_loader, val_loader, val_imgs, actual_classes, num_classes

# ====================== 构建模型 ======================
def build_model(model_name, num_classes):
    if model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "EfficientNet":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型：{model_name}")
    return model.to(DEVICE)

# ====================== 训练模型（核心修复：指定labels参数） ======================
def train_models(total_class_names):
    # 加载数据集
    train_loader, val_loader, val_imgs, actual_classes, num_classes = load_dataset()
    
    # 提取实际类别的名称
    actual_class_names = [total_class_names[cls] for cls in actual_classes]
    # 生成连续的labels参数（0~num_classes-1）
    report_labels = list(range(num_classes))
    
    cnn_models = ["ResNet50", "EfficientNet"]
    cls_metrics = {}
    
    for model_name in cnn_models:
        print(f"\n====================== 训练 {model_name} ======================")
        
        # 构建模型
        model = build_model(model_name, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        best_acc = 0.0
        train_start = time.time()
        final_preds = []
        final_targets = []
        
        # 训练循环
        for epoch in range(TRAIN_EPOCHS):
            # 训练阶段
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_EPOCHS}")
            for imgs, targets in loop:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (preds == targets).sum().item()
                
                loop.set_postfix(loss=round(train_loss/train_total, 4), acc=round(train_correct/train_total, 4))
            
            scheduler.step()
            
            # 验证阶段
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            epoch_preds, epoch_targets = [], []
            
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * imgs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_total += targets.size(0)
                    val_correct += (preds == targets).sum().item()
                    
                    epoch_preds.extend(preds.cpu().numpy())
                    epoch_targets.extend(targets.cpu().numpy())
            
            # 保存最后一轮的预测结果
            if epoch == TRAIN_EPOCHS - 1:
                final_preds = epoch_preds
                final_targets = epoch_targets
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            print(f"Epoch {epoch+1} | 训练准确率：{train_acc:.4f} | 验证准确率：{val_acc:.4f}")
            
            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(RESULT_ROOT, "classification_models", f"{model_name}_best.pt")
                torch.save(model.state_dict(), save_path)
                print(f" 最优模型保存：{save_path}（准确率：{best_acc:.4f}）")
        
        # 推理速度测试
        infer_times = []
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_imgs = []
        for img_path in val_imgs[:100]:
            img = cv2.imread(img_path)
            if img is not None:
                test_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        with torch.no_grad():
            for img in tqdm(test_imgs, desc=f"{model_name} 推理速度测试"):
                img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
                start = time.time()
                model(img_tensor)
                infer_times.append(time.time() - start)
        
        avg_infer_time = np.mean(infer_times) if infer_times else 0
        fps = 1 / avg_infer_time if avg_infer_time > 0 else 0
        
        # ====================== 核心修复：指定labels参数 ======================
        # 强制指定labels参数，确保类别数和名称数完全匹配
        cls_report = classification_report(
            final_targets,
            final_preds,
            labels=report_labels,  # 强制指定连续的标签范围
            target_names=actual_class_names,  # 对应的类别名
            output_dict=True,
            zero_division=0
        )
        
        # 记录指标
        cls_metrics[model_name] = {
            "最优验证准确率": float(best_acc),
            "最终训练准确率": float(train_acc),
            "FPS": fps,
            "平均推理时间(ms)": avg_infer_time * 1000,
            "训练时间(h)": (time.time() - train_start) / 3600,
            "分类报告": cls_report
        }
    
    # 保存指标
    metrics_json = os.path.join(RESULT_ROOT, "metrics", "cnn_classification_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(cls_metrics, f, indent=4, ensure_ascii=False)
    
    # 保存CSV
    metrics_csv = os.path.join(RESULT_ROOT, "metrics", "cnn_classification_metrics.csv")
    pd.DataFrame({
        k: {
            "最优验证准确率": v["最优验证准确率"],
            "FPS": v["FPS"],
            "平均推理时间(ms)": v["平均推理时间(ms)"],
            "训练时间(h)": v["训练时间(h)"]
        } for k, v in cls_metrics.items()
    }).T.round(4).to_csv(metrics_csv, index=True, encoding="utf-8-sig")
    
    print(f"\n 所有CNN模型训练完成！")
    print(f" 指标保存路径：{metrics_json}")
    return cls_metrics

# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        # 1. 解析类别名
        total_class_names = parse_yaml_simple()
        
        # 2. 训练模型
        train_models(total_class_names)
        
        print("\n====================== 训练完成 ======================")
    except Exception as e:
        print(f"\n 训练出错：{str(e)}")
        import traceback
        traceback.print_exc()