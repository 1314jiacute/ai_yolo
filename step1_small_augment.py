import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import warnings

warnings.filterwarnings('ignore')  # 忽略无关警告

# ====================== 基础配置 ======================
SMALL_DATA_ROOT = r"/workspace/examples/PythonProject1/data_dectect"
AUG_SMALL_ROOT = r"/workspace/examples/PythonProject1/data_dectectenhance"
IMG_SIZE = 640
BATCH_SIZE = 2
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ====================== 数据集类 ======================
class SmallAugDataset(Dataset):
    def __init__(self, img_paths, ann_paths, transform=None):
        self.img_paths = img_paths
        self.ann_paths = ann_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取图片
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None, img_path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取YOLO标注
        ann_path = self.ann_paths[idx]
        bboxes = []
        class_labels = []
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cls = int(parts[0])  # 读取时转整数
                        x, y, w, h = map(float, parts[1:])
                        if w > 1e-6 and h > 1e-6 and 0 <= x <= 1 and 0 <= y <= 1:
                            bboxes.append([x, y, w, h])
                            class_labels.append(cls)
                    except (ValueError, TypeError):
                        continue

        # 应用增强
        if self.transform:
            try:
                augmented = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
                img = augmented["image"]
                bboxes = augmented["bboxes"]
                class_labels = augmented["class_labels"]
                # 优化：强制class_labels为整数列表，避免浮点数
                class_labels = [int(float(c)) for c in class_labels]
            except Exception as e:
                print(f"增强失败 {img_path}: {str(e)}")
                img = None

        return img, bboxes, class_labels, img_path


# 增强流水线（无修改）
def get_small_aug_pipeline():
    train_aug = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.4),
        A.HorizontalFlip(p=0.4),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    val_aug = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    return train_aug, val_aug


# 辅助函数：保存有效标注（核心修复）
def save_valid_annotations(save_path, bboxes, class_labels):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        if not bboxes or not class_labels:
            return
        for b, c in zip(bboxes, class_labels):
            if not hasattr(b, '__len__') or len(b) != 4:
                print(f"跳过无效bbox: {b} (长度: {len(b) if hasattr(b, '__len__') else 'N/A'})")
                continue
            try:
                x, y, w, h = map(float, b)
                if w <= 1e-6 or h <= 1e-6 or x < 0 or x > 1 or y < 0 or y > 1:
                    print(f"跳过范围无效的bbox: {b}")
                    continue
                # 最终保障：强制将类别ID转为整数
                cls_id = int(float(c))
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            except (ValueError, TypeError) as e:
                print(f"解析bbox失败 {b}: {str(e)}")
                continue
# 主函数
def augment_small_dataset():
    # 1. 创建输出目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(AUG_SMALL_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(AUG_SMALL_ROOT, "labels", split), exist_ok=True)

    # 2. 获取增强流水线
    train_aug, val_aug = get_small_aug_pipeline()

    # 3. 处理训练集
    print("开始增强小样本训练集...")
    train_img_paths = sorted(glob.glob(os.path.join(SMALL_DATA_ROOT, "images", "train", "*.jpg")))
    # 处理空数据集情况
    if not train_img_paths:
        print(f"警告：训练集路径下未找到图片 {os.path.join(SMALL_DATA_ROOT, 'images', 'train', '*.jpg')}")
    train_ann_paths = [
        os.path.join(SMALL_DATA_ROOT, "labels", "train", os.path.basename(p).replace(".jpg", ".txt"))
        for p in train_img_paths
    ]

    train_dataset = SmallAugDataset(train_img_paths, train_ann_paths, transform=train_aug)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: x  # 直接返回样本列表
    )

    for batch in tqdm(train_loader, desc="训练集增强"):
        for sample in batch:
            img, bboxes, class_labels, img_path = sample
            if img is None:
                print(f"跳过无效样本: {img_path}")
                continue

            # 保存增强后的图片
            img_name = os.path.basename(img_path)
            save_img_path = os.path.join(AUG_SMALL_ROOT, "images", "train", img_name)
            # Tensor转numpy并恢复色彩空间
            img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(save_img_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            # 保存增强后的标注
            save_ann_path = os.path.join(AUG_SMALL_ROOT, "labels", "train",
                                         os.path.basename(img_path).replace(".jpg", ".txt"))
            save_valid_annotations(save_ann_path, bboxes, class_labels)

    # 4. 处理验证集
    print("开始处理小样本验证集（仅归一化）...")
    val_img_paths = sorted(glob.glob(os.path.join(SMALL_DATA_ROOT, "images", "val", "*.jpg")))
    if not val_img_paths:
        print(f"警告：验证集路径下未找到图片 {os.path.join(SMALL_DATA_ROOT, 'images', 'val', '*.jpg')}")
    val_ann_paths = [
        os.path.join(SMALL_DATA_ROOT, "labels", "val", os.path.basename(p).replace(".jpg", ".txt"))
        for p in val_img_paths
    ]

    val_dataset = SmallAugDataset(val_img_paths, val_ann_paths, transform=val_aug)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: x
    )

    for batch in tqdm(val_loader, desc="验证集归一化"):
        for sample in batch:
            img, bboxes, class_labels, img_path = sample
            if img is None:
                print(f"跳过无效样本: {img_path}")
                continue

            # 保存归一化后的图片
            img_name = os.path.basename(img_path)
            save_img_path = os.path.join(AUG_SMALL_ROOT, "images", "val", img_name)
            img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(save_img_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            # 保存归一化后的标注
            save_ann_path = os.path.join(AUG_SMALL_ROOT, "labels", "val",
                                         os.path.basename(img_path).replace(".jpg", ".txt"))
            save_valid_annotations(save_ann_path, bboxes, class_labels)

    # 5. 生成data.yaml
    generate_aug_yaml()

    # 输出统计
    aug_train_num = len(glob.glob(os.path.join(AUG_SMALL_ROOT, "images", "train", "*.jpg")))
    aug_val_num = len(glob.glob(os.path.join(AUG_SMALL_ROOT, "images", "val", "*.jpg")))
    print(f"\n小样本增强完成！")
    print(f"  增强后训练集：{aug_train_num} 张")
    print(f"  增强后验证集：{aug_val_num} 张")
    print(f"  输出路径：{AUG_SMALL_ROOT}")


def generate_aug_yaml(num_classes=102):
    """生成增强后小样本的data.yaml"""
    original_yaml_path = os.path.join(SMALL_DATA_ROOT, "data.yaml")
    class_names = []
    if os.path.exists(original_yaml_path):
        try:
            with open(original_yaml_path, "r") as f:
                data = yaml.safe_load(f)
                class_names = data.get("names", [])
                # 从原始配置读取类别数，覆盖默认值
                if "nc" in data:
                    num_classes = data["nc"]
        except Exception as e:
            print(f"读取原始data.yaml失败: {e}")
            print("使用默认类别数 102")

    new_data = {
        "path": AUG_SMALL_ROOT,
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": class_names if class_names else [f"class_{i}" for i in range(num_classes)]
    }

    # 保存新的data.yaml
    with open(os.path.join(AUG_SMALL_ROOT, "data.yaml"), "w", encoding='utf-8') as f:
        yaml.dump(new_data, f, default_flow_style=False, allow_unicode=True)


# 运行入口
if __name__ == "__main__":
    # 前置检查：确认输入路径存在
    if not os.path.exists(SMALL_DATA_ROOT):
        print(f"错误：输入路径不存在 {SMALL_DATA_ROOT}")
    else:
        augment_small_dataset()