import os
import glob
import random
import shutil
import yaml
from tqdm import tqdm

# ====================== 基础配置 ======================
# 原始数据集根路径（包含images、labels、data.yaml）
ORIGINAL_DATA_ROOT = "/workspace/examples/PythonProject1/data_dectect"  # 替换为你的原始数据路径
# 小样本数据集输出路径
SMALL_DATA_ROOT = "/workspace/examples/PythonProject1/data_dectect_smalltest"
# 抽样数量
TRAIN_SAMPLE_NUM = 16000
VAL_SAMPLE_NUM = 2700
# 随机种子（保证结果可复现）
SEED = 42

random.seed(SEED)

# ====================== 核心抽样函数 ======================
def sample_small_dataset():
    # 1. 创建小样本数据集目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(SMALL_DATA_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(SMALL_DATA_ROOT, "labels", split), exist_ok=True)

    # 2. 抽样训练集
    print("抽样训练集...")
    train_img_paths = sorted(glob.glob(os.path.join(ORIGINAL_DATA_ROOT, "images", "train", "*.jpg")))
    train_ann_paths = [
        os.path.join(ORIGINAL_DATA_ROOT, "labels", "train", os.path.basename(p).replace(".jpg", ".txt"))
        for p in train_img_paths
    ]
    # 过滤有标注的样本
    train_valid_pairs = [(img, ann) for img, ann in zip(train_img_paths, train_ann_paths) if os.path.exists(ann)]
    # 随机抽样
    train_sample_pairs = random.sample(train_valid_pairs, TRAIN_SAMPLE_NUM)

    # 复制训练集样本
    for img_path, ann_path in tqdm(train_sample_pairs, desc="复制训练集样本"):
        shutil.copy(img_path, os.path.join(SMALL_DATA_ROOT, "images", "train", os.path.basename(img_path)))
        shutil.copy(ann_path, os.path.join(SMALL_DATA_ROOT, "labels", "train", os.path.basename(ann_path)))

    # 3. 抽样验证集
    print("抽样验证集...")
    val_img_paths = sorted(glob.glob(os.path.join(ORIGINAL_DATA_ROOT, "images", "val", "*.jpg")))
    val_ann_paths = [
        os.path.join(ORIGINAL_DATA_ROOT, "labels", "val", os.path.basename(p).replace(".jpg", ".txt"))
        for p in val_img_paths
    ]
    # 过滤有标注的样本
    val_valid_pairs = [(img, ann) for img, ann in zip(val_img_paths, val_ann_paths) if os.path.exists(ann)]
    # 随机抽样
    val_sample_pairs = random.sample(val_valid_pairs, VAL_SAMPLE_NUM)

    # 复制验证集样本
    for img_path, ann_path in tqdm(val_sample_pairs, desc="复制验证集样本"):
        shutil.copy(img_path, os.path.join(SMALL_DATA_ROOT, "images", "val", os.path.basename(img_path)))
        shutil.copy(ann_path, os.path.join(SMALL_DATA_ROOT, "labels", "val", os.path.basename(ann_path)))

    # 4. 生成小样本数据集的data.yaml
    generate_small_data_yaml()

    # 5. 输出抽样结果
    final_train_count = len(glob.glob(os.path.join(SMALL_DATA_ROOT, "images", "train", "*.jpg")))
    final_val_count = len(glob.glob(os.path.join(SMALL_DATA_ROOT, "images", "val", "*.jpg")))
    print(f"\n小样本抽样完成！")
    print(f"  训练集：{final_train_count} 张（从16192张中抽样）")
    print(f"  验证集：{final_val_count} 张（从2784张中抽样）")
    print(f"小样本数据集保存在：{SMALL_DATA_ROOT}")

def generate_small_data_yaml(num_classes=102):
    """生成小样本数据集的data.yaml"""
    # 读取原始类别名称
    original_yaml_path = os.path.join(ORIGINAL_DATA_ROOT, "data.yaml")
    class_names = []
    if os.path.exists(original_yaml_path):
        with open(original_yaml_path, "r") as f:
            original_data = yaml.safe_load(f)
            class_names = original_data.get("names", [])

    # 生成新的yaml
    new_data = {
        "path": SMALL_DATA_ROOT,
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": class_names
    }

    with open(os.path.join(SMALL_DATA_ROOT, "data.yaml"), "w") as f:
        yaml.dump(new_data, f, default_flow_style=False)

    print(f"小样本data.yaml已生成：{os.path.join(SMALL_DATA_ROOT, 'data.yaml')}")

if __name__ == "__main__":
    sample_small_dataset()