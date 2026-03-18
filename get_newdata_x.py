
import os
import glob
import shutil
import numpy as np
from sklearn.cluster import KMeans

# ====================== 核心配置（无需修改，严格匹配你的路径） ======================
# 原始数据集根目录
ORIGINAL_ROOT = r"/workspace/examples/PythonProject1/data_dectect_t2"
# 去冗余后的新数据集根目录（最终训练用）
FINAL_ROOT = r"/workspace/examples/PythonProject1/data_dectect_t3"
# 关键配置
ID0_TARGET_COUNT = 1200  # ID0最终保留的核心样本数
ID0_CLASS_ID = 0  # 需去冗余的大类别ID
# 保留的根目录文件（按你的截图）
ROOT_FILES = ["data.yaml", "class_mapping.log"]


# ====================== 步骤1：复刻完整目录结构 ======================
def build_final_structure():
    """完全复刻原数据集的目录结构，确保和目标格式一致"""
    # 定义需要创建的目录
    dirs_to_create = [
        os.path.join(FINAL_ROOT, "images", "train"),
        os.path.join(FINAL_ROOT, "images", "val"),
        os.path.join(FINAL_ROOT, "labels", "train"),
        os.path.join(FINAL_ROOT, "labels", "val")
    ]

    # 创建目录（清空旧数据，保证纯净）
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        # 清空目录内旧文件
        for file in glob.glob(os.path.join(dir_path, "*")):
            if os.path.isfile(file):
                os.remove(file)

    # 复制根目录关键文件（data.yaml + 日志）
    print("📄 复制根目录配置文件...")
    for file_name in ROOT_FILES:
        src = os.path.join(ORIGINAL_ROOT, file_name)
        dst = os.path.join(FINAL_ROOT, file_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"    复制：{file_name}")
        else:
            print(f"    未找到：{src}，跳过")

    print(f" 目录结构创建完成，完美匹配目标格式！")
    print(f"   最终数据集路径：{FINAL_ROOT}")


# ====================== 步骤2：复制val集（完全保留，不做任何处理） ======================
def copy_val_set():
    """验证集原样复制，保证评估的客观性"""
    print("\n📋 复制验证集（完全保留原数据）...")
    # 复制val图片
    val_img_src = os.path.join(ORIGINAL_ROOT, "images", "val")
    val_img_dst = os.path.join(FINAL_ROOT, "images", "val")
    for img in glob.glob(os.path.join(val_img_src, "*.[jp][pn]g")):
        shutil.copy2(img, val_img_dst)

    # 复制val标注
    val_label_src = os.path.join(ORIGINAL_ROOT, "labels", "val")
    val_label_dst = os.path.join(FINAL_ROOT, "labels", "val")
    for txt in glob.glob(os.path.join(val_label_src, "*.txt")):
        shutil.copy2(txt, val_label_dst)

    print(f" 验证集复制完成：图片{len(os.listdir(val_img_dst))}张，标注{len(os.listdir(val_label_dst))}个")


# ====================== 步骤3：筛选ID0的核心train样本（聚类去冗余） ======================
def select_id0_core_samples():
    """提取ID0特征并聚类，筛选核心样本，避免冗余"""
    train_label_src = os.path.join(ORIGINAL_ROOT, "labels", "train")
    # 存储：包含ID0的标注文件 + 对应特征
    id0_files = []
    id0_features = []
    file_feature_map = {}  # 标注文件 → 该文件内的ID0特征列表

    # 遍历所有train标注，提取ID0信息
    all_train_labels = glob.glob(os.path.join(train_label_src, "*.txt"))
    for txt in all_train_labels:
        with open(txt, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        file_id0_feats = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            if int(parts[0]) == ID0_CLASS_ID:
                # 提取YOLO归一化特征：x, y, w, h
                feats = list(map(float, parts[1:5]))
                file_id0_feats.append(feats)

        if file_id0_feats:
            id0_files.append(txt)
            id0_features.extend(file_id0_feats)
            file_feature_map[txt] = file_id0_feats

    # 聚类筛选核心样本
    id0_features = np.array(id0_features)
    if len(id0_features) <= ID0_TARGET_COUNT:
        # 样本不足，直接保留所有含ID0的文件
        selected_id0_files = list(set(id0_files))
        print(f"\n ID0样本不足，保留所有{len(selected_id0_files)}个含ID0的文件")
    else:
        # KMeans聚类：按特征分布选核心样本（n_init=10避免警告）
        kmeans = KMeans(n_clusters=ID0_TARGET_COUNT, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(id0_features)

        # 每个聚类选距离中心最近的样本，映射到标注文件
        selected_indices = []
        for c in range(ID0_TARGET_COUNT):
            cluster_idx = np.where(cluster_labels == c)[0]
            if len(cluster_idx) == 0:
                continue
            # 选距离聚类中心最近的特征
            center = kmeans.cluster_centers_[c]
            distances = [np.linalg.norm(id0_features[i] - center) for i in cluster_idx]
            selected_indices.append(cluster_idx[np.argmin(distances)])

        # 反向映射到标注文件
        selected_id0_files = []
        feat_idx = 0
        for txt in id0_files:
            feat_num = len(file_feature_map[txt])
            for idx in selected_indices:
                if feat_idx <= idx < feat_idx + feat_num:
                    selected_id0_files.append(txt)
                    break
            feat_idx += feat_num

        # 去重（同一文件可能被多个聚类选中）
        selected_id0_files = list(set(selected_id0_files))
        print(f"\n ID0聚类筛选完成：保留{len(selected_id0_files)}个核心文件（目标1200）")

    return selected_id0_files


# ====================== 步骤4：复制train集（ID0核心样本 + 其他类别全量） ======================
def copy_train_set(selected_id0_files):
    """复制训练集：ID0选核心，其他类别全保留"""
    train_img_src = os.path.join(ORIGINAL_ROOT, "images", "train")
    train_label_src = os.path.join(ORIGINAL_ROOT, "labels", "train")
    train_img_dst = os.path.join(FINAL_ROOT, "images", "train")
    train_label_dst = os.path.join(FINAL_ROOT, "labels", "train")

    print("\n 复制训练集（ID0核心 + 其他全量）...")
    # 第一步：复制选中的ID0核心样本
    id0_img_count = 0
    for txt in selected_id0_files:
        # 复制标注
        txt_name = os.path.basename(txt)
        shutil.copy2(txt, os.path.join(train_label_dst, txt_name))
        # 复制对应图片（兼容jpg/png）
        img_base = txt_name.replace(".txt", "")
        for ext in [".jpg", ".png"]:
            img_path = os.path.join(train_img_src, img_base + ext)
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(train_img_dst, os.path.basename(img_path)))
                id0_img_count += 1
                break

    # 第二步：复制不含ID0/未被选中的其他类别样本
    other_img_count = 0
    all_train_labels = glob.glob(os.path.join(train_label_src, "*.txt"))
    for txt in all_train_labels:
        if txt in selected_id0_files:
            continue  # 已复制，跳过

        # 检查是否包含非ID0类别
        with open(txt, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        has_other = any(int(l.split()[0]) != ID0_CLASS_ID for l in lines if len(l.split()) >= 5)

        if not has_other:
            continue  # 仅含ID0且未被选中，视为冗余，跳过

        # 复制标注和图片
        txt_name = os.path.basename(txt)
        shutil.copy2(txt, os.path.join(train_label_dst, txt_name))
        img_base = txt_name.replace(".txt", "")
        for ext in [".jpg", ".png"]:
            img_path = os.path.join(train_img_src, img_base + ext)
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(train_img_dst, os.path.basename(img_path)))
                other_img_count += 1
                break

    print(f" 训练集复制完成：ID0核心{id0_img_count}张，其他类别{other_img_count}张")


# ====================== 步骤5：验证最终数据集 ======================
def verify_final_dataset():
    """验证最终数据集的结构和ID0数量"""
    print("\n 验证最终数据集...")
    # 统计ID0数量
    id0_final_count = 0
    for txt in glob.glob(os.path.join(FINAL_ROOT, "labels", "train", "*.txt")):
        with open(txt, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        id0_final_count += sum(1 for l in lines if len(l.split()) >= 5 and int(l.split()[0]) == ID0_CLASS_ID)

    # 打印目录结构校验
    print(" 最终目录结构校验：")
    expected_paths = [
        f"{FINAL_ROOT}/images/train",
        f"{FINAL_ROOT}/images/val",
        f"{FINAL_ROOT}/labels/train",
        f"{FINAL_ROOT}/labels/val",
        f"{FINAL_ROOT}/data.yaml"
    ]
    for path in expected_paths:
        status = " 存在" if os.path.exists(path) else " 缺失"
        print(f"   {status}：{path}")

    print(f"\n 核心指标校验：")
    print(f"   ID0最终样本数：{id0_final_count}")
    print(f"   训练集图片数：{len(os.listdir(os.path.join(FINAL_ROOT, 'images', 'train')))}")
    print(f"   验证集图片数：{len(os.listdir(os.path.join(FINAL_ROOT, 'images', 'val')))}")


# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        # 1. 构建目标目录结构
        build_final_structure()
        # 2. 复制验证集（全保留）
        copy_val_set()
        # 3. 筛选ID0核心训练样本
        selected_id0 = select_id0_core_samples()
        # 4. 复制训练集（去冗余）
        copy_train_set(selected_id0)
        # 5. 验证最终结果
        verify_final_dataset()

        print("\n 最终数据集生成完成！")
        print(f" 训练时直接使用：{FINAL_ROOT}")
        print(f" data.yaml已自动复制，无需修改路径即可训练")
    except Exception as e:
        print(f"\n 执行出错：{e}")
        import traceback

        traceback.print_exc()