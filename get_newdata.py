import os
import shutil
import yaml
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ===================== 核心配置 =====================
# 原始数据集根路径（你的 data_dectect 目录）
ORIGINAL_DATA_DIR = r"/workspace/examples/PythonProject1/data_dectect"
# 新数据集保存路径
NEW_DATA_DIR = r"/workspace/examples/PythonProject1/data_dectect_t2"
# 最小样本数阈值
MIN_SAMPLE_COUNT = 300
# 原始数据集的结构类型（你现在的是结构2）
DATA_STRUCTURE_TYPE = 2  # images/train, labels/train


# ===================== 路径适配函数 =====================
def get_original_paths(set_name):
    """获取原始数据集的图片和标注路径"""
    if DATA_STRUCTURE_TYPE == 1:
        # 结构1: train/images, train/labels
        img_dir = os.path.join(ORIGINAL_DATA_DIR, set_name, "images")
        lbl_dir = os.path.join(ORIGINAL_DATA_DIR, set_name, "labels")
    else:
        # 结构2: images/train, labels/train
        img_dir = os.path.join(ORIGINAL_DATA_DIR, "images", set_name)
        lbl_dir = os.path.join(ORIGINAL_DATA_DIR, "labels", set_name)
    return img_dir, lbl_dir


# ===================== 核心功能函数 =====================
def count_class_samples():
    """统计每个原始ID的样本数，保留名称映射"""
    yaml_path = os.path.join(ORIGINAL_DATA_DIR, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"找不到原始data.yaml: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    original_names = yaml_data['names']  # 字典格式 {0: '稻纵卷叶螟', ...}
    class_count = defaultdict(int, {cls_id: 0 for cls_id in original_names.keys()})
    all_labels = {}
    sets = ["train", "val"]

    # 统计总文件数
    total_files = 0
    for set_name in sets:
        _, lbl_dir = get_original_paths(set_name)
        if os.path.exists(lbl_dir):
            total_files += len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])

    print(f"开始统计 {total_files} 个标注文件，共 {len(original_names)} 个类别...")
    processed_files = 0

    for set_name in sets:
        img_dir, lbl_dir = get_original_paths(set_name)
        if not os.path.exists(lbl_dir):
            print(f"警告：{lbl_dir} 不存在，跳过 {set_name} 集")
            continue

        for label_file in os.listdir(lbl_dir):
            if not label_file.endswith(".txt"):
                continue

            processed_files += 1
            if processed_files % 1000 == 0:
                print(f"已处理 {processed_files}/{total_files} 个文件")

            label_path = os.path.join(lbl_dir, label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            file_classes = []
            for line in lines:
                try:
                    cls_id = int(line.split()[0])
                    if cls_id in original_names:
                        class_count[cls_id] += 1
                        file_classes.append(cls_id)
                except (IndexError, ValueError):
                    print(f"警告：{label_path} 行格式错误，跳过: {line}")
                    continue

            all_labels[(set_name, label_file)] = {
                "lines": lines,
                "classes": file_classes,
                "img_dir": img_dir
            }

    print(f"统计完成！共处理 {processed_files} 个文件")
    return class_count, all_labels, original_names


def filter_classes(class_count, min_samples, original_names):
    """筛选样本数达标（≥ min_samples）的类别"""
    qualified = [(cls_id, cnt) for cls_id, cnt in class_count.items() if cnt >= min_samples]
    unqualified = [(cls_id, cnt) for cls_id, cnt in class_count.items() if cnt < min_samples]

    qualified_sorted = sorted(qualified, key=lambda x: x[1], reverse=True)
    unqualified_sorted = sorted(unqualified, key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print(f"类别筛选结果（阈值：{min_samples} 样本）")
    print("=" * 60)
    print(f"原始类别总数：{len(class_count)}")
    print(f"达标类别数：{len(qualified_sorted)}")
    print(f"删除类别数：{len(unqualified_sorted)}")

    print("\n【达标类别（按样本数降序）】")
    if qualified_sorted:
        for i, (cls_id, cnt) in enumerate(qualified_sorted[:20]):
            print(f"{i + 1:2d}. ID:{cls_id:<4} | {original_names[cls_id]:<12} | {cnt:>5d} 样本")
        if len(qualified_sorted) > 20:
            print(f"... 还有 {len(qualified_sorted) - 20} 个达标类别")
    else:
        print("  无达标类别！请降低阈值")

    print("\n【被删除的类别（按样本数降序）】")
    if unqualified_sorted:
        for i, (cls_id, cnt) in enumerate(unqualified_sorted[:20]):
            print(f"{i + 1:2d}. ID:{cls_id:<4} | {original_names[cls_id]:<12} | {cnt:>5d} 样本")
        if len(unqualified_sorted) > 20:
            print(f"... 还有 {len(unqualified_sorted) - 20} 个被删除类别")

    # 保存删除日志
    os.makedirs(NEW_DATA_DIR, exist_ok=True)
    log_path = os.path.join(NEW_DATA_DIR, "deleted_classes.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"删除阈值：{min_samples} 样本\n")
        f.write("=" * 30 + "达标类别" + "=" * 30 + "\n")
        for cls_id, cnt in qualified_sorted:
            f.write(f"{cls_id}: {original_names[cls_id]} - {cnt} 样本\n")
        f.write("\n" + "=" * 30 + "被删除类别" + "=" * 30 + "\n")
        for cls_id, cnt in unqualified_sorted:
            f.write(f"{cls_id}: {original_names[cls_id]} - {cnt} 样本\n")
    print(f"\n删除日志已保存至：{log_path}")

    return [cls_id for cls_id, cnt in qualified_sorted]


def create_cleaned_dataset(qualified_original_ids, all_labels, original_names):
    """生成复刻原始目录结构的新数据集"""
    # 1. 先清理旧目录
    if os.path.exists(NEW_DATA_DIR):
        shutil.rmtree(NEW_DATA_DIR)

    # 2. 创建和原始结构完全一样的目录
    for set_name in ["train", "val"]:
        os.makedirs(os.path.join(NEW_DATA_DIR, "images", set_name), exist_ok=True)
        os.makedirs(os.path.join(NEW_DATA_DIR, "labels", set_name), exist_ok=True)

    # 3. 构建新的连续索引映射（0 ~ nc-1）
    original2new = {old_id: new_id for new_id, old_id in enumerate(qualified_original_ids)}
    new2name = {new_id: original_names[old_id] for old_id, new_id in original2new.items()}

    # 保存映射日志
    mapping_log = os.path.join(NEW_DATA_DIR, "class_mapping.log")
    with open(mapping_log, 'w', encoding='utf-8') as f:
        f.write("原始ID → 新索引 → 类别名称\n")
        f.write("=" * 50 + "\n")
        for old_id, new_id in sorted(original2new.items(), key=lambda x: x[1]):
            f.write(f"{old_id:>4} → {new_id:>2} → {original_names[old_id]}\n")
    print(f"\n类别映射日志已保存至：{mapping_log}")

    # 4. 处理标注和图片
    kept_files = 0
    skipped_files = 0
    img_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    total = len(all_labels)
    processed = 0

    print("\n开始生成新数据集...")
    for (set_name, label_file), label_info in all_labels.items():
        processed += 1
        if processed % 1000 == 0:
            print(f"处理进度：{processed}/{total} 文件")

        new_lines = []
        keep_file = False
        for line in label_info["lines"]:
            try:
                parts = line.split()
                old_cls_id = int(parts[0])
                if old_cls_id in original2new:
                    new_cls_id = original2new[old_cls_id]
                    new_line = f"{new_cls_id} {' '.join(parts[1:])}"
                    new_lines.append(new_line + "\n")
                    keep_file = True
            except (IndexError, ValueError):
                continue

        if keep_file:
            kept_files += 1
            # 保存标注到新目录：labels/train 或 labels/val
            new_lbl_path = os.path.join(NEW_DATA_DIR, "labels", set_name, label_file)
            with open(new_lbl_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            # 复制图片到新目录：images/train 或 images/val
            img_base = os.path.splitext(label_file)[0]
            old_img_dir = label_info["img_dir"]
            for fmt in img_formats:
                old_img_path = os.path.join(old_img_dir, img_base + fmt)
                if os.path.exists(old_img_path):
                    new_img_path = os.path.join(NEW_DATA_DIR, "images", set_name, img_base + fmt)
                    shutil.copy2(old_img_path, new_img_path)
                    break
        else:
            skipped_files += 1

    # 5. 生成完全匹配目录结构的 data.yaml
    nc = len(qualified_original_ids)
    new_names_dict = {new_id: new2name[new_id] for new_id in sorted(new2name.keys())}

    yaml_path = os.path.join(NEW_DATA_DIR, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {NEW_DATA_DIR}\n")
        f.write(f"train: images/train\n")  # 完全复刻原始结构
        f.write(f"val: images/val\n")  # 完全复刻原始结构
        f.write(f"nc: {nc}\n")
        f.write(f"names:\n")
        for new_id in sorted(new_names_dict.keys()):
            f.write(f"  {new_id}: {new_names_dict[new_id]}\n")

    # 输出最终信息
    print("\n" + "=" * 60)
    print("  数据集清理完成！目录结构与原始完全一致")
    print("=" * 60)
    print(f"保留文件数：{kept_files}")
    print(f"跳过文件数：{skipped_files}")
    print(f"新类别数：{nc}")
    print(f"新数据集路径：{NEW_DATA_DIR}")
    print(f"新data.yaml路径：{yaml_path}")
    print("\n新数据集目录结构：")
    print(f"  {NEW_DATA_DIR}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  ├── labels/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  └── data.yaml")


# ===================== 主执行流程 =====================
if __name__ == "__main__":
    try:
        class_count, all_labels, original_names = count_class_samples()
        qualified_ids = filter_classes(class_count, MIN_SAMPLE_COUNT, original_names)

        if qualified_ids:
            confirm = input("\n是否确认生成新数据集？(y/n): ")
            if confirm.lower() == 'y':
                create_cleaned_dataset(qualified_ids, all_labels, original_names)
            else:
                print("操作已取消")
        else:
            print("\n 无达标类别，请降低 MIN_SAMPLE_COUNT 后重试")
    except Exception as e:
        print(f"\n  执行出错: {e}")