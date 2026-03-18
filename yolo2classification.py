import os
import yaml
import shutil
import argparse
from tqdm import tqdm

def load_yolo_yaml(yaml_path):
    """加载YOLO的data.yaml配置文件，获取类别信息和数据集路径（适配字典/列表格式的names）"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取关键信息
    nc = config.get('nc', 0)  # 类别数
    names = config.get('names', {})  # 类别名称（兼容字典/列表）
    
    # 统一转换为：类别ID→类别名的字典 + 按ID排序的类别名列表
    class_name_dict = {}
    if isinstance(names, dict):
        # 处理字典格式（{0: '叶蝉科', 1: '蚜虫'}）
        for k, v in names.items():
            try:
                class_id = int(k)
                class_name_dict[class_id] = v
            except ValueError:
                continue
    elif isinstance(names, list):
        # 处理列表格式（['叶蝉科', '蚜虫']）
        for class_id, name in enumerate(names):
            class_name_dict[class_id] = name
    
    # 按类别ID排序，生成类别名列表
    class_names = [class_name_dict[cid] for cid in sorted(class_name_dict.keys())]
    # 确保类别数匹配
    if nc != len(class_names):
        print(f" 警告：yaml中nc={nc}与实际类别数={len(class_names)}不匹配，以实际为准")
        nc = len(class_names)
    
    train_img_path = config.get('train', '')  # 训练集图片路径
    val_img_path = config.get('val', '')    # 验证集图片路径
    
    # 处理路径（兼容不同写法）
    if isinstance(train_img_path, list):
        train_img_path = train_img_path[0]
    if isinstance(val_img_path, list):
        val_img_path = val_img_path[0]
    
    return {
        'nc': nc,
        'class_name_dict': class_name_dict,  # 新增：类别ID→名称的字典
        'class_names': class_names,          # 按ID排序的类别名列表
        'train_img': train_img_path,
        'val_img': val_img_path
    }

def extract_classification_data(
    yolo_config, 
    output_root,
    train_label_path=None,  # 手动指定训练标注路径
    val_label_path=None,    # 手动指定验证标注路径
    handle_multi_label='first',  # first:取第一个标签, skip:跳过多标签, all:复制到所有类别
    img_formats=('.jpg', '.jpeg', '.png', '.bmp')
):
    """
    从YOLO数据集提取分类数据（支持手动指定标注路径）
    :param yolo_config: load_yolo_yaml返回的配置字典
    :param output_root: 分类数据集输出根目录
    :param train_label_path: 训练集标注文件路径（手动指定）
    :param val_label_path: 验证集标注文件路径（手动指定）
    :param handle_multi_label: 多标签图片处理策略
    :param img_formats: 支持的图片格式
    """
    # 创建分类数据集目录结构
    train_output = os.path.join(output_root, 'train')
    val_output = os.path.join(output_root, 'val')
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    
    # 为每个类别创建文件夹（使用字符串类型的类别名）
    for cls_name in yolo_config['class_names']:
        # 确保类别名是字符串（防御性处理）
        cls_name_str = str(cls_name).strip()
        os.makedirs(os.path.join(train_output, cls_name_str), exist_ok=True)
        os.makedirs(os.path.join(val_output, cls_name_str), exist_ok=True)
    
    # 定义处理单批次数据的函数
    def process_dataset(img_dir, label_dir, output_dir, dataset_type):
        if not label_dir or not os.path.exists(label_dir):
            print(f"  {dataset_type}标注文件夹不存在：{label_dir}")
            return
        
        # 遍历所有标注文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        if not label_files:
            print(f"  {dataset_type}标注文件夹中无txt文件：{label_dir}")
            return
        
        skipped_multi_label = 0
        copied_imgs = 0
        
        for label_file in tqdm(label_files, desc=f"处理{dataset_type}集"):
            # 读取标注文件
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # 跳过空标注文件
            if not lines:
                continue
            
            # 处理多标签图片
            class_ids = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        class_id = int(parts[0])
                        # 检查类别ID是否有效
                        if class_id in yolo_config['class_name_dict']:
                            class_ids.append(class_id)
                    except ValueError:
                        continue
            
            # 过滤无效类别ID
            class_ids = [cid for cid in class_ids if cid in yolo_config['class_name_dict']]
            if not class_ids:
                continue
            
            # 根据策略处理多标签
            target_class_ids = []
            if handle_multi_label == 'first':
                target_class_ids = [class_ids[0]]  # 取第一个标签
            elif handle_multi_label == 'all':
                target_class_ids = list(set(class_ids))  # 去重后复制到所有类别
            elif handle_multi_label == 'skip':
                if len(class_ids) > 1:
                    skipped_multi_label += 1
                    continue
                target_class_ids = class_ids
            
            # 复制图片到对应类别文件夹
            img_name = label_file.replace('.txt', '')
            img_path = None
            for fmt in img_formats:
                temp_path = os.path.join(img_dir, img_name + fmt)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            if img_path is None:
                print(f" 未找到图片：{os.path.join(img_dir, img_name)}")
                continue
            
            # 复制到目标类别文件夹（使用类别名字典获取正确名称）
            for cid in target_class_ids:
                cls_name = yolo_config['class_name_dict'][cid]
                cls_name_str = str(cls_name).strip()  # 确保是字符串
                dst_path = os.path.join(output_dir, cls_name_str, os.path.basename(img_path))
                # 避免重复复制（如果已存在则跳过）
                if not os.path.exists(dst_path):
                    shutil.copy2(img_path, dst_path)
                    copied_imgs += 1
        
        print(f"\n{dataset_type}集处理完成：")
        print(f"  - 成功复制图片数：{copied_imgs}")
        print(f"  - 跳过的多标签图片数：{skipped_multi_label}")
    
    # 处理训练集和验证集（使用手动指定的标注路径）
    process_dataset(yolo_config['train_img'], train_label_path, train_output, '训练')
    process_dataset(yolo_config['val_img'], val_label_path, val_output, '验证')
    
    # 统计每个类别的图片数量
    print("\n  分类数据集统计：")
    for split in ['train', 'val']:
        print(f"\n{split}集：")
        split_dir = os.path.join(output_root, split)
        for cls_name in yolo_config['class_names']:
            cls_name_str = str(cls_name).strip()
            cls_dir = os.path.join(split_dir, cls_name_str)
            if os.path.exists(cls_dir):
                img_count = len([f for f in os.listdir(cls_dir) if f.endswith(img_formats)])
                print(f"  {cls_name_str}: {img_count} 张")
    
    print(f"\n  分类数据集创建完成！")
    print(f" 保存路径：{output_root}")

def main():
    # 命令行参数解析（支持手动指定标注路径）
    parser = argparse.ArgumentParser(description='从YOLO数据集创建分类数据集（支持手动指定标注路径）')
    parser.add_argument('--yaml_path', 
                        default='/workspace/examples/PythonProject1/data_dectect_t2/data.yaml',
                        help='YOLO的data.yaml文件路径')
    parser.add_argument('--output_root', 
                        default='/workspace/examples/PythonProject1/pest_classification_data',
                        help='分类数据集输出根目录')
    parser.add_argument('--train_label_path', 
                        default='/workspace/examples/PythonProject1/data_dectect_t2/labels/train',  # 手动指定训练标注路径
                        help='训练集标注文件路径（手动指定）')
    parser.add_argument('--val_label_path', 
                        default='/workspace/examples/PythonProject1/data_dectect_t2/labels/val',    # 手动指定验证标注路径
                        help='验证集标注文件路径（手动指定）')
    parser.add_argument('--multi_label_strategy', 
                        default='first', 
                        choices=['first', 'skip', 'all'],
                        help='多标签图片处理策略：first(取第一个)/skip(跳过)/all(复制到所有类别)')
    
    args = parser.parse_args()
    
    # 1. 加载YOLO配置
    print("  加载YOLO配置文件...")
    yolo_config = load_yolo_yaml(args.yaml_path)
    print(f" 加载完成：类别数={yolo_config['nc']}，类别列表={yolo_config['class_names']}")
    print(f"  训练图片路径：{yolo_config['train_img']}")
    print(f"  验证图片路径：{yolo_config['val_img']}")
    print(f"  训练标注路径：{args.train_label_path}")
    print(f"  验证标注路径：{args.val_label_path}")
    
    # 2. 提取分类数据
    print("\n  开始提取分类数据...")
    extract_classification_data(
        yolo_config, 
        args.output_root,
        train_label_path=args.train_label_path,
        val_label_path=args.val_label_path,
        handle_multi_label=args.multi_label_strategy
    )

if __name__ == "__main__":
    main()