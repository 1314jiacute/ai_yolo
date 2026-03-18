import os

# 定义训练集图片文件夹路径
train_images_dir = "/workspace/examples/PythonProject1/data_dectect/images/train"
train_val_images_dir = "/workspace/examples/PythonProject1/data_dectect/images/val"
# 检查文件夹是否存在
if not os.path.exists(train_images_dir):
    print(f"错误：文件夹 {train_images_dir} 不存在！")
else:
    # 统计文件夹中的文件数量（不区分文件类型，假设都是图片）
    image_count = len([f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))])
    print(f"训练集图片数量：{image_count} 张")
    image_countx = len([f for f in os.listdir(train_val_images_dir) if os.path.isfile(os.path.join(train_val_images_dir, f))])
    print(f"验证集图片数量：{image_countx} 张")