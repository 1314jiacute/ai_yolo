from PIL import Image
import os

# 1. 定义路径
input_dir = "FaceData/images"  # 原始图片所在文件夹
output_dir = "FaceData/resized_images"  # 调整后图片的输出文件夹

# 2. 创建输出文件夹（不存在则自动创建）
os.makedirs(output_dir, exist_ok=True)

# 3. 遍历原始图片并调整尺寸
for filename in os.listdir(input_dir):
    # 只处理.jpg格式的图片
    if filename.endswith(".jpg"):
        # 拼接原始图片的完整路径
        img_path = os.path.join(input_dir, filename)
        # 打开图片
        try:
            with Image.open(img_path) as img:
                # 调整尺寸为640x640（直接拉伸适配，若需保持比例可改用resize+填充，这里按需求直接调整）
                resized_img = img.resize((640, 640), Image.Resampling.LANCZOS)  # LANCZOS是高清缩放算法
                # 拼接输出路径并保存
                output_path = os.path.join(output_dir, filename)
                resized_img.save(output_path)
                print(f"已处理：{filename} → 保存到 {output_path}")
        except Exception as e:
            print(f"处理{filename}失败：{e}")

print("所有图片尺寸调整完成！")