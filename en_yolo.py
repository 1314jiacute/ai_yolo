import os
import cv2
import numpy as np
import random

# ===================== 核心配置（针对害虫优化） =====================
BASE_DIR = r"D:\PycharmProjects\PythonProject1\data_dectect_small_en"
# 增强参数（平衡轮廓和纹理，无需调整）
SHARP_STRENGTH = 1.7  # 轮廓增强强度
CONTRAST_ALPHA = 1.5  # 对比度系数
BRIGHTNESS_BETA = 3  # 亮度补偿
NOISE_KERNEL = (3, 3)  # 降噪核大小


# ===================================================================

def pest_enhance_core(img):
    """核心增强逻辑：突出轮廓+保留纹理+提升清晰度"""
    # 1. 基础降噪（仅去噪点，保留害虫细节）
    img_denoise = cv2.GaussianBlur(img, NOISE_KERNEL, 0.6)

    # 2. 轮廓增强（非纯线条，保留纹理）
    # 边缘增强核：强化轮廓但不丢失纹理
    contour_kernel = np.array([
        [-0.15, -0.15, -0.15],
        [-0.15, 2.2, -0.15],
        [-0.15, -0.15, -0.15]
    ])
    img_contour = cv2.filter2D(img_denoise, -1, contour_kernel)

    # 3. 对比度+亮度自适应增强（突出害虫）
    img_contrast = cv2.convertScaleAbs(
        img_contour,
        alpha=CONTRAST_ALPHA,
        beta=BRIGHTNESS_BETA
    )

    # 4. 最终优化：防过增强+色彩还原
    img_final = np.clip(img_contrast, 0, 255).astype(np.uint8)
    # 轻微色彩平衡，避免偏色
    img_final = cv2.cvtColor(
        cv2.cvtColor(img_final, cv2.COLOR_BGR2LAB),
        cv2.COLOR_LAB2BGR
    )

    return img_final


def batch_process_yolo_dataset():
    """批量处理YOLO数据集（train/val）"""
    # 定义输入输出目录
    input_paths = {
        "train": os.path.join(BASE_DIR, "images/train"),
        "val": os.path.join(BASE_DIR, "images/val")
    }
    output_paths = {
        "train": os.path.join(BASE_DIR, "pest_enhanced/train"),
        "val": os.path.join(BASE_DIR, "pest_enhanced/val")
    }

    # 创建输出目录
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # 统计处理数量
    processed_count = 0
    skipped_count = 0

    # 批量处理
    for phase in ["train", "val"]:
        in_dir = input_paths[phase]
        out_dir = output_paths[phase]

        if not os.path.exists(in_dir):
            print(f"  跳过{phase}集：目录不存在 -> {in_dir}")
            continue

        for img_name in os.listdir(in_dir):
            # 仅处理图片文件
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                skipped_count += 1
                continue

            # 读取图片
            img_path = os.path.join(in_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  跳过无法读取的图片：{img_name}")
                skipped_count += 1
                continue

            # 增强处理
            img_enhanced = pest_enhance_core(img)

            # 保存增强后的图片（保持原文件名，适配YOLO标注）
            save_path = os.path.join(out_dir, img_name)
            cv2.imwrite(save_path, img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

            processed_count += 1
            print(f" 处理完成：{img_name}（{phase}集）")

    # 生成对比图（直观验证效果）
    generate_comparison_img(input_paths["train"], output_paths["train"])

    # 打印处理总结
    print("\n" + "=" * 50)
    print(f"处理完成！总计处理：{processed_count} 张 | 跳过：{skipped_count} 张")
    print(f"增强后数据集路径：{BASE_DIR}/pest_enhanced")
    print("  注：YOLO标注文件（labels目录）无需修改，直接使用即可")
    print("=" * 50)


def generate_comparison_img(train_in, train_out):
    """生成原图vs增强图的对比图，方便验证效果"""
    # 随机选一张图片生成对比
    img_names = [f for f in os.listdir(train_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_names:
        return

    sample_name = random.choice(img_names)
    img_original = cv2.imread(os.path.join(train_in, sample_name))
    img_enhanced = cv2.imread(os.path.join(train_out, sample_name))

    # 统一尺寸（避免极少数情况尺寸不一致）
    if img_original.shape != img_enhanced.shape:
        img_enhanced = cv2.resize(img_enhanced, (img_original.shape[1], img_original.shape[0]))

    # 添加文字标注
    cv2.putText(img_original, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img_enhanced, "Enhanced (Pest)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 横向拼接对比图
    comparison_img = np.hstack((img_original, img_enhanced))
    # 保存对比图
    save_path = os.path.join(BASE_DIR, "pest_enhance_comparison.png")
    cv2.imwrite(save_path, comparison_img)
    print(f"\n  对比图已保存：{save_path}（可直接查看增强效果）")


if __name__ == "__main__":
    print("=" * 60)
    print("Python 3.13 害虫数据集增强启动（YOLO小模型专属）")
    print(f"数据集根目录：{BASE_DIR}")
    print("=" * 60)
    batch_process_yolo_dataset()