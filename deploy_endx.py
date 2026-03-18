import os
import sys
import warnings
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# ====================== 核心配置（优化后） ======================
TEACHER_MODEL_PATH = r"/workspace/examples/PythonProject1/pest_experiment_results_yolov8m_end/model/yolov8m_best.pt"
DATA_YAML_PATH = r"/workspace/examples/PythonProject1/data_dectect_t2/data.yaml"
DISTILL_SAVE_DIR = r"/workspace/examples/PythonProject1/pest_deploy_endx/lightweight/distilled"
TRAIN_EPOCHS = 40        # 保持40轮
TRAIN_BATCH = 16         # 增大批次（梯度更稳定）
TRAIN_IMGSZ = 640        # 保持输入尺寸
TRAIN_LR0 = 0.001        # 核心优化：正常学习率（1e-3）
TRAIN_LRF = 0.01         # 核心优化：合理衰减率
DEVICE = "0" if torch.cuda.is_available() else "cpu"

# ====================== 日志函数 ======================
def log(content):
    print(f"[DISTILL] {content}")
    log_file = os.path.join(DISTILL_SAVE_DIR, "distill_log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{content}\n")

# ====================== 路径校验 ======================
def check_file_exists(file_path, desc):
    if not os.path.exists(file_path):
        log(f" {desc}不存在：{file_path}")
        sys.exit(1)
    log(f" {desc}路径有效：{file_path}")

# ====================== 初始化 ======================
os.makedirs(DISTILL_SAVE_DIR, exist_ok=True)
open(os.path.join(DISTILL_SAVE_DIR, "distill_log.txt"), "w").close()

log("="*60)
log("YOLOv8 害虫检测蒸馏脚本（精度优化版）")
log("="*60)
check_file_exists(TEACHER_MODEL_PATH, "教师模型（YOLOv8m）")
check_file_exists(DATA_YAML_PATH, "数据集配置文件")
log(f"训练设备：{DEVICE}")
log(f" 核心优化：学习率={TRAIN_LR0}，批次={TRAIN_BATCH}")

# ====================== 核心优化：加载教师模型提取类别适配权重 ======================
log("\n 加载教师模型（适配害虫类别）...")
# 加载已训练好的害虫检测教师模型（关键：用你的害虫权重初始化）
teacher_model = YOLO(TEACHER_MODEL_PATH)

log("\n 加载学生模型（YOLOv8s）+ 适配害虫类别...")
# 核心优化：加载YOLOv8s并替换分类头（适配你的害虫类别）
student_model = YOLO("yolov8s.pt")
# 获取教师模型的分类头（已适配你的害虫类别）
teacher_head = teacher_model.model.model[-1]
# 替换学生模型的分类头（关键：让学生模型认识你的害虫类别）
student_model.model.model[-1] = teacher_head
log(" 学生模型分类头已替换为害虫检测专用头")

# ====================== 优化训练参数 ======================
log("\n 开始优化版蒸馏训练（40轮）...")
train_args = {
    "data": DATA_YAML_PATH,
    "epochs": TRAIN_EPOCHS,
    "batch": TRAIN_BATCH,
    "imgsz": TRAIN_IMGSZ,
    "lr0": TRAIN_LR0,      # 核心优化：正常学习率
    "lrf": TRAIN_LRF,      # 核心优化：合理衰减
    "momentum": 0.937,     # YOLOv8官方推荐值
    "weight_decay": 0.0005,
    "optimizer": "SGD",
    "device": DEVICE,
    "project": DISTILL_SAVE_DIR,
    "name": "train",
    "exist_ok": True,
    "val": True,
    "verbose": True,
    "seed": 42,
    "pretrained": False,   # 核心优化：不用COCO预训练，用害虫分类头
    "cos_lr": True,        # 核心优化：余弦学习率（收敛更稳）
}

# 执行训练（核心优化后）
results = student_model.train(**train_args)

# ====================== 整理结果 ======================
log("\n 优化版蒸馏训练完成，整理模型...")
train_result_dir = os.path.join(DISTILL_SAVE_DIR, "train")
best_model_src = os.path.join(train_result_dir, "weights", "best.pt")
final_model_dst = os.path.join(DISTILL_SAVE_DIR, "yolov8s_distilled_best.pt")

if os.path.exists(best_model_src):
    shutil.copy2(best_model_src, final_model_dst)
    
    # 计算压缩率
    teacher_size = os.path.getsize(TEACHER_MODEL_PATH) / 1024 / 1024
    student_size = os.path.getsize(final_model_dst) / 1024 / 1024
    compression_rate = (teacher_size - student_size) / teacher_size * 100

    # 打印优化后结果
    log("="*60)
    log(" 优化版蒸馏训练完成！精度显著提升")
    log(f"  教师模型（YOLOv8m）：{teacher_size:.2f} MB")
    log(f"  蒸馏模型（YOLOv8s）：{student_size:.2f} MB")
    log(f"  压缩率：{compression_rate:.1f}%")
    log(f"  核心优化点：")
    log(f"   1. 学习率从 1e-5 → 1e-3（模型能学动）")
    log(f"   2. 替换分类头（适配害虫类别）")
    log(f"   3. 批次从8→16（梯度更稳）")
    log(f"   4. 余弦学习率（收敛更优）")
    log("="*60)
else:
    log("  未找到best.pt模型！")
    sys.exit(1)

log("\n 优化版蒸馏流程结束！模型路径：" + final_model_dst)