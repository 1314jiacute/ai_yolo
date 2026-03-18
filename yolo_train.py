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
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# ====================== 基础配置 ======================
# 增强后数据集根路径
DATA_ROOT = r"/workspace/examples/PythonProject1/data_dectectenhance"
# 已有data.yaml路径
DATA_YAML_PATH = r"/workspace/examples/PythonProject1/data_dectectenhance/data.yaml"
# 实验结果保存路径
RESULT_ROOT = r"/workspace/examples/PythonProject1/pest_experiment_results"
# 随机种子
SEED = 42
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 模型训练超参数
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMG_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# 创建结果保存目录
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "detection_models"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "metrics"), exist_ok=True)

# 固定随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ====================== 解析已有data.yaml ======================
def parse_existing_yaml():
    """验证并解析已有data.yaml"""
    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError(f"data.yaml文件不存在：{DATA_YAML_PATH}")

    import yaml as pyyaml
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        yaml_data = pyyaml.safe_load(f)

    # 验证核心字段
    required_fields = ["path", "train", "val", "nc", "names"]
    for field in required_fields:
        if field not in yaml_data:
            raise ValueError(f"data.yaml缺少字段：{field}")

    # 修正数据集根路径
    yaml_data["path"] = DATA_ROOT
    num_classes = yaml_data["nc"]
    class_names = yaml_data["names"]

    # 验证数据集路径
    train_img_path = os.path.join(yaml_data["path"], yaml_data["train"])
    val_img_path = os.path.join(yaml_data["path"], yaml_data["val"])
    if not os.path.exists(train_img_path):
        raise FileNotFoundError(f"训练集路径不存在：{train_img_path}")
    if not os.path.exists(val_img_path):
        raise FileNotFoundError(f"验证集路径不存在：{val_img_path}")

    print(f"\n  data.yaml验证通过！")
    print(f" 类别数：{num_classes} | 训练集：{len(glob.glob(os.path.join(train_img_path, '*')))}张")
    return DATA_YAML_PATH, num_classes, class_names


# ====================== 获取模型参数 ======================
def get_model_params(model):
    """计算模型参数数量（单位：M）"""
    total_params = sum(p.numel() for p in model.model.parameters())
    return round(total_params / 1e6, 2)


# ====================== 训练YOLO检测模型 ======================
def train_yolo_models(config_path):
    """训练YOLOv8n/YOLOv5s（基线） + YOLOv8m（上界）"""
    # 定义要训练的模型
    yolo_models = {
        "YOLOv8n": {"pretrained": "yolov8n.pt",
                    "save_path": os.path.join(RESULT_ROOT, "detection_models", "yolov8n.pt")},
        "YOLOv5s": {"pretrained": "yolov5s.pt",
                    "save_path": os.path.join(RESULT_ROOT, "detection_models", "yolov5s.pt")},
        "YOLOv8m": {"pretrained": "yolov8m.pt",
                    "save_path": os.path.join(RESULT_ROOT, "detection_models", "yolov8m.pt")}
    }

    detection_metrics = {}

    for model_name, info in yolo_models.items():
        print(f"\n====================== 开始训练 {model_name} ======================")

        # 加载预训练模型
        model = YOLO(info["pretrained"])

        # 训练模型
        start_time = time.time()
        train_results = model.train(
            data=config_path,
            epochs=TRAIN_EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            lr0=LEARNING_RATE,
            device=0 if torch.cuda.is_available() else None,
            project=RESULT_ROOT,
            name=f"yolo_train_{model_name}",
            exist_ok=True,
            seed=SEED,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        train_time = (time.time() - start_time) / 3600  # 转换为小时

        # 保存模型
        model.save(info["save_path"])

        # 验证模型
        val_results = model.val(
            data=config_path,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=0 if torch.cuda.is_available() else None,
            verbose=False
        )

        # 测试推理速度（取验证集100张图片）
        val_img_dir = os.path.join(DATA_ROOT, "images", "val")
        val_img_paths = glob.glob(os.path.join(val_img_dir, "*"))[:100]
        infer_times = []

        model.eval()
        for img_path in tqdm(val_img_paths, desc=f"{model_name} 推理速度测试"):
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            start = time.time()
            model(img, imgsz=IMG_SIZE, verbose=False)
            infer_times.append(time.time() - start)

        # 计算速度指标
        avg_infer_time = np.mean(infer_times) if infer_times else 0
        fps = 1 / avg_infer_time if avg_infer_time > 0 else 0

        # 计算模型体积
        model_size = os.path.getsize(info["save_path"]) / 1024 / 1024 if os.path.exists(info["save_path"]) else 0

        # 记录指标
        detection_metrics[model_name] = {
            "模型类型": "检测模型",
            "参数数(M)": get_model_params(model),
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "平均精度": float(val_results.box.mp),
            "平均召回": float(val_results.box.mr),
            "FPS": fps,
            "平均推理时间(ms)": avg_infer_time * 1000,
            "模型体积(MB)": model_size,
            "训练时间(h)": train_time
        }

        # 打印当前模型结果
        print(f"\n  {model_name} 训练完成！")
        print(
            f"核心指标：mAP50={detection_metrics[model_name]['mAP50']:.4f} | FPS={fps:.2f} | 模型体积={model_size:.2f}MB")

    # 保存指标到文件
    metrics_json_path = os.path.join(RESULT_ROOT, "metrics", "yolo_detection_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(detection_metrics, f, indent=4, ensure_ascii=False)

    # 保存为CSV表格
    metrics_csv_path = os.path.join(RESULT_ROOT, "metrics", "yolo_detection_metrics.csv")
    pd.DataFrame(detection_metrics).T.round(4).to_csv(metrics_csv_path, index=True, encoding="utf-8-sig")

    print(f"\n  所有YOLO模型训练完成！")
    print(f"  指标文件保存路径：{metrics_json_path}")
    print(f" 模型保存路径：{os.path.join(RESULT_ROOT, 'detection_models')}")

    return detection_metrics


# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        # 1. 解析data.yaml
        config_path, num_classes, class_names = parse_existing_yaml()

        # 2. 训练YOLO模型
        train_yolo_models(config_path)

        print("\n====================== YOLO训练脚本执行完成 ======================")
    except Exception as e:
        print(f"\n 训练出错：{str(e)}")
        import traceback

        traceback.print_exc()