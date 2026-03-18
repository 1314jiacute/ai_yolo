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
import random
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# ====================== 基础配置 ======================
DATA_ROOT = r"/workspace/examples/PythonProject1/data_dectect_t3"
DATA_YAML_PATH = r"/workspace/examples/PythonProject1/data_dectect_t3/data.yaml"
RESULT_ROOT = r"/workspace/examples/PythonProject1/pest_experiment_results_yolov8m_optimized_final"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ====================== 超参数配置 ======================
TRAIN_EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 0.008
IMG_SIZE = 960
CONF_THRESHOLD = 0.18
IOU_THRESHOLD = 0.38
WARMUP_EPOCHS = 5
PATIENCE = 8
WEIGHT_DECAY = 0.0001

AUGMENT_STRATEGY = {
    "hsv_h": 0.005,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.4,
    "shear": 1.0,
    "perspective": 0.0001,
    "flipud": 0.1,
    "fliplr": 0.5,
    "mosaic": 0.8,
    "mixup": 0.0,
    "copy_paste": 0.0
}

# 创建目录
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "model"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "plots"), exist_ok=True)

# 固定随机种子
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# ====================== 工具函数 ======================
def parse_and_validate_yaml():
    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError(f"data.yaml文件不存在：{DATA_YAML_PATH}")

    import yaml as pyyaml
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        yaml_data = pyyaml.safe_load(f)

    required_fields = ["path", "train", "val", "nc", "names"]
    for field in required_fields:
        if field not in yaml_data:
            raise ValueError(f"data.yaml缺少字段：{field}")

    yaml_data["path"] = DATA_ROOT
    train_img_path = os.path.join(yaml_data["path"], yaml_data["train"])
    val_img_path = os.path.join(yaml_data["path"], yaml_data["val"])
    
    if not os.path.exists(train_img_path):
        raise FileNotFoundError(f"训练集路径不存在：{train_img_path}")
    if not os.path.exists(val_img_path):
        raise FileNotFoundError(f"验证集路径不存在：{val_img_path}")
    
    train_imgs = glob.glob(os.path.join(train_img_path, "*.[jp][pn]g"))
    val_imgs = glob.glob(os.path.join(val_img_path, "*.[jp][pn]g"))
    
    print(f"\n data.yaml验证通过！")
    print(f"  数据集信息：")
    print(f"   - 类别数：{yaml_data['nc']} | 类别：{yaml_data['names']}")
    print(f"   - 训练集：{len(train_imgs)}张 | 验证集：{len(val_imgs)}张")
    
    if len(train_imgs) < 50:
        warnings.warn(f" 训练集数量过少（{len(train_imgs)}张），建议补充数据！")
    
    return DATA_YAML_PATH, yaml_data["nc"], yaml_data["names"]

def check_annotation_quality():
    train_label_path = os.path.join(DATA_ROOT, "labels", "train")
    val_label_path = os.path.join(DATA_ROOT, "labels", "val")
    
    problematic_labels = []
    
    for label_path in [train_label_path, val_label_path]:
        if not os.path.exists(label_path):
            warnings.warn(f" 标签目录不存在：{label_path}")
            continue
            
        label_files = glob.glob(os.path.join(label_path, "*.txt"))
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        problematic_labels.append(f"{label_file}: 行{line_num+1} - 标注字段不足")
                        continue
                    
                    cls_id = int(parts[0])
                    if cls_id < 0:
                        problematic_labels.append(f"{label_file}: 行{line_num+1} - 类别ID为负数")
                    
                    coords = list(map(float, parts[1:5]))
                    for coord in coords:
                        if coord < 0 or coord > 1:
                            problematic_labels.append(f"{label_file}: 行{line_num+1} - 坐标超出0-1范围")
                            
            except Exception as e:
                problematic_labels.append(f"{label_file}: 解析错误 - {str(e)}")
    
    if problematic_labels:
        print(f"\n 发现{len(problematic_labels)}个标注问题（前10个）：")
        for problem in problematic_labels[:10]:
            print(f"   {problem}")
    else:
        print("\n  标注质量检查通过！")

def get_model_parameters(model):
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        return {
            "总参数(M)": round(total_params / 1e6, 2),
            "可训练参数(M)": round(trainable_params / 1e6, 2)
        }
    except Exception as e:
        print(f"  计算参数失败：{e}")
        return {"总参数(M)": 0, "可训练参数(M)": 0}

# ====================== 核心训练函数 ======================
def train_yolov8m_optimized(config_path):
    print("\n 开始YOLOv8m专属优化训练（适配ultralytics 8.4.19）")
    print("="*60)
    
    # 加载模型
    model = YOLO("yolov8m.pt")
    param_info = get_model_parameters(model)
    print(f"\n  模型参数信息：")
    print(f"   - 总参数：{param_info['总参数(M)']}M")
    print(f"   - 可训练参数：{param_info['可训练参数(M)']}M")
    
    # 检查标注
    check_annotation_quality()
    
    # 训练配置
    train_config = {
        "data": config_path,
        "epochs": TRAIN_EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "lr0": LEARNING_RATE,
        "lrf": 0.005,
        "momentum": 0.937,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 8.5,
        "cls": 1.2,
        "dfl": 1.5,
        "kobj": 1.3,
        "patience": PATIENCE,
        "device": 0 if torch.cuda.is_available() else None,
        "project": RESULT_ROOT,
        "name": "yolov8m_training_final",
        "exist_ok": True,
        "seed": SEED,
        "conf": CONF_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "verbose": True,
        "plots": True,
        "save": True,
        "save_period": 5,
        "val": True,
        "cos_lr": True,
        "optimizer": "AdamW",
        "nbs": 32,
        "close_mosaic": 15,
        "hsv_h": AUGMENT_STRATEGY["hsv_h"],
        "hsv_s": AUGMENT_STRATEGY["hsv_s"],
        "hsv_v": AUGMENT_STRATEGY["hsv_v"],
        "degrees": AUGMENT_STRATEGY["degrees"],
        "translate": AUGMENT_STRATEGY["translate"],
        "scale": AUGMENT_STRATEGY["scale"],
        "shear": AUGMENT_STRATEGY["shear"],
        "perspective": AUGMENT_STRATEGY["perspective"],
        "flipud": AUGMENT_STRATEGY["flipud"],
        "fliplr": AUGMENT_STRATEGY["fliplr"],
        "mosaic": AUGMENT_STRATEGY["mosaic"],
        "mixup": AUGMENT_STRATEGY["mixup"],
        "copy_paste": AUGMENT_STRATEGY["copy_paste"],
    }
    
    # 开始训练
    start_time = time.time()
    print("\n🏋️ 开始训练...")
    train_results = model.train(**train_config)
    train_hours = (time.time() - start_time) / 3600
    
    # 保存最佳模型
    best_model_src = os.path.join(RESULT_ROOT, "yolov8m_training_final", "weights", "best.pt")
    best_model_dst = os.path.join(RESULT_ROOT, "model", "yolov8m_best_final.pt")
    
    if os.path.exists(best_model_src):
        import shutil
        shutil.copy2(best_model_src, best_model_dst)
        print(f"\n 最佳模型已保存至：{best_model_dst}")
        model_size_mb = os.path.getsize(best_model_dst) / 1024 / 1024
        print(f"  模型体积：{model_size_mb:.2f} MB")
    
    # 模型验证
    best_model = YOLO(best_model_dst)
    print("\n  开始模型验证...")
    val_results = best_model.val(
        data=config_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=0 if torch.cuda.is_available() else None,
        verbose=True,
        plots=True,
        save_json=False,
        max_det=300
    )
    
    # 推理速度测试
    val_img_dir = os.path.join(DATA_ROOT, "images", "val")
    val_img_paths = glob.glob(os.path.join(val_img_dir, "*.[jp][pn]g"))[:100]
    infer_times = []
    
    best_model.eval()
    with torch.no_grad():
        for img_path in tqdm(val_img_paths, desc="📈 推理速度测试"):
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            start = time.time()
            results = best_model(
                img, 
                imgsz=IMG_SIZE, 
                verbose=False, 
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                agnostic_nms=True,
                max_det=300
            )
            infer_times.append(time.time() - start)
    
    # 计算推理指标
    avg_infer_time = np.mean(infer_times) if infer_times else 0
    fps = 1 / avg_infer_time if avg_infer_time > 0 else 0

    # ====================== 关键修复：正确处理数组类型的指标 ======================
    # 1. 处理标量指标（直接转float）
    map50 = round(float(val_results.box.map50), 4)
    map50_95 = round(float(val_results.box.map), 4)
    
    # 2. 处理数组指标（先算平均值再转float）
    # 注意：需要先转numpy数组，再计算均值，避免维度错误
    mp = round(float(np.mean(val_results.box.mp.cpu().numpy())), 4)  # 平均精度
    mr = round(float(np.mean(val_results.box.mr.cpu().numpy())), 4)  # 平均召回
    f1 = round(float(np.mean(val_results.box.f1.cpu().numpy())), 4)  # F1分数
    conf = round(float(np.mean(val_results.box.conf.cpu().numpy())), 4)  # 平均置信度
    
    # ====================== 整理指标 ======================
    metrics = {
        "模型名称": "YOLOv8m (Optimized Final 8.4.19)",
        "训练轮数": TRAIN_EPOCHS,
        "训练时间(h)": round(train_hours, 2),
        "参数信息": param_info,
        "模型体积(MB)": round(model_size_mb, 2),
        "核心评估指标": {
            "mAP50": map50,
            "mAP50-95": map50_95,
            "平均精度(MP)": mp,
            "平均召回(MR)": mr,
            "F1分数": f1,
            "平均置信度": conf
        },
        "推理性能": {
            "平均推理时间(ms)": round(avg_infer_time * 1000, 2),
            "FPS": round(fps, 2)
        },
        "超参数配置": {
            "初始学习率": LEARNING_RATE,
            "批次大小": BATCH_SIZE,
            "图像尺寸": IMG_SIZE,
            "置信度阈值": CONF_THRESHOLD,
            "IOU阈值": IOU_THRESHOLD,
            "box损失权重": 8.5,
            "cls损失权重": 1.2,
            "kobj损失权重": 1.3
        }
    }
    
    # 保存指标
    metrics_path = os.path.join(RESULT_ROOT, "metrics", "yolov8m_optimized_final_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    metrics_flat = {
        "模型名称": metrics["模型名称"],
        "训练轮数": metrics["训练轮数"],
        "训练时间(h)": metrics["训练时间(h)"],
        "总参数(M)": metrics["参数信息"]["总参数(M)"],
        "模型体积(MB)": metrics["模型体积(MB)"],
        "mAP50": metrics["核心评估指标"]["mAP50"],
        "mAP50-95": metrics["核心评估指标"]["mAP50-95"],
        "平均召回": metrics["核心评估指标"]["平均召回(MR)"],
        "FPS": metrics["推理性能"]["FPS"],
        "平均推理时间(ms)": metrics["推理性能"]["平均推理时间(ms)"]
    }
    pd.DataFrame([metrics_flat]).to_csv(
        os.path.join(RESULT_ROOT, "metrics", "yolov8m_optimized_final_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("  YOLOv8m 最终优化训练完成（适配8.4.19）！")
    print("="*60)
    print(f"\n  最终评估结果：")
    print(f"   mAP50: {metrics['核心评估指标']['mAP50']} (目标: >0.80)")
    print(f"   mAP50-95: {metrics['核心评估指标']['mAP50-95']}")
    print(f"   平均召回率: {metrics['核心评估指标']['平均召回(MR)']}")
    print(f"   FPS: {metrics['推理性能']['FPS']}")
    print(f"\n  结果文件保存至：{RESULT_ROOT}")
    
    return metrics

# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        config_path, num_classes, class_names = parse_and_validate_yaml()
        final_metrics = train_yolov8m_optimized(config_path)
        print("\n  所有任务完成！")
    except Exception as e:
        print(f"\n 训练过程出错：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)