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
# 增强后数据集根路径
DATA_ROOT = r"/workspace/examples/PythonProject1/data_dectect_t2"
# 已有data.yaml路径
DATA_YAML_PATH = r"/workspace/examples/PythonProject1/data_dectect_t2/data.yaml"
# 实验结果保存路径（专属YOLOv8m）
RESULT_ROOT = r"/workspace/examples/PythonProject1/pest_experiment_results_yolov8m_optimizedx"
# 随机种子
SEED = 42
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ====================== YOLOv8m 专属优化超参数 ======================
TRAIN_EPOCHS = 80      # YOLOv8m需要更多训练轮数充分收敛
BATCH_SIZE = 16             # 降低批次大小，适配YOLOv8m的参数量
LEARNING_RATE = 8e-5        # 精细调整的学习率
IMG_SIZE = 640
CONF_THRESHOLD = 0.25       # 降低置信度阈值减少漏检
IOU_THRESHOLD = 0.45        # 优化IOU阈值
WARMUP_EPOCHS = 8           # 更长的热身期
PATIENCE = 8               # 早停耐心值，防止过拟合
WEIGHT_DECAY = 0.0005       # 权重衰减防止过拟合

# YOLOv8m专属数据增强策略（针对害虫检测优化）
AUGMENT_STRATEGY = {
    "hsv_h": 0.02,          # 适度色调增强
    "hsv_s": 0.6,           # 饱和度增强
    "hsv_v": 0.4,           # 明度增强
    "degrees": 15.0,        # 旋转角度
    "translate": 0.15,      # 平移
    "scale": 0.6,           # 缩放范围
    "shear": 3.0,           # 剪切
    "perspective": 0.001,   # 透视变换
    "flipud": 0.2,          # 上下翻转
    "fliplr": 0.5,          # 左右翻转
    "mosaic": 1.0,          # Mosaic增强（对小目标友好）
    "mixup": 0.15,          # Mixup增强
    "copy_paste": 0.15      # Copy-paste增强
}

# 创建结果保存目录
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "model"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "plots"), exist_ok=True)

# 固定随机种子（保证结果可复现）
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# ====================== 通用工具函数（兼容张量/数组的指标处理） ======================
def safe_extract_metric(metric):
    """
    安全提取评估指标，兼容PyTorch张量和numpy数组/标量
    :param metric: 输入指标（张量/数组/标量）
    :return: 标量float值
    """
    try:
        # 如果是PyTorch张量
        if torch.is_tensor(metric):
            # 转到CPU → 转numpy → 计算均值（兼容标量/数组）
            value = metric.cpu().numpy()
        else:
            # 已经是numpy类型，直接使用
            value = metric
        
        # 计算均值（兼容数组/标量）
        if isinstance(value, (np.ndarray, list, tuple)):
            mean_value = np.mean(value)
        else:
            mean_value = value
        
        # 转为float并返回
        return float(mean_value)
    except Exception as e:
        print(f" 提取指标时出错：{e}，返回默认值0.0")
        return 0.0

# ====================== 解析并验证data.yaml ======================
def parse_and_validate_yaml():
    """验证并解析data.yaml，增加数据集质量检查"""
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
    
    # 验证数据集路径
    train_img_path = os.path.join(yaml_data["path"], yaml_data["train"])
    val_img_path = os.path.join(yaml_data["path"], yaml_data["val"])
    
    if not os.path.exists(train_img_path):
        raise FileNotFoundError(f"训练集路径不存在：{train_img_path}")
    if not os.path.exists(val_img_path):
        raise FileNotFoundError(f"验证集路径不存在：{val_img_path}")
    
    # 统计数据集信息
    train_imgs = glob.glob(os.path.join(train_img_path, "*.[jp][pn]g"))  # 只匹配图片文件
    val_imgs = glob.glob(os.path.join(val_img_path, "*.[jp][pn]g"))
    
    print(f"\n data.yaml验证通过！")
    print(f" 数据集信息：")
    print(f"   - 类别数：{yaml_data['nc']} | 类别：{yaml_data['names']}")
    print(f"   - 训练集：{len(train_imgs)}张 | 验证集：{len(val_imgs)}张")
    
    # 基础检查
    if len(train_imgs) < 50:
        warnings.warn(f"  训练集数量过少（{len(train_imgs)}张），建议补充数据！")
    
    return DATA_YAML_PATH, yaml_data["nc"], yaml_data["names"]

# ====================== 检查标注质量 ======================
def check_annotation_quality():
    """检查标注文件质量，找出异常标注"""
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
                    
                    # 检查类别ID是否有效
                    cls_id = int(parts[0])
                    if cls_id < 0:
                        problematic_labels.append(f"{label_file}: 行{line_num+1} - 类别ID为负数")
                    
                    # 检查坐标是否在0-1范围内
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
        print("\n 标注质量检查通过！")

# ====================== 模型参数计算 ======================
def get_model_parameters(model):
    """计算模型参数数量（单位：M）"""
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        return {
            "总参数(M)": round(total_params / 1e6, 2),
            "可训练参数(M)": round(trainable_params / 1e6, 2)
        }
    except Exception as e:
        print(f" 计算参数失败：{e}")
        return {"总参数(M)": 0, "可训练参数(M)": 0}

# ====================== 训练YOLOv8m（核心函数） ======================
def train_yolov8m_optimized(config_path):
    """专属优化的YOLOv8m训练函数（适配新版ultralytics API）"""
    print("\n 开始YOLOv8m专属优化训练")
    print("="*60)
    
    # 1. 加载预训练模型
    model = YOLO("yolov8m.pt")
    param_info = get_model_parameters(model)
    print(f"\n 模型参数信息：")
    print(f"   - 总参数：{param_info['总参数(M)']}M")
    print(f"   - 可训练参数：{param_info['可训练参数(M)']}M")
    
    # 2. 检查标注质量
    check_annotation_quality()
    
    # 3. 训练配置（YOLOv8m专属优化）
    train_config = {
        "data": config_path,
        "epochs": TRAIN_EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "lr0": LEARNING_RATE,          # 初始学习率
        "lrf": 0.01,                   # 最终学习率因子
        "momentum": 0.937,             # 动量
        "weight_decay": WEIGHT_DECAY,  # 权重衰减
        "warmup_epochs": WARMUP_EPOCHS, # 热身轮数
        "warmup_momentum": 0.8,        # 热身动量
        "warmup_bias_lr": 0.1,         # 热身偏置学习率
        "box": 7.5,                    # 框损失权重
        "cls": 0.5,                    # 分类损失权重
        "dfl": 1.5,                    # DFL损失权重
        "patience": PATIENCE,          # 早停耐心值
        "device": 0 if torch.cuda.is_available() else None,
        "project": RESULT_ROOT,
        "name": "yolov8m_training",
        "exist_ok": True,
        "seed": SEED,
        "conf": CONF_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "verbose": True,
        "plots": True,                 # 生成训练可视化图表
        "save": True,                  # 保存最佳模型
        "save_period": -1,             # 只保存最后一轮
        "val": True,                   # 每轮验证
        "cos_lr": True,                # 余弦学习率调度
        "optimizer": "SGD",            # SGD优化器更稳定
        "nbs": 64,                     # 标称批次大小
        # 数据增强参数
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
    
    # 4. 开始训练
    start_time = time.time()
    print("\n 开始训练...")
    train_results = model.train(**train_config)
    train_hours = (time.time() - start_time) / 3600
    
    # 5. 保存最佳模型
    best_model_src = os.path.join(RESULT_ROOT, "yolov8m_training", "weights", "best.pt")
    best_model_dst = os.path.join(RESULT_ROOT, "model", "yolov8m_best.pt")
    
    if os.path.exists(best_model_src):
        import shutil
        shutil.copy2(best_model_src, best_model_dst)
        print(f"\n 最佳模型已保存至：{best_model_dst}")
        
        # 计算模型大小
        model_size_mb = os.path.getsize(best_model_dst) / 1024 / 1024
        print(f" 模型体积：{model_size_mb:.2f} MB")
    else:
        # 防止模型文件不存在导致后续报错
        print(f" 最佳模型文件不存在：{best_model_src}")
        model_size_mb = 0.0
    
    # 6. 使用最佳模型验证
    best_model = YOLO(best_model_dst) if model_size_mb > 0 else model
    print("\n 开始模型验证...")
    val_results = best_model.val(
        data=config_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=0 if torch.cuda.is_available() else None,
        verbose=True,
        plots=True,
        save_json=False
    )
    
    # 7. 测试推理速度
    val_img_dir = os.path.join(DATA_ROOT, "images", "val")
    val_img_paths = glob.glob(os.path.join(val_img_dir, "*.[jp][pn]g"))[:100]
    infer_times = []
    confidences = []  # 手动收集置信度
    
    best_model.eval()
    with torch.no_grad():
        for img_path in tqdm(val_img_paths, desc=" 推理速度测试"):
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            start = time.time()
            results = best_model(img, imgsz=IMG_SIZE, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            infer_times.append(time.time() - start)
            
            # 手动收集置信度（替代val_results.box.conf）
            if results and len(results) > 0 and results[0].boxes is not None:
                box_confs = results[0].boxes.conf.cpu().numpy() if torch.is_tensor(results[0].boxes.conf) else results[0].boxes.conf
                if len(box_confs) > 0:
                    confidences.extend(box_confs.tolist())
    
    # 8. 计算核心指标
    avg_infer_time = np.mean(infer_times) if infer_times else 0
    fps = 1 / avg_infer_time if avg_infer_time > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0.0  # 手动计算平均置信度
    
    # 9. 整理所有指标（适配新版API，移除conf属性调用）
    # 标量指标直接提取
    map50 = round(safe_extract_metric(val_results.box.map50), 4)
    map50_95 = round(safe_extract_metric(val_results.box.map), 4)
    
    # 数组/张量指标安全提取（只保留存在的属性）
    mp = round(safe_extract_metric(val_results.box.mp), 4)       # 平均精度
    mr = round(safe_extract_metric(val_results.box.mr), 4)       # 平均召回
    f1 = round(safe_extract_metric(val_results.box.f1), 4)       # F1分数
    
    metrics = {
        "模型名称": "YOLOv8m (Optimized)",
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
            "平均置信度": round(float(avg_confidence), 4)  # 使用手动计算的置信度
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
            "IOU阈值": IOU_THRESHOLD
        }
    }
    
    # 10. 保存指标
    metrics_path = os.path.join(RESULT_ROOT, "metrics", "yolov8m_optimized_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    # 保存为CSV格式（便于查看）
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
        os.path.join(RESULT_ROOT, "metrics", "yolov8m_optimized_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    
    # 11. 打印最终结果
    print("\n" + "="*60)
    print("  YOLOv8m 优化训练完成！")
    print("="*60)
    print(f"\n  最终评估结果：")
    print(f"   mAP50: {metrics['核心评估指标']['mAP50']} (目标: >0.7)")
    print(f"   mAP50-95: {metrics['核心评估指标']['mAP50-95']}")
    print(f"   平均召回率: {metrics['核心评估指标']['平均召回(MR)']} (解决漏检问题)")
    print(f"   F1分数: {metrics['核心评估指标']['F1分数']}")
    print(f"   平均置信度: {metrics['核心评估指标']['平均置信度']}")
    print(f"   FPS: {metrics['推理性能']['FPS']}")
    print(f"\n  结果文件保存至：{RESULT_ROOT}")
    
    return metrics

# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        # 1. 解析验证配置文件
        config_path, num_classes, class_names = parse_and_validate_yaml()
        
        # 2. 训练优化版YOLOv8m
        final_metrics = train_yolov8m_optimized(config_path)
        
        print("\n 所有任务完成！")
        
    except Exception as e:
        print(f"\n 训练过程出错：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)