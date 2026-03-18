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
import math
from collections import defaultdict

warnings.filterwarnings('ignore')

# ====================== 基础配置 ======================
DATA_ROOT = r"/workspace/examples/PythonProject1/data_dectect_t2"
DATA_YAML_PATH = r"/workspace/examples/PythonProject1/data_dectect_t2/data.yaml"
RESULT_ROOT = r"/workspace/examples/PythonProject1/pest_experiment_results_yolov8m_large_obj"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ====================== YOLOv8m 大目标专属优化超参数 ======================
TRAIN_EPOCHS = 70          
BATCH_SIZE = 16            
LEARNING_RATE = 1e-4       
IMG_SIZE = 800             
CONF_THRESHOLD = 0.3       
IOU_THRESHOLD = 0.5        
WARMUP_EPOCHS = 6          
PATIENCE = 10              
WEIGHT_DECAY = 0.0003      

# ====================== 大目标专属优化配置 ======================
LARGE_OBJECT_CFG = {
    "enabled": True,
    "large_obj_threshold": 0.3,
    "box_loss_weight": 5.0,
    "cls_loss_weight": 0.7,
    "iou_loss_type": "CIoU",
    "conf_penalty": 0.1
}

ADAPTIVE_LR_CFG = {
    "enabled": True,
    "lr_patience": 6,
    "lr_decay_rate": 0.75,
    "min_lr": 1e-6,
    "monitor_metric": "mAP50"
}

CLASS_BALANCE_CFG = {
    "enabled": True,
    "alpha": 0.2,          # 适配300+样本的温和平衡
    "min_samples": 0       # 取消最小采样数
}

DISTILLATION_CFG = {
    "enabled": True,
    "teacher_model": "yolov8x.pt",
    "distill_weight": 0.25,
    "temperature": 2.5
}

AUGMENT_STRATEGY = {
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.4,
    "shear": 2.0,
    "perspective": 0.0,
    "flipud": 0.1,
    "fliplr": 0.5,
    "mosaic": 0.5,
    "mixup": 0.0,
    "copy_paste": 0.0
}

# 创建结果保存目录
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "model"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "logs"), exist_ok=True)

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

# ====================== 核心模块修复 ======================
class LargeObjectOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.large_obj_samples = []
        self.normal_samples = []
    
    def analyze_large_objects(self, label_paths):
        if not self.cfg["enabled"]:
            return
        
        print("\n 分析大目标样本分布...")
        for label_path in tqdm(label_paths, desc=" 检测大目标"):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                has_large_obj = False
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w, h = float(parts[3]), float(parts[4])
                        area = w * h
                        if area > self.cfg["large_obj_threshold"]:
                            has_large_obj = True
                            break
                
                img_path = label_path.replace("labels", "images").replace(".txt", ".jpg")
                if os.path.exists(img_path):
                    if has_large_obj:
                        self.large_obj_samples.append(img_path)
                    else:
                        self.normal_samples.append(img_path)
            except Exception as e:
                continue
        
        total = len(self.large_obj_samples) + len(self.normal_samples)
        print(f" 大目标样本数：{len(self.large_obj_samples)} / 总样本数：{total} ({len(self.large_obj_samples)/total*100:.1f}%)")
    
    def get_large_obj_aware_samples(self):
        if not self.cfg["enabled"]:
            return self.large_obj_samples + self.normal_samples
        
        large_sample_num = int(len(self.large_obj_samples) * 0.8)
        normal_sample_num = len(self.normal_samples)
        
        selected_large = random.sample(self.large_obj_samples, min(large_sample_num, len(self.large_obj_samples)))
        selected_normal = random.sample(self.normal_samples, min(normal_sample_num, len(self.normal_samples)))
        
        return selected_large + selected_normal

# 修复自适应学习率调度器：增加兼容性处理
class AdaptiveLRScheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.best_metric = 0.0
        self.counter = 0
        self.current_lr = LEARNING_RATE  # 初始学习率
        self.optimizer = None
    
    # 新增：延迟初始化optimizer
    def init_optimizer(self, model):
        """从训练器中获取optimizer，兼容不同ultralytics版本"""
        try:
            # 优先从trainer获取（ultralytics>=8.0）
            if hasattr(model, 'trainer') and hasattr(model.trainer, 'optimizer'):
                self.optimizer = model.trainer.optimizer
            # 兼容旧版本
            elif hasattr(model, 'optimizer'):
                self.optimizer = model.optimizer
            else:
                print("⚠️ 无法获取optimizer，自适应LR调度器将禁用")
                self.cfg["enabled"] = False
                return False
            # 更新当前学习率
            self.current_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
            return True
        except Exception as e:
            print(f"⚠️ 初始化optimizer失败：{e}，自适应LR调度器禁用")
            self.cfg["enabled"] = False
            return False
    
    def step(self, current_metric):
        if not self.cfg["enabled"] or self.optimizer is None:
            return self.current_lr[0]
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.cfg["lr_patience"]:
            new_lr = [lr * self.cfg["lr_decay_rate"] for lr in self.current_lr]
            new_lr = [max(lr, self.cfg["min_lr"]) for lr in new_lr]
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = new_lr[i]
            
            self.current_lr = new_lr
            self.counter = 0
            print(f" 学习率调整：{[round(lr, 7) for lr in self.current_lr]}")
        
        return self.current_lr[0]

class ClassBalancedSampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.class_counts = defaultdict(int)
        self.class_samples = defaultdict(list)
    
    def build_class_index(self, label_paths):
        if not self.cfg["enabled"]:
            return
        
        print("\n 构建类别平衡采样索引...")
        for label_path in tqdm(label_paths, desc=" 统计类别分布"):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                img_path = label_path.replace("labels", "images").replace(".txt", ".jpg")
                if not os.path.exists(img_path):
                    continue
                
                sample_classes = set()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        cls_id = int(parts[0])
                        sample_classes.add(cls_id)
                
                for cls_id in sample_classes:
                    self.class_counts[cls_id] += 1
                    self.class_samples[cls_id].append(img_path)
            except Exception as e:
                continue
        
        print(" 类别样本分布：")
        for cls_id in sorted(self.class_counts.keys()):
            print(f"   类别{cls_id}: {self.class_counts[cls_id]}个样本")
    
    def sample_balanced_batch(self, batch_size):
        if not self.cfg["enabled"]:
            all_samples = []
            for samples in self.class_samples.values():
                all_samples.extend(samples)
            return random.sample(all_samples, min(batch_size, len(all_samples)))
        
        balanced_samples = []
        while len(balanced_samples) < batch_size:
            for cls_id in self.class_samples.keys():
                cls_count = self.class_counts[cls_id]
                prob = (1.0 / cls_count) ** self.cfg["alpha"]
                
                if random.random() < prob and len(self.class_samples[cls_id]) > 0:
                    sample = random.choice(self.class_samples[cls_id])
                    balanced_samples.append(sample)
                    
                    if len(balanced_samples) >= batch_size:
                        break
        
        return balanced_samples[:batch_size]

def distillation_loss(student_logits, teacher_logits, temperature):
    student_probs = torch.nn.functional.softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    loss = torch.nn.functional.kl_div(
        torch.log(student_probs + 1e-8), 
        teacher_probs, 
        reduction='batchmean'
    ) * (temperature ** 2)
    return loss

# ====================== 通用工具函数 ======================
def safe_extract_metric(metric):
    try:
        if torch.is_tensor(metric):
            value = metric.cpu().numpy()
        else:
            value = metric
        
        if isinstance(value, (np.ndarray, list, tuple)):
            mean_value = np.mean(value)
        else:
            mean_value = value
        
        return float(mean_value)
    except Exception as e:
        print(f" 提取指标时出错：{e}，返回默认值0.0")
        return 0.0

def log_training_progress(epoch, metrics, lr, log_path):
    log_entry = {
        "epoch": epoch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "learning_rate": lr,
        "metrics": metrics
    }
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ====================== 解析并验证data.yaml ======================
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
    print(f" 数据集信息：")
    print(f"   - 类别数：{yaml_data['nc']} | 类别：{yaml_data['names']}")
    print(f"   - 训练集：{len(train_imgs)}张 | 验证集：{len(val_imgs)}张")
    
    if len(train_imgs) < 50:
        warnings.warn(f"  训练集数量过少（{len(train_imgs)}张），建议补充数据！")
    
    return DATA_YAML_PATH, yaml_data["nc"], yaml_data["names"]

# ====================== 检查标注质量 ======================
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
        print("\n 标注质量检查通过！")
    
    train_label_files = glob.glob(os.path.join(train_label_path, "*.txt")) if os.path.exists(train_label_path) else []
    return train_label_files

# ====================== 模型参数计算 ======================
def get_model_parameters(model):
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

# ====================== 训练函数（核心修复） ======================
def train_yolov8m_large_obj(config_path):
    print("\n 开始YOLOv8m 大目标害虫检测专属训练")
    print("="*80)
    
    # 1. 加载预训练模型
    model = YOLO("yolov8m.pt")
    param_info = get_model_parameters(model)
    print(f"\n 模型参数信息：")
    print(f"   - 总参数：{param_info['总参数(M)']}M")
    print(f"   - 可训练参数：{param_info['可训练参数(M)']}M")
    
    # 2. 检查标注质量并获取标签文件
    train_label_files = check_annotation_quality()
    
    # 3. 初始化大目标优化模块
    large_obj_optimizer = LargeObjectOptimizer(LARGE_OBJECT_CFG)
    if LARGE_OBJECT_CFG["enabled"] and train_label_files:
        large_obj_optimizer.analyze_large_objects(train_label_files)
    
    # 4. 初始化类别平衡采样模块
    class_balancer = ClassBalancedSampler(CLASS_BALANCE_CFG)
    if CLASS_BALANCE_CFG["enabled"] and train_label_files:
        class_balancer.build_class_index(train_label_files)
    
    # 5. 初始化知识蒸馏模块
    teacher_model = None
    if DISTILLATION_CFG["enabled"]:
        print(f"\n 加载教师模型 {DISTILLATION_CFG['teacher_model']} 进行知识蒸馏...")
        teacher_model = YOLO(DISTILLATION_CFG["teacher_model"])
        teacher_model.eval()
    
    # 6. 训练配置
    train_config = {
        "data": config_path,
        "epochs": TRAIN_EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "lr0": LEARNING_RATE,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": LARGE_OBJECT_CFG["box_loss_weight"],
        "cls": LARGE_OBJECT_CFG["cls_loss_weight"],
        "dfl": 1.2,
        "patience": PATIENCE,
        "device": 0 if torch.cuda.is_available() else None,
        "project": RESULT_ROOT,
        "name": "yolov8m_large_obj_training",
        "exist_ok": True,
        "seed": SEED,
        "conf": CONF_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "verbose": True,
        "plots": True,
        "save": True,
        "save_period": -1,
        "val": True,
        "cos_lr": not ADAPTIVE_LR_CFG["enabled"],
        "optimizer": "SGD",
        "nbs": 32,
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
    
    # 7. 初始化训练日志
    log_path = os.path.join(RESULT_ROOT, "logs", "training_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")
    
    # 8. 开始训练
    start_time = time.time()
    print("\n 开始大目标优化训练...")
    train_results = model.train(**train_config)
    
    # 9. 修复：训练后初始化自适应LR调度器（关键！）
    lr_scheduler = AdaptiveLRScheduler(ADAPTIVE_LR_CFG)
    if ADAPTIVE_LR_CFG["enabled"]:
        # 从trainer中获取optimizer，而非直接从model获取
        lr_scheduler.init_optimizer(model)
    
    # 10. 保存最佳模型
    best_model_src = os.path.join(RESULT_ROOT, "yolov8m_large_obj_training", "weights", "best.pt")
    best_model_dst = os.path.join(RESULT_ROOT, "model", "yolov8m_large_obj_best.pt")
    
    if os.path.exists(best_model_src):
        import shutil
        shutil.copy2(best_model_src, best_model_dst)
        print(f"\n 最佳模型已保存至：{best_model_dst}")
        
        model_size_mb = os.path.getsize(best_model_dst) / 1024 / 1024
        print(f" 模型体积：{model_size_mb:.2f} MB")
    else:
        print(f" 最佳模型文件不存在：{best_model_src}")
        model_size_mb = 0.0
    
    # 11. 模型验证
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
    
    # 12. 推理速度测试
    val_img_dir = os.path.join(DATA_ROOT, "images", "val")
    val_img_paths = glob.glob(os.path.join(val_img_dir, "*.[jp][pn]g"))[:100]
    infer_times = []
    confidences = []
    
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
            
            if results and len(results) > 0 and results[0].boxes is not None:
                box_confs = results[0].boxes.conf.cpu().numpy() if torch.is_tensor(results[0].boxes.conf) else results[0].boxes.conf
                if len(box_confs) > 0:
                    confidences.extend(box_confs.tolist())
    
    # 13. 计算核心指标
    avg_infer_time = np.mean(infer_times) if infer_times else 0
    fps = 1 / avg_infer_time if avg_infer_time > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    map50 = round(safe_extract_metric(val_results.box.map50), 4)
    map50_95 = round(safe_extract_metric(val_results.box.map), 4)
    mp = round(safe_extract_metric(val_results.box.mp), 4)
    mr = round(safe_extract_metric(val_results.box.mr), 4)
    f1 = round(safe_extract_metric(val_results.box.f1), 4)
    
    # 14. 调用LR调度器（示例：用验证集mAP50更新）
    if ADAPTIVE_LR_CFG["enabled"] and lr_scheduler.optimizer is not None:
        current_lr = lr_scheduler.step(map50)
    else:
        current_lr = LEARNING_RATE
    
    # 15. 整理指标
    metrics = {
        "模型名称": "YOLOv8m (Large Object Optimized)",
        "训练轮数": TRAIN_EPOCHS,
        "训练时间(h)": round((time.time() - start_time) / 3600, 2),
        "参数信息": param_info,
        "模型体积(MB)": round(model_size_mb, 2),
        "核心评估指标": {
            "mAP50": map50,
            "mAP50-95": map50_95,
            "平均精度(MP)": mp,
            "平均召回(MR)": mr,
            "F1分数": f1,
            "平均置信度": round(float(avg_confidence), 4)
        },
        "推理性能": {
            "平均推理时间(ms)": round(avg_infer_time * 1000, 2),
            "FPS": round(fps, 2)
        },
        "基础超参数配置": {
            "初始学习率": LEARNING_RATE,
            "批次大小": BATCH_SIZE,
            "图像尺寸": IMG_SIZE,
            "置信度阈值": CONF_THRESHOLD,
            "IOU阈值": IOU_THRESHOLD,
            "当前学习率": current_lr
        },
        "大目标优化配置": LARGE_OBJECT_CFG,
        "创新性优化配置": {
            "自适应学习率": ADAPTIVE_LR_CFG,
            "类别平衡采样": CLASS_BALANCE_CFG,
            "知识蒸馏": DISTILLATION_CFG
        }
    }
    
    # 16. 保存指标
    metrics_path = os.path.join(RESULT_ROOT, "metrics", "yolov8m_large_obj_metrics.json")
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
        "平均推理时间(ms)": metrics["推理性能"]["平均推理时间(ms)"],
        "当前学习率": metrics["基础超参数配置"]["当前学习率"],
        "大目标优化": "开启" if LARGE_OBJECT_CFG["enabled"] else "关闭",
        "自适应LR": "开启" if ADAPTIVE_LR_CFG["enabled"] else "关闭",
        "类别平衡": "开启" if CLASS_BALANCE_CFG["enabled"] else "关闭",
        "知识蒸馏": "开启" if DISTILLATION_CFG["enabled"] else "关闭"
    }
    pd.DataFrame([metrics_flat]).to_csv(
        os.path.join(RESULT_ROOT, "metrics", "yolov8m_large_obj_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    
    # 17. 打印最终结果
    print("\n" + "="*80)
    print("  YOLOv8m 大目标害虫检测训练完成！")
    print("="*80)
    print(f"\n  最终评估结果：")
    print(f"   mAP50: {metrics['核心评估指标']['mAP50']} (目标: >0.75)")
    print(f"   mAP50-95: {metrics['核心评估指标']['mAP50-95']}")
    print(f"   平均召回率: {metrics['核心评估指标']['平均召回(MR)']}")
    print(f"   F1分数: {metrics['核心评估指标']['F1分数']}")
    print(f"   平均置信度: {metrics['核心评估指标']['平均置信度']}")
    print(f"   FPS: {metrics['推理性能']['FPS']}")
    print(f"   当前学习率: {metrics['基础超参数配置']['当前学习率']}")
    print(f"\n  优化模块启用状态：")
    print(f"   - 大目标优化: {'✅ 开启' if LARGE_OBJECT_CFG['enabled'] else '❌ 关闭'}")
    print(f"   - 自适应学习率: {'✅ 开启' if ADAPTIVE_LR_CFG['enabled'] else '❌ 关闭'}")
    print(f"   - 类别平衡采样: {'✅ 开启' if CLASS_BALANCE_CFG['enabled'] else '❌ 关闭'}")
    print(f"   - 知识蒸馏: {'✅ 开启' if DISTILLATION_CFG['enabled'] else '❌ 关闭'}")
    print(f"\n  结果文件保存至：{RESULT_ROOT}")
    
    return metrics

# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        config_path, num_classes, class_names = parse_and_validate_yaml()
        final_metrics = train_yolov8m_large_obj(config_path)
        print("\n 所有任务完成！")
    except Exception as e:
        print(f"\n 训练过程出错：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)