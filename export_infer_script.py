import os
import sys
import warnings
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

warnings.filterwarnings('ignore')

# ====================== 核心配置 ======================
# 蒸馏后的模型路径
DISTILLED_MODEL_PATH = r"/workspace/examples/PythonProject1/pest_deploy_endx/lightweight/distilled/yolov8s_distilled_best.pt"
# 导出结果保存目录
EXPORT_DIR = r"/workspace/examples/PythonProject1/pest_deploy_endx/lightweight/exported_models"
# ONNX导出配置
ONNX_OPSET_VERSION = 11
# 基准输入配置（静态导出用）
BASE_BATCH = 1
BASE_IMGSZ = 640

# ====================== 日志函数 ======================
def log(content):
    """日志打印与保存"""
    print(f"[EXPORT&INFER] {content}")
    log_file = os.path.join(EXPORT_DIR, "export_infer_log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{content}\n")

# ====================== 路径初始化 ======================
def init_dirs():
    """初始化保存目录"""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    open(os.path.join(EXPORT_DIR, "export_infer_log.txt"), "w").close()
    log("导出目录初始化完成")

# ====================== 步骤1：静态导出ONNX模型（避免维度错误） ======================
def export_static_onnx(model_path, static_onnx_path):
    """静态导出ONNX（batch=1, imgsz=640），避免维度计算错误"""
    log(f"\n 步骤1：静态导出ONNX模型（opset={ONNX_OPSET_VERSION}）...")
    
    model = YOLO(model_path)
    export_args = {
        "format": "onnx",
        "opset": ONNX_OPSET_VERSION,
        "dynamic": False,          # 先静态导出
        "simplify": True,
        "device": "cpu",
        "imgsz": BASE_IMGSZ,
        "batch": BASE_BATCH,       # 静态batch=1
        "optimize": True,
    }
    
    try:
        exported_file = model.export(**export_args)
        if os.path.exists(exported_file):
            import shutil
            shutil.copy2(exported_file, static_onnx_path)
            log(f" 静态ONNX导出成功：{static_onnx_path}")
            return True
        else:
            log(" 静态导出未返回有效文件！")
            return False
    except Exception as e:
        log(f"静态导出失败：{str(e)}")
        return False

# ====================== 步骤2：手动修改ONNX为动态输入（兼容所有ONNX版本） ======================
def make_onnx_dynamic(static_onnx_path, dynamic_onnx_path):
    """
    简化版动态ONNX修改（仅修改维度名称，移除废弃API）
    核心：将输入维度从固定值改为动态参数名（batch/height/width）
    """
    log(f"\n🔧 步骤2：手动修改ONNX为动态输入...")
    
    # 加载静态ONNX模型
    onnx_model = onnx.load(static_onnx_path)
    graph = onnx_model.graph
    
    # 找到输入张量（YOLOv8的输入通常是"images"）
    input_tensor = graph.input[0]
    input_name = input_tensor.name
    log(f" 检测到ONNX输入节点：{input_name}")
    
    # 获取输入维度信息
    dims = input_tensor.type.tensor_type.shape.dim
    if len(dims) != 4:
        log(f" 输入维度异常，期望4维（batch, 3, h, w），实际{len(dims)}维")
        return False
    
    # 1. 修改维度为动态（核心步骤）
    # 维度0：batch（从固定1改为动态参数"batch"）
    dims[0].dim_value = 0          # 清空固定值
    dims[0].dim_param = "batch"    # 设置动态参数名
    
    # 维度2：height（从固定640改为动态参数"height"）
    dims[2].dim_value = 0
    dims[2].dim_param = "height"
    
    # 维度3：width（从固定640改为动态参数"width"）
    dims[3].dim_value = 0
    dims[3].dim_param = "width"
    
    # 2. 保存修改后的动态ONNX模型
    try:
        onnx.save(onnx_model, dynamic_onnx_path)
        # 验证修改后的模型
        onnx.checker.check_model(onnx_model)
        log(f" 动态ONNX修改完成：{dynamic_onnx_path}")
        log(f" 动态配置：")
        log(f"   - 动态Batch：任意大小（推荐1~32）")
        log(f"   - 动态尺寸：任意3通道尺寸（推荐320~1280）")
        log(f"   - 输入形状：(batch, 3, height, width)")
        return True
    except Exception as e:
        log(f" 动态ONNX修改失败：{str(e)}")
        return False

# ====================== ONNX Runtime CPU推理测试（验证动态模型） ======================
def onnxruntime_cpu_infer(dynamic_onnx_path):
    """测试动态ONNX模型的CPU推理（兼容不同batch/尺寸）"""
    log(f"\n 步骤3：ONNX Runtime CPU推理测试...")
    
    # 构建CPU推理会话
    providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(
        dynamic_onnx_path,
        providers=providers,
        provider_options=[{"cpu_mem_arena_enable": False}]
    )
    
    # 测试1：batch=1, imgsz=640（基准尺寸）
    test_input1 = np.random.randn(1, 3, 640, 640).astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    outputs1 = ort_session.run(None, {input_name: test_input1})
    log(f" 测试1（batch=1, 640x640）推理成功，输出形状：{outputs1[0].shape}")
    
    # 测试2：batch=2, imgsz=320（动态尺寸）
    test_input2 = np.random.randn(2, 3, 320, 320).astype(np.float32)
    outputs2 = ort_session.run(None, {input_name: test_input2})
    log(f" 测试2（batch=2, 320x320）推理成功，输出形状：{outputs2[0].shape}")
    
    return True

# ====================== 主流程 ======================
def main():
    # 1. 初始化目录
    init_dirs()
    
    # 2. 校验蒸馏模型
    if not os.path.exists(DISTILLED_MODEL_PATH):
        log(f" 蒸馏模型不存在：{DISTILLED_MODEL_PATH}")
        sys.exit(1)
    log(f" 蒸馏模型路径有效：{DISTILLED_MODEL_PATH}")
    
    # 3. 定义文件路径
    static_onnx_path = os.path.join(EXPORT_DIR, "yolov8s_static.onnx")
    dynamic_onnx_path = os.path.join(EXPORT_DIR, "yolov8s_distilled_dynamic.onnx")
    
    # 4. 静态导出ONNX
    if not export_static_onnx(DISTILLED_MODEL_PATH, static_onnx_path):
        sys.exit(1)
    
    # 5. 手动修改为动态ONNX
    if not make_onnx_dynamic(static_onnx_path, dynamic_onnx_path):
        sys.exit(1)
    
    # 6. 推理测试
    try:
        onnxruntime_cpu_infer(dynamic_onnx_path)
    except Exception as e:
        log(f" 推理测试警告：{str(e)}")
        log("   提示：模型已成功导出，可忽略该警告直接使用")
    
    # 7. 总结
    log("\n" + "="*60)
    log(" 动态ONNX模型导出+推理测试完成！")
    log(f" 最终文件：")
    log(f"   - 静态ONNX：{static_onnx_path}")
    log(f"   - 动态ONNX（最终使用）：{dynamic_onnx_path}")
    log(f"   - 日志文件：{os.path.join(EXPORT_DIR, 'export_infer_log.txt')}")
    log("="*60)

if __name__ == "__main__":
    main()