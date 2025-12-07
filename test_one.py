import os
import shutil
from ultralytics import YOLO

# 1. 清除ultralytics缓存（避免读取旧配置）
cache_dir = r"C:\Users\15284\AppData\Roaming\Ultralytics\cache"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("✅ 已清除缓存")

# 2. 绝对路径指向修正后的data.yaml
yaml_path = r"D:\PycharmProjects\PythonProject1\dataone\data.yaml"

# 3. 加载模型（102类建议用yolov8m.pt，精度更高）
model = YOLO("yolov8m.pt")

# 4. 训练（CPU训练调小batch和epochs，先验证流程）
results = model.train(
    data=yaml_path,
    epochs=5,          # 先训练5轮验证流程（102类CPU训练慢）
    batch=2,           # 102类显存占用高，CPU设batch=2
    imgsz=640,
    device='cpu',
    lr0=0.001,         # 多类别建议降低学习率，避免过拟合
    save=True,
    val=True,
    plots=True,
    patience=50,
    amp=False,
    workers=0          # CPU训练设workers=0，避免多线程错误
)

print("✅ 训练启动成功！")