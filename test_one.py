import os
import shutil
from ultralytics import YOLO



# 2. 绝对路径指向修正后的data.yaml


# 3. 加载模型（102类建议用yolov8m.pt，精度更高）
model = YOLO("yolov8m.pt")

# 4. 训练（CPU训练调小batch和epochs，先验证流程）
results = model.train(
    data="/workspace/examples/PythonProject1/data_dectect_t2/data.yaml",
    epochs=5,          # 先训练5轮验证流程（102类CPU训练慢）
    batch=16,           # 102类显存占用高，CPU设batch=2
    imgsz=640,
    device=0,
    lr0=0.001,         # 多类别建议降低学习率，避免过拟合
    save=True,
    val=True,
    plots=True,
    patience=50,
    amp=False,
    workers=1         # CPU训练设workers=0，避免多线程错误
)

print("  训练启动成功！")