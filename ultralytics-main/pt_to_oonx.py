from ultralytics import YOLO

# 加载 YOLO 模型（.pt 文件）
model = YOLO("F:/YOLO/ultralytics-main/runs/detect/train3/weights/best.pt")  # 替换为你的模型路径

# 导出模型为 .onnx 格式
model.export(format="onnx", opset=11, simplify=True)  # 使用 opset 11，可以根据需要调整 opset 版本
