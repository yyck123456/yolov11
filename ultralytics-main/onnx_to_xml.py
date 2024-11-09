from openvino.tools.mo import convert_model

# 指定 ONNX 模型路径
onnx_model_path = "F:\\YOLO\\ultralytics-main\\runs\\detect\\train3\\weights\\best.onnx"
output_dir = "F:\\YOLO\\ultralytics-main\\output"

# 将 ONNX 模型转换为 IR 格式并指定输出路径
ir_model = convert_model(onnx_model_path)

# 保存模型到指定目录
ir_model.serialize(f"{output_dir}\\best.xml", f"{output_dir}\\best.bin")

print("模型已成功转换并保存为 IR 格式！")
