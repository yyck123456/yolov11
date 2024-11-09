import os
import numpy as np
import cv2
import openvino.runtime as ov
import matplotlib.pyplot as plt

# 初始化推理引擎
core = ov.Core()

# 加载模型
model_path = 'F:/YOLO/ultralytics-main/output/best.xml'
model = core.read_model(model_path)

# 编译模型
compiled_model = core.compile_model(model, "CPU")

# 获取输入和输出 blob 名称
input_name = model.input(0).get_any_name()
output_name = model.output(0).get_any_name()

# 准备图像的预处理
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (model.input(0).shape[2], model.input(0).shape[3]))
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = image.astype(np.float32)
    image /= 255.0  # 归一化
    return image

# 测试函数
def test_model(image_path):
    input_data = preprocess_image(image_path)
    input_data = np.expand_dims(input_data, axis=0)  # 增加 batch 维度

    # 进行推理
    result = compiled_model([input_data])
    output = result[output_name].squeeze()  # 根据实际输出格式处理

    print("Model output:", output)

    # 可视化图像
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB

    # 假设 output 是检测框和类标签的列表
    # 你需要根据你的模型输出格式调整这部分代码
    for detection in output:  # 这里需要根据输出格式调整
        # 示例：获取边框和置信度（假设输出包含这些信息）
        x1, y1, x2, y2, confidence, class_id = detection[:6]  # 根据输出格式调整
        if confidence > 0.5:  # 设定阈值
            # 绘制边框
            cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # 绘制标签
            label = f"Class {int(class_id)}: {confidence:.2f}"
            cv2.putText(original_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示处理后的图像
    plt.imshow(original_image)
    plt.axis('off')  # 关闭坐标轴
    plt.title(f"Processed Image: {image_path}")
    plt.show()

# 测试文件夹中的图像
test_folder = 'F:/YOLO/ultralytics-main/datasets/test'
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # 检查文件类型
        print(f'Testing image: {img_path}')
        test_model(img_path)
