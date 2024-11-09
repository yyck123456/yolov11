import cv2
import os

# 模型文件路径
model_xml = 'F:/YOLO/ultralytics-main/output/best.xml'
model_bin = 'F:/YOLO/ultralytics-main/output/best.bin'

# 加载模型
net = cv2.dnn.readNet(model_xml, model_bin)

# 使用默认后端和目标
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 文件夹路径
image_folder = "F:/YOLO/ultralytics-main/datasets/test"

# 遍历文件夹中的每个图像文件并进行推理
for image_file in os.listdir(image_folder):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):  # 检查是否为图像文件
        image_path = os.path.join(image_folder, image_file)

        # 读取图像
        img = cv2.imread(image_path)
        blob = cv2.dnn.blobFromImage(img, size=(416, 416), swapRB=True, crop=False)  # 调整为模型输入大小

        # 设置输入并进行前向推理
        net.setInput(blob)
        detections = net.forward()

        # 遍历推理结果并显示
        for detection in detections:
            for obj in detection:
                confidence = obj[5]  # 置信度
                if confidence > 0.5:  # 设定阈值
                    # 获取边界框坐标
                    x1, y1, x2, y2 = (obj[0:4] * [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绘制矩形框

        # 显示图像
        cv2.imshow('Detections', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
