import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from ultralytics import YOLO
from PIL import Image

image_folder = "F:/YOLO/ultralytics-main/datasets/test"
def create_xml(filename, image_size, detections, output_folder):
    """
    创建并保存 XML 文件。

    参数：
    - filename: 图像文件名
    - image_size: 图像尺寸 (width, height)
    - detections: 检测结果列表，每个检测包含类别、边界框坐标和置信度
    """
    # 创建 XML 文件结构
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = os.path.join(image_folder, filename)

    # 图像大小
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_size[0])
    ET.SubElement(size, "height").text = str(image_size[1])
    ET.SubElement(size, "depth").text = "3"

    # 对每个检测目标创建对应的 XML 标签
    for detection in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = detection['class']
        ET.SubElement(obj, "confidence").text = str(round(detection['confidence'], 2))

        # 边界框信息
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(detection['xmin'])
        ET.SubElement(bndbox, "ymin").text = str(detection['ymin'])
        ET.SubElement(bndbox, "xmax").text = str(detection['xmax'])
        ET.SubElement(bndbox, "ymax").text = str(detection['ymax'])

    # 保存 XML 文件
    xml_str = ET.tostring(annotation, 'utf-8')
    parsed_str = parseString(xml_str).toprettyxml(indent="  ")
    with open(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.xml"), "w") as f:
        f.write(parsed_str)


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run YOLO detection and save results to XML")
    parser.add_argument("-dir", "--input_dir", type=str, required=True, help="Input directory with images/videos")
    parser.add_argument("-model", "--model_path", type=str, required=True, help="Path to YOLO model .pt file")
    parser.add_argument("-out", "--output_dir", type=str, required=True, help="Directory to save XML outputs")

    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.input_dir):
        raise ValueError(f"输入目录不存在: {args.input_dir}")
    if not os.path.exists(args.model_path):
        raise ValueError(f"模型文件不存在: {args.model_path}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 YOLO 模型
    model = YOLO(args.model_path)

    # 遍历文件夹中的所有图片
    for image_file in os.listdir(args.input_dir):
        image_path = os.path.join(args.input_dir, image_file)

        # 使用 YOLO 进行推理
        results = model(image_path)

        # 获取图像尺寸
        with Image.open(image_path) as img:
            width, height = img.size

        # 解析结果
        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    'class': model.names[int(box.cls)],  # 类别名称
                    'confidence': float(box.conf),  # 置信度
                    'xmin': int(box.xyxy[0].item()),  # 左上角 x
                    'ymin': int(box.xyxy[1].item()),  # 左上角 y
                    'xmax': int(box.xyxy[2].item()),  # 右下角 x
                    'ymax': int(box.xyxy[3].item())   # 右下角 y
                }
                detections.append(detection)

        # 创建 XML 文件
        create_xml(image_file, (width, height), detections, args.output_dir)

    print("转换完成！所有图片的检测结果已保存为 XML 文件。")
