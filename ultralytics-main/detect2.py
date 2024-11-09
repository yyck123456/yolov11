import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from ultralytics import YOLO


def create_voc_xml(filename, width, height, detections, output_path):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"  # 假设是RGB图像

    for detection in detections:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = detection["class"]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(detection["xmin"]))
        ET.SubElement(bndbox, "ymin").text = str(int(detection["ymin"]))
        ET.SubElement(bndbox, "xmax").text = str(int(detection["xmax"]))
        ET.SubElement(bndbox, "ymax").text = str(int(detection["ymax"]))

    # 保存为漂亮的XML格式
    xml_str = ET.tostring(root, encoding='utf-8')
    pretty_xml_as_string = parseString(xml_str).toprettyxml()

    with open(output_path, "w") as f:
        f.write(pretty_xml_as_string)


def main(image_folder, model_path, output_folder):
    model = YOLO(model_path)
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_file)
            results = model(image_path)

            # 获取图像尺寸
            width, height = results[0].orig_shape[1], results[0].orig_shape[0]

            # 收集检测结果
            detections = []
            for box in results[0].boxes:
                # 使用 .flatten() 将 box.xyxy 转换为一维列表，然后解包
                xmin, ymin, xmax, ymax = box.xyxy.flatten().tolist()

                detection = {
                    "class": model.names[int(box.cls)],  # 类别名称
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
                detections.append(detection)

            # 保存为XML文件
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.xml")
            create_voc_xml(image_file, width, height, detections, output_path)

            print(f"Processed and saved XML: {output_path}")


if __name__ == "__main__":
    # 直接设置路径
    image_folder = "test"
    model_path = "best.pt"
    output_folder = "output"

    main(image_folder, model_path, output_folder)
