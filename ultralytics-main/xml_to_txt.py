import os
import xml.etree.ElementTree as ET

# 类别名称
classes = ['Header', 'Title', 'Text','Figure','Foot']  # 替换为你的类别名称

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_txt_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(output_txt_file, 'w') as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} " + " ".join([str(a) for a in bb]) + '\n')

# 示例：将XML文件转换为YOLO格式
xml_dir = "F:/voc_data/labels/valid1"
output_dir = "F:/voc_data/labels/valid"

for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        output_txt_file = os.path.join(output_dir, xml_file.replace(".xml", ".txt"))
        convert_annotation(os.path.join(xml_dir, xml_file), output_txt_file)
