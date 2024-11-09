import os
import argparse
from ultralytics import YOLO

def main(image_folder, model_path, output_folder):
    model = YOLO(model_path)
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_file)
            results = model(image_path)

            for i, result in enumerate(results):
                output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_result_{i}.png")
                result.save(output_path)

            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    # 直接设置路径
    image_folder = "test"
    model_path = "best.pt"
    output_folder = "output"

    main(image_folder, model_path, output_folder)
