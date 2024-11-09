from ultralytics import YOLO

def train_model():
    # 加载模型
    model = YOLO("F:/YOLO/ultralytics-main/runs/detect/train/weights/best.pt")

    # 训练模型
    train_results = model.train(
        data="F:/YOLO/ultralytics-main/datasets/data.yml",  # 数据集 YAML 路径
        epochs=100,  # 训练轮次
        imgsz=640,  # 训练图像尺寸
        device="0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
    )

    # 评估模型在验证集上的性能
    metrics = model.val()

    # 打印评估结果
    print("Validation metrics:", metrics)

if __name__ == '__main__':
    train_model()
