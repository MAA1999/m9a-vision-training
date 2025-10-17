from ultralytics import YOLO

if __name__ == '__main__':

    # 加载 YOLOv11 预训练模型
    model = YOLO('yolo11n.pt')

    # 开始训练
    results = model.train(
        data='dataset/v1/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # GPU
        workers=2,
        project='runs/train',
        name='limbo_stage_lightest',
        exist_ok=True
    )