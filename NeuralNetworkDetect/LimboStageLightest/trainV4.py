from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/limbo_expanded/weights/best.pt')
    
    results = model.train(
        data='dataset/v4/data.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=1e-4,         # 更小学习率
        freeze=10,
        rect=True,
        workers=2,
        project='runs/train',
        name='limbo_v4'
    )
