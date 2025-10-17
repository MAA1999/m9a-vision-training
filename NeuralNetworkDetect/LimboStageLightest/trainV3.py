from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/limbo_expanded/weights/best.pt')
    
    results = model.train(
        data='dataset/v3/data.yaml',
        epochs=80,          # 中等轮数
        imgsz=640,
        batch=16,
        lr0=0.001,         # 中等学习率
        project='runs/train',
        name='limbo_v3'
    )
