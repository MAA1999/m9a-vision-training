from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/limbo_v4/weights/best.pt')

    model.export(
        format='onnx',
        imgsz=640, # 和前面训练的尺寸一致
        simplify=True
    )
