from ultralytics import YOLO

if __name__ == '__main__':

    # 加载 YOLOv11 预训练模型（小数据集用 s 模型）
    model = YOLO('yolo11s.pt')

    # 开始训练
    results = model.train(
        data='datasets/v2/data.yaml',
        
        # 基础参数
        epochs=150,         # 67张数据，轮数可适当减少
        imgsz=640,          # 先用 640 测试
        batch=16,           # 数据增多，可增加 batch
        device=0,
        workers=2,
        
        # 防止过拟合
        patience=40,        # 早停
        
        # 数据增强（适度减少，数据量增加了）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,           # 不旋转
        perspective=0,       # 不做透视变换
        shear=0,             # 不剪切
        translate=0.1,       # 平移
        scale=0.5,           # 缩放
        flipud=0.0,          # 不上下翻转
        fliplr=0.5,          # 左右翻转
        mosaic=0.3,          # 适度减少（原 0.5）
        mixup=0.1,           # 适度减少（原 0.2）
        copy_paste=0.2,      # 适度减少（原 0.3）

        # 正则化
        dropout=0.1,

        # 学习率
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,

        # 其他
        rect=True,
        cos_lr=True,
        close_mosaic=10,

        project='runs/train',
        name='stage_complete_v2',
        exist_ok=True,

        # 保存设置
        save_period=10,
        plots=True,
        )