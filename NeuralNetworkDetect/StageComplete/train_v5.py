from ultralytics import YOLO

if __name__ == '__main__':

    # 加载 YOLOv11 预训练模型（103张数据可继续用 s 模型）
    model = YOLO('yolo11s.pt')

    # 开始训练
    results = model.train(
        data='datasets/v5/data.yaml',
        
        # 基础参数
        epochs=150,         # 103张数据，可适当减少轮数
        imgsz=640,          # 先用 640 测试
        batch=16,           # 保持 16
        device=0,
        workers=2,
        
        # 防止过拟合
        patience=35,        # 早停可适当收紧
        
        # 数据增强（进一步减少，数据量接近 100）
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
        mosaic=0.2,          # 继续减少（v2是0.3）
        mixup=0.05,          # 继续减少（v2是0.1）
        copy_paste=0.1,      # 继续减少（v2是0.2）

        # 正则化
        dropout=0.05,        # 数据增多，可减少 dropout

        # 学习率
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,

        # 其他
        rect=True,
        cos_lr=True,
        close_mosaic=10,

        project='runs/train',
        name='stage_complete_v5',
        exist_ok=True,

        # 保存设置
        save_period=10,
        plots=True,
        )
