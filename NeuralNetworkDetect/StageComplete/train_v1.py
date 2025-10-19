from ultralytics import YOLO

if __name__ == '__main__':

    # 加载 YOLOv11 预训练模型（小数据集用 s 模型）
    model = YOLO('yolo11s.pt')

    # 开始训练
    results = model.train(
        data='datasets/v1/data.yaml',
        
        # 基础参数
        epochs=200,         # 小数据集需要更多轮次
        imgsz=640,          # 先用 640 测试
        batch=8,            # 减小 batch
        device=0,
        workers=2,
        
        # 防止过拟合
        patience=50,        # 早停
        
        # 数据增强（小数据集关键）
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
        mosaic=0.5,          # 拼接增强
        mixup=0.2,           # 混合增强
        copy_paste=0.3,      # 复制粘贴

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
        name='stage_complete_v1',
        exist_ok=True,

        # 保存设置
        save_period=10,
        plots=True,
        )