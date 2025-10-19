from ultralytics import YOLO

if __name__ == '__main__':

    # 加载 YOLOv11 预训练模型（247张数据，可考虑升级到 m 或继续用 s）
    model = YOLO('yolo11s.pt')  # 若显存足够可改为 'yolo11m.pt'

    # 开始训练
    results = model.train(
        data='datasets/v8/data.yaml',
        
        # 基础参数
        epochs=100,         # 247张数据，可减少轮数
        imgsz=640,          # 保持 640（小目标可考虑 960/1280）
        batch=-1,           # 自动选择最优 batch（约 batch=8）
        device=0,
        workers=2,          # Windows 下过多 workers 会复制缓存占满内存
        cache='disk',       # 避免非确定性，改用磁盘缓存
        
        # 防止过拟合
        patience=30,        # 早停继续收紧
        
        # 数据增强（大幅减少，接近中型数据集）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0,           # 不旋转
        perspective=0,       # 不做透视变换
        shear=0,             # 不剪切
        translate=0.1,       # 平移
        scale=0.5,           # 缩放
        flipud=0.0,          # 不上下翻转
        fliplr=0.0,          # 不左右翻转
        mosaic=0.1,          # 大幅减少（v5是0.2）
        mixup=0.0,           # 关闭（数据足够不需要）
        copy_paste=0.0,      # 关闭（数据足够不需要）

        # 正则化
        dropout=0.0,         # 关闭 dropout（数据量足够）

        # 学习率
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,

        # 其他
        rect=True,
        cos_lr=True,
        close_mosaic=10,

        project='runs/train',
        name='stage_complete_v8',
        exist_ok=True,

        # 保存设置
        save_period=10,
        plots=True,
        )
