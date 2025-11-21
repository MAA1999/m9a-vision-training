from ultralytics import YOLO

if __name__ == '__main__':

    # 加载 YOLOv11 预训练模型（小数据集用 s 模型）
    model = YOLO('yolo11n.pt')

    # 开始训练
    results = model.train(
        data='dataset/v6/data.yaml',
        
        # 基础参数
        epochs=300,         # 增加训练轮次，让小样本类别有更多学习机会
        imgsz=640,
        batch=6,            # 减小 batch，增加梯度更新频率
        device=0,
        workers=2,
        
        # 防止过拟合
        patience=80,
        
        # 数据增强（针对小样本加强）
        hsv_h=0,          # 稍微增加色相变化，增加多样性
        hsv_s=0,           # 增加饱和度变化
        hsv_v=0,             # 亮度变化
        degrees=0,           # 不旋转
        perspective=0,       # 不做透视变换
        shear=0,             # 不剪切
        translate=0.15,      # 增加平移，帮助小样本泛化
        scale=0,             # 不缩放
        flipud=0.0,          # 不上下翻转
        fliplr=0.0,          # 不左右翻转
        mosaic=0.8,          # Mosaic：拼接4张图，让小样本更频繁出现
        mixup=0.3,           # Mixup：混合2张图，增加样本多样性
        copy_paste=0.5,      # 增加复制粘贴，让小样本重复出现

        # 正则化（减小防止小样本过早停止学习）
        # dropout=0.05,        # 降低 dropout

        # 学习率（小样本需要更小心的学习）
        # lr0=0.005,           # 降低初始学习率，更细致地学习
        # lrf=0.005,           # 降低最终学习率
        # warmup_epochs=5,     # 增加 warmup
        
        # 损失权重（提高分类损失权重）
        # cls=1.0,             # 增加分类损失权重（默认 0.5）
        # box=7.5,             # 保持边界框权重
        # dfl=1.5,             # 保持 DFL 权重

        # 其他
        rect=True,           # 矩形训练节省内存
        cos_lr=True,         # 余弦学习率
        close_mosaic=20,     # 更晚关闭 mosaic，让增强持续更久
        
        # 自动增强
        auto_augment='randaugment',  # 使用随机增强
        erasing=0.4,         # 随机擦除增强

        project='runs/train',
        name='limbo_v6',
        exist_ok=True,

        # 保存设置
        save_period=10,
        plots=True,
        )