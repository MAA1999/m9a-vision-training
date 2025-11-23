from ultralytics import YOLO

if __name__ == '__main__':

    # 加载 YOLOv11 预训练模型（换成更大的模型以提高检测能力）
    model = YOLO('yolo11s.pt')  # 从 n 换成 s，提高模型容量

    # 开始训练
    results = model.train(
        data='dataset/v8/data.yaml',
        
        # 基础参数
        epochs=300,         # 恢复更多轮次，让模型充分学习
        imgsz=640,
        batch=4,            # 减小 batch，更精细的梯度更新
        device=0,
        workers=2,
        
        # 防止过拟合
        patience=50,        # 适当延长耐心值
        
        # 数据增强 - 轻度增强，保持真实性
        hsv_h=0.01,         # 轻微色相变化
        hsv_s=0.3,          # 适度饱和度变化
        hsv_v=0.0,          # 不改变亮度！亮度是关键特征
        degrees=0,          # 不旋转
        perspective=0,      # 不做透视变换
        shear=0,            # 不剪切
        translate=0.1,      # 适度平移
        scale=0,            # 关闭缩放（
        flipud=0.0,         # 不上下翻转
        fliplr=0.0,         # 不左右翻转（保持原始特征）
        mosaic=1.0,         # 使用 mosaic 但在后期关闭
        mixup=0.0,          # 关闭 mixup
        copy_paste=0.0,     # 关闭 copy_paste

        # 正则化 - 降低以提高学习能力
        dropout=0.0,        # 关闭 dropout，让模型充分学习

        # 学习率 - 降低以更细致学习
        lr0=0.005,          # 降低初始学习率
        lrf=0.001,          # 更低的最终学习率
        warmup_epochs=5,    # 更长的预热
        
        # 损失权重 - 提高召回率（减少漏检）
        cls=0.5,            # 分类损失
        box=7.5,            # 边界框损失
        dfl=1.5,            # DFL 损失
        
        # 检测参数 - 关键！降低置信度阈值
        conf=None,          # 训练时使用默认
        iou=0.6,            # 降低 IoU 阈值，更容易检测（默认 0.7）

        # 其他
        rect=True,
        cos_lr=True,
        close_mosaic=10,    # 训练后期关闭 mosaic，学习真实特征
        
        # 关闭额外增强
        auto_augment=None,
        erasing=0.0,
        
        # 使用 AMP 混合精度加速
        amp=True,

        project='runs/train',
        name='limbo_v10',   # 新版本
        exist_ok=True,

        # 保存设置
        save_period=10,
        plots=True,
        )