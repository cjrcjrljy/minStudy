#coding:utf-8
from ultralytics import YOLO
import torch

def main():
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("使用CPU训练")
    
    # 加载预训练模型
    model = YOLO("yolov8n.pt")
    
    # 使用GPU训练，增加batch size和添加更多训练参数
    if torch.cuda.is_available():
        # GPU训练参数 - 降低batch size避免显存不足
        results = model.train(
            data='datasets/PlateData/data.yaml',
            epochs=100,                    # 增加训练轮数
            batch=8,                       # 降低batch size（适合6GB显存）
            imgsz=640,                     # 图像尺寸
            device=device,                 # 指定设备
            workers=4,                     # 减少数据加载进程数
            project='runs/detect',         # 项目目录
            name='train_gpu',              # 训练名称
            exist_ok=True,                 # 允许覆盖已有的训练结果
            save=True,                     # 保存模型
            save_period=10,                # 每10个epoch保存一次模型
            cache=False,                   # 关闭缓存节省内存
            mixup=0.0,                     # 关闭mixup节省内存
            mosaic=1.0,                    # 数据增强：mosaic
            optimizer='AdamW',             # 优化器
            lr0=0.01,                      # 初始学习率
            lrf=0.01,                      # 最终学习率比例
            momentum=0.937,                # 动量
            weight_decay=0.0005,           # 权重衰减
            warmup_epochs=3.0,             # 热身轮数
            warmup_momentum=0.8,           # 热身动量
            box=7.5,                       # 边界框损失权重
            cls=0.5,                       # 分类损失权重
            dfl=1.5,                       # DFL损失权重
            verbose=True,                  # 显示详细训练进度
            plots=True,                    # 生成训练图表
        )
    else:
        # CPU训练参数（较保守的配置）
        results = model.train(
            data='datasets/PlateData/data.yaml',
            epochs=50,
            batch=4,
            imgsz=640,
            device=device,
            workers=4,
            project='runs/detect',
            name='train_cpu',
            exist_ok=True,
            verbose=True,                  # 显示详细训练进度
            plots=True,                    # 生成训练图表
        )
    
    print("训练完成！")
    
    # 验证模型
    print("开始验证模型...")
    metrics = model.val()
    print(f"验证结果: mAP50={metrics.box.map50:.3f}, mAP50-95={metrics.box.map:.3f}")
    
    # 将模型转为onnx格式（可选）
    # success = model.export(format='onnx')
    # if success:
    #     print("模型已导出为ONNX格式")

if __name__ == '__main__':
    main()