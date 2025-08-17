#coding:utf-8
from ultralytics import YOLO
import cv2
import detect_tools as tools
from PIL import ImageFont
from paddleocr import PaddleOCR
import glob
import os

def get_license_result(ocr, image):
    """
    image:输入的车牌截取照片
    输出，车牌号与置信度
    """
    try:
        result = ocr.ocr(image, cls=True)
        if result and result[0] and len(result[0]) > 0:
            license_name, conf = result[0][0][1]
            if '·' in license_name:
                license_name = license_name.replace('·', '')
            return license_name, conf
        else:
            return None, None
    except Exception as e:
        print(f"OCR识别出错: {e}")
        return None, None

def process_image(img_path, model, ocr, fontC):
    """
    处理单张图片
    """
    print(f"正在处理图片: {img_path}")
    
    # 读取图片
    now_img = tools.img_cvread(img_path)
    if now_img is None:
        print(f"无法读取图片: {img_path}")
        return
    
    try:
        # 检测图片
        results = model(img_path)[0]
        
        # 检查是否有检测结果
        if results.boxes is None or len(results.boxes) == 0:
            print(f"在图片 {img_path} 中未检测到车牌")
            return
            
        location_list = results.boxes.xyxy.tolist()
        
        if len(location_list) >= 1:
            location_list = [list(map(int, e)) for e in location_list]
            print(f"检测到 {len(location_list)} 个车牌区域")
            
            # 截取每个车牌区域的照片
            license_imgs = []
            for each in location_list:
                x1, y1, x2, y2 = each
                cropImg = now_img[y1:y2, x1:x2]
                license_imgs.append(cropImg)
                
            # 车牌识别结果
            lisence_res = []
            conf_list = []
            for i, each in enumerate(license_imgs):
                license_num, conf = get_license_result(ocr, each)
                if license_num:
                    lisence_res.append(license_num)
                    conf_list.append(conf)
                    print(f"车牌 {i+1}: {license_num} (置信度: {conf:.2f})")
                else:
                    lisence_res.append('无法识别')
                    conf_list.append(0)
                    print(f"车牌 {i+1}: 无法识别")
                    
            # 在图片上标注结果
            for text, box in zip(lisence_res, location_list):
                now_img = tools.drawRectBox(now_img, box, text, fontC)
            
            # 调整显示尺寸
            now_img = cv2.resize(now_img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f"Detection - {os.path.basename(img_path)}", now_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"处理图片 {img_path} 时出错: {e}")

# 主程序
if __name__ == "__main__":
    print("正在初始化模型...")
    
    # 创建字体对象
    fontC = tools.create_font("Font/platech.ttf", 50)
    
    # 加载ocr模型 - 简化配置，使用默认模型
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        print("PaddleOCR 初始化成功")
    except Exception as e:
        print(f"PaddleOCR 初始化失败: {e}")
        exit(1)
    
    # 所需加载的模型目录
    model_path = 'models/best.pt'
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        exit(1)
        
    # 加载预训练模型
    try:
        model = YOLO(model_path, task='detect')
        print("YOLO模型加载成功")
    except Exception as e:
        print(f"YOLO模型加载失败: {e}")
        exit(1)
    
    # 查找当前目录下所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(ext))
        image_files.extend(glob.glob(ext.upper()))
    
    # 也检查子目录中的图片
    for root, dirs, files in os.walk('.'):
        for ext in image_extensions:
            pattern = os.path.join(root, ext)
            image_files.extend(glob.glob(pattern))
    
    # 去重并过滤
    image_files = list(set(image_files))
    image_files = [f for f in image_files if os.path.isfile(f)]
    
    if not image_files:
        print("当前目录下未找到任何图片文件")
        print("支持的图片格式: jpg, jpeg, png, bmp, tiff, tif")
    else:
        print(f"找到 {len(image_files)} 个图片文件:")
        for img_file in image_files:
            print(f"  - {img_file}")
        
        print("\n开始处理图片...")
        for img_path in image_files:
            process_image(img_path, model, ocr, fontC)
            
    print("处理完成！")