# coding:utf-8
from ultralytics import YOLO
import cv2
import detect_tools as tools
import paddlehub as hub
from PIL import ImageFont, Image
import numpy as np
import re
import os
import glob

if __name__ == "__main__":
    # 设置图片文件夹路径
    img_folder = "TestFiles"  # 你的文件夹路径
    img_paths = glob.glob(os.path.join(img_folder, "*"))  # 获取所有文件
    img_paths = [p for p in img_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]  # 过滤图片

    # 加载模型
    fontC = ImageFont.truetype("Font/platech.ttf", 50, 0)
    yolo_model_path = r'D:\Study\college_course\da_2_xia\xiaoxueqi\firstWeek\FirstWeek\runs\detect\train5\weights\best.pt'
    model = YOLO(yolo_model_path, task='detect')
    ocr = hub.Module(name="ch_pp-ocrv3")

    for img_path in img_paths:
        print(f"\n【正在处理】{img_path}")
        now_img = tools.img_cvread(img_path)

        # YOLO检测
        results = model(img_path)[0]
        location_list = results.boxes.xyxy.tolist()
        if len(location_list) >= 1:
            location_list = [list(map(int, e)) for e in location_list]
            license_imgs = []
            for each in location_list:
                x1, y1, x2, y2 = each
                cropImg = now_img[y1:y2, x1:x2]
                cropImg = cv2.resize(cropImg, (240, 80), interpolation=cv2.INTER_LINEAR)
                license_imgs.append(cropImg)

            # OCR识别
            lisence_res = []
            conf_list = []
            for cropImg in license_imgs:
                result = ocr.recognize_text(images=[cropImg])
                for each in result:
                    text = each['data'][0]['text']
                    conf = each['data'][0]['confidence']
                    print("识别结果：", text)
                    print("置信度：", conf)
                    lisence_res.append(text)
                    conf_list.append(conf)

            # 绘制结果
            for text, box in zip(lisence_res, location_list):
                now_img = tools.drawRectBox(now_img, box, text, fontC)

        # 显示图片
        now_img = cv2.resize(now_img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("YOLOv8 Detection", now_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()