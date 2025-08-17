# coding:utf-8
from ultralytics import YOLO
import cv2
import detect_tools as tools
import paddlehub as hub
from PIL import ImageFont, Image
from paddleocr import PaddleOCR
import numpy as np
import re


if __name__ == "__main__":
    # 需要检测的图片地址
    img_path = "D:/Code/py/yolo/datasets/PlateData/images/test/01-90_265-231&522_405&574-405&571_235&574_231&523_403&522-0_0_3_1_28_29_30_30-134-56.jpg"
    now_img = tools.img_cvread(img_path)

    fontC = ImageFont.truetype("Font/platech.ttf", 50, 0)
    # 加载ocr模型
    ocr = PaddleOCR(lang="ch")
    # YOLO模型路径
    yolo_model_path = r'models/best.pt'
    model = YOLO(yolo_model_path, task='detect')

    # 检测
    results = model(img_path)[0]
    location_list = results.boxes.xyxy.tolist()
    if len(location_list) >= 1:
        location_list = [list(map(int, e)) for e in location_list]
        license_imgs = []
        for each in location_list:
            x1, y1, x2, y2 = each
            cropImg = now_img[y1:y2, x1:x2]

            # 对车牌进行缩放，标准化宽高
            cropImg = cv2.resize(cropImg, (240, 80), interpolation=cv2.INTER_LINEAR)
            license_imgs.append(cropImg)
            cv2.imshow('crop_plate', cropImg)
            cv2.waitKey(500)

        # 车牌识别结果
        lisence_res = []
        conf_list = []

        ocr = hub.Module(name="ch_pp-ocrv3")
        result = ocr.recognize_text(images=[cropImg])
        print(result)
        for each in result:
            text = each['data'][0]['text']  # ✅ 注意这里是 [0]
            conf = each['data'][0]['confidence']
            print("识别结果：", text)
            print("置信度：", conf)
            # 去除小数点

            lisence_res.append(text)
            conf_list.append(conf)
        # 在图片上绘制识别结果
        for i, (text, box) in enumerate(zip(lisence_res, location_list)):
            now_img = tools.drawRectBox(now_img, box, text, fontC)




    now_img = cv2.resize(now_img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("YOLOv8 Detection", now_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()