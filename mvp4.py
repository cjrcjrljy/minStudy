from ultralytics import YOLO
import cv2
import detect_tools as tools
import paddlehub as hub
from PIL import ImageFont, Image
from paddleocr import PaddleOCR

yolo_model_path = r'D:\Study\college_course\da_2_xia\xiaoxueqi\firstWeek\FirstWeek\runs\detect\train5\weights\best.pt'
model = YOLO(yolo_model_path, task='detect')



reslut = model("TestFiles/1.mp4",show=True)