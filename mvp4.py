from ultralytics import YOLO
import cv2
import detect_tools as tools
import paddlehub as hub
from PIL import ImageFont, Image
from paddleocr import PaddleOCR

yolo_model_path = r'models\best.pt'
model = YOLO(yolo_model_path, task='detect')



reslut = model("TestFiles/1.mp4",show=True)