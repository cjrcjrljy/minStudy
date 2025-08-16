import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QLabel, QHBoxLayout, QFrame, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import cv2
from PIL import ImageFont
# from paddleocr import PaddleOCR
# from ultralytics import YOLO

class FilePickerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_file = None
        self.enter_btn = None
        self.crop_img_labels = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('图片选择器')
        self.showMaximized()

        main_layout = QHBoxLayout(self)

        # 左侧区域
        left_frame = QFrame(self)
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_frame.setFixedWidth(400)

        self.left_layout = QVBoxLayout(left_frame)
        self.left_layout.setAlignment(Qt.AlignTop)

        self.label = QLabel('尚未选择图片', self)
        self.left_layout.addWidget(self.label)

        self.btn = QPushButton('选择图片', self)
        self.btn.clicked.connect(self.showFileDialog)
        self.left_layout.addWidget(self.btn)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(350, 350)
        self.left_layout.addWidget(self.image_label)

        # 车牌裁剪结果区域
        self.crop_area = QHBoxLayout()
        self.left_layout.addLayout(self.crop_area)

        left_frame.setLayout(self.left_layout)

        # 右侧区域暂为空
        right_frame = QFrame(self)
        right_frame.setFrameShape(QFrame.StyledPanel)

        main_layout.addWidget(left_frame)
        main_layout.addWidget(right_frame)

        self.setLayout(main_layout)

    def showFileDialog(self):
        options = QFileDialog.Options()
        file_filter = "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", file_filter, options=options
        )
        if file_path:
            self.selected_file = file_path
            self.label.setText(f'已选择图片: {file_path}')
            self.displayImage(file_path)
            self.showEnterButton()

    def displayImage(self, file_path):
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.clear()
            self.image_label.setText("无法加载图片")

    def showEnterButton(self):
        if self.enter_btn is None:
            self.enter_btn = QPushButton('进入停车场', self)
            self.enter_btn.clicked.connect(self.enterParking)
            self.left_layout.addWidget(self.enter_btn)
        else:
            self.enter_btn.show()

    def enterParking(self):
        """
        进入停车场按钮槽函数：
        1. 读取图片路径
        2. YOLO检测车牌区域并裁剪
        3. 每个车牌裁剪图片显示在窗口
        """
        img_path = self.selected_file
        if not img_path:
            return

        now_img = cv2.imread(img_path)

        # TODO: 按你的模型路径和API加载YOLO模型
        # yolo_model_path = "你的模型路径"
        # model = YOLO(yolo_model_path, task='detect')

        # 检测
        # results = model(img_path)[0]
        # location_list = results.boxes.xyxy.tolist()
        img_path = "TestFiles/016015625-88_90-298&486_503&565-499&550_298&565_298&495_503&486-0_0_3_27_27_24_30_24-108-194.jpg"
        now_img = tools.img_cvread(img_path)

        fontC = ImageFont.truetype("Font/platech.ttf", 50, 0)
        # 加载ocr模型
        ocr = PaddleOCR(lang="ch")
        # YOLO模型路径
        yolo_model_path = r'D:\Study\college_course\da_2_xia\xiaoxueqi\firstWeek\FirstWeek\runs\detect\train5\weights\best.pt'
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
        # --- 以下是模拟结果，你换成你的检测结果即可 ---
        # 假设检测出两个区域，直接用固定坐标测试
        location_list = [
            [50, 100, 250, 180],
            [260, 120, 480, 200]
        ]
        # --- 结束模拟 ---

        # 清空之前的裁剪显示
        for label in self.crop_img_labels:
            self.crop_area.removeWidget(label)
            label.deleteLater()
        self.crop_img_labels.clear()

        if len(location_list) >= 1:
            license_imgs = []
            for each in location_list:
                x1, y1, x2, y2 = each
                cropImg = now_img[y1:y2, x1:x2]
                cropImg = cv2.resize(cropImg, (240, 80), interpolation=cv2.INTER_LINEAR)
                license_imgs.append(cropImg)

                # 显示在界面
                qimg = self.cvMatToQImage(cropImg)
                pixmap = QPixmap.fromImage(qimg)
                crop_label = QLabel(self)
                crop_label.setPixmap(pixmap)
                crop_label.setFixedSize(240, 80)
                crop_label.setAlignment(Qt.AlignCenter)
                self.crop_area.addWidget(crop_label)
                self.crop_img_labels.append(crop_label)

            # 车牌识别（按需接入OCR）
            # ocr = PaddleOCR(lang="ch")
            # result = ocr.ocr(cropImg)
            # TODO: 你的OCR逻辑...

    def cvMatToQImage(self, cvImg):
        """
        OpenCV图片转QImage
        """
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return qImg

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FilePickerWindow()
    window.show()
    sys.exit(app.exec_())