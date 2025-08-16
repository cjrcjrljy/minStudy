import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QLabel, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
from ultralytics import YOLO
import detect_tools as tools

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
        # 重置所有显示区域
        self.resetDisplay()
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

    def resetDisplay(self):
        # 清空主图
        self.image_label.clear()
        self.image_label.setText("")
        # 清空车牌裁剪图
        for label in self.crop_img_labels:
            self.crop_area.removeWidget(label)
            label.deleteLater()
        self.crop_img_labels.clear()
        # 重置文件名标签
        self.label.setText('尚未选择图片')
        # 隐藏“进入停车场”按钮
        if self.enter_btn is not None:
            self.enter_btn.hide()

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
        进入停车场按钮槽函数。
        显示标注后的图片和裁剪放大的车牌图片
        """
        now_img = tools.img_cvread(self.selected_file)  # BGR格式

        # YOLO检测
        yolo_model_path = r'D:\Study\college_course\da_2_xia\xiaoxueqi\firstWeek\FirstWeek\runs\detect\train5\weights\best.pt'
        model = YOLO(yolo_model_path, task='detect')
        results = model(self.selected_file)[0]
        location_list = results.boxes.xyxy.tolist()
        crop_imgs = []
        if len(location_list) >= 1:
            location_list = [list(map(int, e)) for e in location_list]
            # 在原图上画框
            for each in location_list:
                x1, y1, x2, y2 = each
                cv2.rectangle(now_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 标注主图
                cropImg = now_img[y1:y2, x1:x2]
                cropImg = cv2.resize(cropImg, (240, 80), interpolation=cv2.INTER_LINEAR)
                crop_imgs.append(cropImg)
        # 显示标注主图
        self.displayLabeledImage(now_img)
        # 显示所有车牌大图
        self.displayCropImgs(crop_imgs)

    def displayLabeledImage(self, img):
        """
        显示标注（画框）后的图片到主界面
        img: OpenCV格式（BGR）
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytesPerLine = 3 * width
        qimg = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def displayCropImgs(self, cropImgList):
        # 清理之前的车牌图片标签
        for label in self.crop_img_labels:
            self.crop_area.removeWidget(label)
            label.deleteLater()
        self.crop_img_labels.clear()

        for cropImg in cropImgList:
            qimg = self.cvMatToQImage(cropImg)
            pixmap = QPixmap.fromImage(qimg)
            crop_label = QLabel(self)
            crop_label.setPixmap(pixmap)
            crop_label.setFixedSize(240, 80)
            crop_label.setAlignment(Qt.AlignCenter)
            self.crop_area.addWidget(crop_label)
            self.crop_img_labels.append(crop_label)

    def cvMatToQImage(self, cvImg):
        """OpenCV图片转QImage"""
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return qImg

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FilePickerWindow()
    window.show()
    sys.exit(app.exec_())