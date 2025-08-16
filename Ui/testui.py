import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QLabel, QHBoxLayout, QFrame, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QMovie, QFont
from PyQt5.QtCore import Qt, QTimer, QTime
import cv2
from ultralytics import YOLO
import detect_tools as tools

class FilePickerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_file = None
        self.enter_btn = None
        self.crop_img_labels = []
        self.loading_label = None
        self.loading_movie = None
        self.start_time = QTime(8, 0, 0)  # 默认08:00
        self.timer = None
        self.time_label = None
        self.virtual_seconds = 0  # 虚拟已过秒数
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

        # 加载动画label
        self.loading_label = QLabel(self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setFixedSize(50, 50)
        self.loading_label.setVisible(False)
        self.left_layout.addWidget(self.loading_label)

        # 车牌裁剪结果区域
        self.crop_area = QHBoxLayout()
        self.left_layout.addLayout(self.crop_area)

        left_frame.setLayout(self.left_layout)

        # 右侧区域
        right_frame = QFrame(self)
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setAlignment(Qt.AlignTop)

        # 时间显示区域
        self.time_label = QLabel(self)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.time_label.setFont(QFont("Arial", 20, QFont.Bold))  # 字体大小改为20
        self.time_label.setFixedHeight(50)                       # 高度也调小
        right_layout.addWidget(self.time_label)
        right_frame.setLayout(right_layout)

        main_layout.addWidget(left_frame)
        main_layout.addWidget(right_frame)

        self.setLayout(main_layout)
        self.initTimerDisplay()

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
        # 隐藏 loading 动画
        self.hideLoading()
        # 重置右上角时间
        self.setStartTime(self.start_time)  # 重新初始化时间
        self.virtual_seconds = 0

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

    def showLoading(self):
        self.loading_movie = QMovie("loading.gif")
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.setVisible(True)
        self.loading_movie.start()
        QApplication.processEvents()

    def hideLoading(self):
        if self.loading_movie:
            self.loading_movie.stop()
        self.loading_label.setVisible(False)

    def enterParking(self):
        self.showLoading()
        now_img = tools.img_cvread(self.selected_file)  # BGR格式

        # YOLO检测
        yolo_model_path = r'D:\Study\college_course\da_2_xia\xiaoxueqi\firstWeek\FirstWeek\runs\detect\train5\weights\best.pt'
        model = YOLO(yolo_model_path, task='detect')
        results = model(self.selected_file)[0]
        location_list = results.boxes.xyxy.tolist()
        crop_imgs = []
        if len(location_list) >= 1:
            location_list = [list(map(int, e)) for e in location_list]
            for each in location_list:
                x1, y1, x2, y2 = each
                cv2.rectangle(now_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cropImg = now_img[y1:y2, x1:x2]
                cropImg = cv2.resize(cropImg, (240, 80), interpolation=cv2.INTER_LINEAR)
                crop_imgs.append(cropImg)
        self.displayLabeledImage(now_img)
        self.displayCropImgs(crop_imgs)
        self.hideLoading()

    def displayLabeledImage(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytesPerLine = 3 * width
        qimg = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def displayCropImgs(self, cropImgList):
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
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return qImg

    # --------- 时间显示相关 ----------
    def initTimerDisplay(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTimeDisplay)
        self.setStartTime(self.start_time)
        self.timer.start(1000)  # 1s更新一次

    def setStartTime(self, qtime_or_str=None):
        """
        外部可调用，传入QTime或'HH:MM'字符串
        """
        if isinstance(qtime_or_str, QTime):
            self.start_time = qtime_or_str
        elif isinstance(qtime_or_str, str):
            try:
                h, m = map(int, qtime_or_str.split(":"))
                self.start_time = QTime(h, m, 0)
            except:
                self.start_time = QTime(8, 0, 0)
        else:
            self.start_time = QTime(8, 0, 0)
        self.virtual_seconds = 0
        self.updateTimeDisplay()  # 刷新显示

    def updateTimeDisplay(self):
        # 1秒对应虚拟1分钟
        current_minute = self.virtual_seconds
        current_time = self.start_time.addSecs(current_minute * 60)
        self.time_label.setText(current_time.toString('HH:mm'))  # 只显示时间，不加前缀
        self.virtual_seconds += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FilePickerWindow()
    window.show()
    sys.exit(app.exec_())