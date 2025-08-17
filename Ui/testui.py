import sys
import os

# 添加父目录到Python路径，以便导入detect_tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QLabel, QHBoxLayout, QFrame, QGridLayout, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage, QMovie, QFont
from PyQt5.QtCore import Qt, QTimer, QTime
import cv2
from ultralytics import YOLO
import detect_tools as tools
import paddlehub as hub
from datetime import datetime
import json

# 导入后端系统
from parking_backend import ParkingBackend

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
        
        # 初始化后端系统和OCR
        self.parking_backend = ParkingBackend()
        self.ocr = hub.Module(name="ch_pp-ocrv3")
        
        # 添加信息显示区域
        self.info_display = None
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('智能停车场管理系统 - 车辆检测 (时间模拟: 1秒=1分钟)')
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
        self.time_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.time_label.setFixedHeight(50)
        right_layout.addWidget(self.time_label)
        
        # 停车场信息显示区域
        info_label = QLabel("停车场管理信息 (时间模拟: 1秒=1分钟)", self)
        info_label.setFont(QFont("Arial", 14, QFont.Bold))
        info_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(info_label)
        
        # 信息显示文本框
        self.info_display = QTextEdit(self)
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(300)
        self.info_display.setFont(QFont("Consolas", 10))
        right_layout.addWidget(self.info_display)
        
        # 统计信息显示
        self.stats_label = QLabel("统计信息加载中...", self)
        self.stats_label.setFont(QFont("Arial", 10))
        self.stats_label.setWordWrap(True)
        right_layout.addWidget(self.stats_label)
        
        # 当前在场车辆显示
        self.current_vehicles_label = QLabel("当前在场车辆: 0", self)
        self.current_vehicles_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(self.current_vehicles_label)
        
        right_frame.setLayout(right_layout)
        
        # 更新信息显示
        self.update_info_display()

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
        # 时间相关的代码已去除，不再清零

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
            self.enter_btn = QPushButton('检测到车辆', self)
            self.enter_btn.clicked.connect(self.detectVehicle)
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

    def detectVehicle(self):
        self.showLoading()
        now_img = tools.img_cvread(self.selected_file)  # BGR格式

        # YOLO检测
        yolo_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best.pt')
        model = YOLO(yolo_model_path, task='detect')
        results = model(self.selected_file)[0]
        location_list = results.boxes.xyxy.tolist()
        crop_imgs = []
        plate_numbers = []  # 存储识别到的车牌号
        
        if len(location_list) >= 1:
            location_list = [list(map(int, e)) for e in location_list]
            for each in location_list:
                x1, y1, x2, y2 = each
                cv2.rectangle(now_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cropImg = now_img[y1:y2, x1:x2]
                cropImg = cv2.resize(cropImg, (240, 80), interpolation=cv2.INTER_LINEAR)
                crop_imgs.append(cropImg)
                
                # OCR识别车牌号
                try:
                    result = self.ocr.recognize_text(images=[cropImg])
                    for ocr_result in result:
                        if ocr_result['data']:
                            text = ocr_result['data'][0]['text']
                            confidence = ocr_result['data'][0]['confidence']
                            if confidence > 0.7:  # 置信度阈值
                                plate_numbers.append(text)
                                print(f"识别到车牌: {text}, 置信度: {confidence:.2f}")
                except Exception as e:
                    print(f"OCR识别失败: {e}")
        
        # 处理识别到的车牌，使用虚拟时间
        self.process_plates(plate_numbers)
        
        self.displayLabeledImage(now_img)
        self.displayCropImgs(crop_imgs)
        self.hideLoading()
        
        # 更新信息显示
        self.update_info_display()

    def process_plates(self, plate_numbers):
        """处理识别到的车牌号，使用虚拟时间"""
        # 使用虚拟时间而不是真实系统时间
        virtual_time = self.getVirtualDateTime()
        
        for plate_number in plate_numbers:
            if plate_number:  # 确保车牌号不为空
                result = self.parking_backend.process_plate_recognition(plate_number, virtual_time)
                self.display_parking_result(result)
    
    def getVirtualDateTime(self):
        """获取虚拟时间对应的datetime对象"""
        from datetime import datetime, date
        # 获取当前虚拟时间
        current_minute = self.virtual_seconds
        current_time = self.start_time.addSecs(current_minute * 60)
        
        # 转换为datetime对象，使用今天的日期
        today = date.today()
        virtual_datetime = datetime.combine(
            today, 
            current_time.toPyTime()
        )
        return virtual_datetime

    def display_parking_result(self, result):
        """显示停车处理结果，使用虚拟时间显示"""
        message = result.get('message', '')
        action = result.get('action', '')
        
        # 使用虚拟时间显示而不是真实时间
        virtual_time_str = self.getCurrentVirtualTime()
        
        if action == '进入':
            display_text = f"[{virtual_time_str}] 🚗 进入: {result['plate_number']}"
        elif action == '驶出':
            duration = result.get('duration', '未知')
            display_text = f"[{virtual_time_str}] 🚙 驶出: {result['plate_number']} (停车时长: {duration})"
        else:
            display_text = f"[{virtual_time_str}] ⚠️ {message}"
        
        # 添加到显示区域
        if self.info_display:
            self.info_display.append(display_text)
            # 自动滚动到底部
            scrollbar = self.info_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def update_info_display(self):
        """更新信息显示"""
        if not self.info_display:
            return
            
        # 获取统计信息
        stats = self.parking_backend.get_statistics()
        stats_text = f"""统计信息:
• 总完成停车次数: {stats['total_completed_parkings']}
• 当前在场车辆: {stats['current_vehicles_count']}
• 平均停车时长: {stats['average_parking_duration']}
• 总识别次数: {stats['total_recognitions']}"""
        
        self.stats_label.setText(stats_text)
        
        # 更新当前在场车辆
        current_vehicles = self.parking_backend.get_current_vehicles()
        if current_vehicles:
            vehicles_text = f"当前在场车辆: {len(current_vehicles)}\n"
            for vehicle in current_vehicles:
                vehicles_text += f"• {vehicle['plate_number']}\n"
        else:
            vehicles_text = "当前在场车辆: 0"
        
        self.current_vehicles_label.setText(vehicles_text)

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
        # 不再清零virtual_seconds
        self.updateTimeDisplay()  # 刷新显示

    def updateTimeDisplay(self):
        # 1秒对应虚拟1分钟（秒模拟分钟）
        current_minute = self.virtual_seconds
        current_time = self.start_time.addSecs(current_minute * 60)
        self.time_label.setText(current_time.toString('HH:mm'))
        self.virtual_seconds += 1

    def getCurrentVirtualTime(self):
        """
        返回当前显示的虚拟时间字符串，格式 'HH:mm'
        """
        current_minute = self.virtual_seconds
        current_time = self.start_time.addSecs(current_minute * 60)
        return current_time.toString('HH:mm')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FilePickerWindow()
    window.show()
    sys.exit(app.exec_())