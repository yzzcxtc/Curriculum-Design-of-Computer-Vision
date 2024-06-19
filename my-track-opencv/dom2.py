import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, \
    QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt


class TrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.trackers = cv2.legacy.MultiTracker_create()
        self.model = self.load_yolov5()  # 加载YOLOv5模型
        self.frame = None

    def initUI(self):
        self.setWindowTitle('基于YOLOv5和OpenCV的车辆检测与跟踪')

        # 创建主布局
        main_layout = QVBoxLayout()

        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label)

        # 创建打开视频按钮
        self.open_button = QPushButton('打开视频')
        self.open_button.clicked.connect(self.open_file)
        main_layout.addWidget(self.open_button)

        # 创建检测车辆按钮
        self.detect_button = QPushButton('检测车辆')
        self.detect_button.clicked.connect(self.detect_vehicles)
        main_layout.addWidget(self.detect_button)

        # 创建开始跟踪按钮
        self.start_button = QPushButton('开始跟踪')
        self.start_button.clicked.connect(self.start_tracking)
        main_layout.addWidget(self.start_button)

        # 设置主布局
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_yolov5(self):
        """
        从本地路径加载YOLOv5模型。

        torch.hub.load参数：
        - 'ultralytics/yolov5': 在GitHub上的YOLOv5仓库。
        - 'custom': 表示加载自定义模型，而不是预定义模型。
        - path='yolov5s.pt': YOLOv5模型权重文件的路径。
        - source='local': 指定从本地加载模型。
        """
        model = torch.hub.load(
            'D:\\桌面\\计算机视觉\\计算机视觉课程设计\\my-track-opencv\\yolov5-5.0',
            'custom',
            'D:\\桌面\\计算机视觉\\计算机视觉课程设计\\my-track-opencv\\yolov5s.pt',
            source='local'
        )
        return model

    def open_file(self):
        """
        使用文件对话框打开视频文件并初始化视频捕捉对象。
        """
        filename, _ = QFileDialog.getOpenFileName(self, '打开视频文件')
        if filename:
            self.cap = cv2.VideoCapture(filename)
            self.timer.stop()
            self.trackers = cv2.legacy.MultiTracker_create()

    def detect_vehicles(self):
        """
        使用YOLOv5模型在视频的第一帧检测车辆。
        将检测到的车辆添加到MultiTracker中。
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                results = self.model(frame)  # 进行车辆检测
                detections = results.xyxy[0].numpy()  # 获取检测结果
                print(f'Detected objects: {detections}')  # 打印检测到的对象

                for *xyxy, conf, cls in detections:
                    if self.model.names[int(cls)] == 'car':  # 检查检测到的对象是否是车辆
                        x, y, x2, y2 = map(int, xyxy)
                        w, h = x2 - x, y2 - y
                        tracker = cv2.legacy.TrackerCSRT_create()  # 创建 TrackerCSRT 实例
                        self.trackers.add(tracker, frame, (x, y, w, h))  # 将 TrackerCSRT 实例添加到 MultiTracker

                print(f'Trackers added: {self.trackers.getObjects()}')  # 打印添加到 MultiTracker 中的跟踪对象
                cv2.destroyAllWindows()
                print('检测完毕')

    def start_tracking(self):
        print('进入了1')
        if self.cap.isOpened() and len(self.trackers.getObjects()) > 0:
            print('进入了2')
            self.timer.start(30)
        else:
            print('Tracker not initialized or no objects detected')

    def update_frame(self):
        """
        更新视频帧，逐帧处理视频并在视频中绘制跟踪框。
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                success, boxes = self.trackers.update(frame)
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            else:
                self.cap.release()
                self.timer.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TrackerApp()
    ex.show()
    sys.exit(app.exec_())
