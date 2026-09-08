import sys
import time
import os
import shutil
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QListWidget,
                             QListWidgetItem, QSplitter, QMessageBox, QDialog,
                             QFormLayout, QLineEdit, QSpinBox, QGroupBox,
                             QProgressDialog, QInputDialog, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
import logging
from datetime import datetime

# 配置日志：同时输出到控制台和文件
logging.basicConfig(
    level=logging.DEBUG,  # 日志级别：DEBUG（最详细）
    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",  # 日志格式
    handlers=[
        logging.FileHandler("face_recognition.log", encoding="utf-8"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)  # 创建日志实例
class FaceRecognitionWorker(QThread):
    """人脸识别工作线程，处理视频流和识别逻辑"""
    frame_processed = pyqtSignal(np.ndarray)  # 发送处理后的帧
    recognition_result = pyqtSignal(str, str)  # 发送识别结果 (名称, 结果)
    status_updated = pyqtSignal(str)  # 发送状态更新

    def __init__(self):
        super().__init__()
        self.running = False
        self.recognizing = False
        self.detecting = True  # 实时检测人脸
        self.collecting_data = False  # 是否正在收集数据
        self.data_count = 0  # 已收集的数据数量
        self.target_count = 20  # 目标收集数量
        self.person_name = ""  # 当前收集数据的人员名称

        # 加载人脸检测器
        self.face_cascade = self.load_face_detector()

        # 加载人脸识别模型
        self.recognizer = None
        self.label_map = {}  # 从ID映射到姓名
        self.load_recognition_model()

    def load_face_detector(self):
        """加载人脸检测器"""
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'  # 当前目录
        ]

        for path in cascade_paths:
            if os.path.exists(path):
                return cv2.CascadeClassifier(path)

        self.status_updated.emit("无法加载人脸检测模型")
        return None

    def load_recognition_model(self):
        """加载训练好的人脸识别模型"""
        try:
            if os.path.exists("face_recognizer.yml") and os.path.exists("label_map.txt"):
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read("face_recognizer.yml")

                # 加载标签映射
                self.label_map = {}
                with open("label_map.txt", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            id, name = line.strip().split(',', 1)
                            self.label_map[int(id)] = name
                self.status_updated.emit(f"模型加载成功，共识别 {len(self.label_map)} 人")
                return True
            else:
                self.status_updated.emit("未找到识别模型，请先训练模型")
        except Exception as e:
            self.status_updated.emit(f"模型加载失败: {str(e)}")

        self.recognizer = None
        return False

    def start_recognition(self):
        """开始一次人脸识别"""
        if self.recognizer is None:
            self.status_updated.emit("未加载识别模型，无法识别")
            return
        self.recognizing = True

    def start_data_collection(self, name, count=20):
        """开始收集人脸数据"""
        self.person_name = name
        self.target_count = count
        self.data_count = 0
        self.collecting_data = True
        self.status_updated.emit(f"开始收集 {name} 的人脸数据，共需 {count} 张")

        # 创建保存目录
        self.data_dir = os.path.join("dataset", name)
        if os.path.exists(self.data_dir):
            # 清空已有数据
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def stop_data_collection(self):
        """停止收集人脸数据"""
        self.collecting_data = False
        self.status_updated.emit(f"数据收集已停止，共收集 {self.data_count} 张图像")

    def toggle_detection(self):
        """切换实时检测状态"""
        self.detecting = not self.detecting
        status = "开启" if self.detecting else "关闭"
        self.status_updated.emit(f"{status}实时人脸检测")
        return self.detecting

    def recognize_face(self, gray_face):
        """识别单张人脸图像"""
        if self.recognizer is None:
            return "未知人员", "未加载模型"

        # 预测
        label_id, confidence = self.recognizer.predict(gray_face)

        # 置信度越低，匹配度越高（阈值设为80）
        if confidence < 80:
            name = self.label_map.get(label_id, "未知人员")
            return name, f"匹配成功 (置信度: {confidence:.1f})"
        else:
            return "未知人员", f"匹配失败 (置信度: {confidence:.1f})"

    def run(self):
        self.running = True
        # 打开摄像头
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.status_updated.emit("无法打开摄像头")
            self.running = False

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_updated.emit("无法获取视频帧")
                break

            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = []
            if self.face_cascade is not None and self.detecting:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

            # 绘制人脸框
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 处理数据收集
            if self.collecting_data and len(faces) > 0 and self.data_count < self.target_count:
                x, y, w, h = faces[0]
                # 截取并预处理人脸
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (100, 100))

                # 保存人脸图像
                save_path = os.path.join(self.data_dir, f"{self.data_count}.jpg")
                cv2.imwrite(save_path, face_img)

                self.data_count += 1
                cv2.putText(frame, f"收集: {self.data_count}/{self.target_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if self.data_count >= self.target_count:
                    self.collecting_data = False
                    self.status_updated.emit(f"{self.person_name} 的人脸数据收集完成，共 {self.data_count} 张")

            # 处理人脸识别
            if self.recognizing and len(faces) > 0:
                x, y, w, h = faces[0]
                gray_face = gray[y:y + h, x:x + w]
                gray_face = cv2.resize(gray_face, (100, 100))

                name, result = self.recognize_face(gray_face)
                self.recognition_result.emit(name, result)
                self.recognizing = False

                # 在画面上显示识别结果
                cv2.putText(frame, f"{name}: {result}",
                            (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 转换为RGB格式并发送
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_processed.emit(rgb_frame)

            # 控制帧率
            time.sleep(0.03)

        cap.release()
        self.status_updated.emit("摄像头已关闭")

    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()


class TrainModelDialog(QDialog):
    """模型训练对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练人脸识别模型")
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout()

        # 数据集路径设置
        path_layout = QHBoxLayout()
        self.dataset_path = QLineEdit("dataset")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_dataset)
        path_layout.addWidget(QLabel("数据集路径:"))
        path_layout.addWidget(self.dataset_path)
        path_layout.addWidget(browse_btn)

        # 模型参数设置
        params_group = QGroupBox("模型参数 (LBPH)")
        params_layout = QFormLayout()

        self.radius = QSpinBox()
        self.radius.setRange(1, 5)
        self.radius.setValue(1)
        params_layout.addRow("半径:", self.radius)

        self.neighbors = QSpinBox()
        self.neighbors.setRange(1, 10)
        self.neighbors.setValue(8)
        params_layout.addRow("邻居数:", self.neighbors)

        self.grid_x = QSpinBox()
        self.grid_x.setRange(1, 10)
        self.grid_x.setValue(8)
        params_layout.addRow("网格X:", self.grid_x)

        self.grid_y = QSpinBox()
        self.grid_y.setRange(1, 10)
        self.grid_y.setValue(8)
        params_layout.addRow("网格Y:", self.grid_y)

        params_group.setLayout(params_layout)

        # 按钮
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("开始训练")
        self.cancel_btn = QPushButton("取消")

        self.train_btn.clicked.connect(self.train_model)
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.cancel_btn)

        # 添加到主布局
        layout.addLayout(path_layout)
        layout.addWidget(params_group)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def browse_dataset(self):
        """浏览选择数据集路径"""
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.dataset_path.setText(path)

    def train_model(self):
        """训练模型"""
        dataset_path = self.dataset_path.text()

        if not os.path.exists(dataset_path):
            QMessageBox.warning(self, "错误", f"数据集路径不存在: {dataset_path}")
            return

        # 获取所有人员文件夹
        person_dirs = [d for d in os.listdir(dataset_path)
                       if os.path.isdir(os.path.join(dataset_path, d))]

        if not person_dirs:
            QMessageBox.warning(self, "错误", "数据集中未找到人员文件夹")
            return

        # 创建进度对话框
        progress = QProgressDialog("正在训练模型...", "取消", 0, len(person_dirs), self)
        progress.setWindowTitle("训练中")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)

        try:
            # 准备训练数据
            faces = []
            labels = []
            label_id = {}
            current_id = 0

            for i, person_name in enumerate(person_dirs):
                progress.setValue(i)
                progress.setLabelText(f"正在处理: {person_name}")
                QApplication.processEvents()

                if progress.wasCanceled():
                    QMessageBox.information(self, "取消", "训练已取消")
                    return

                # 分配唯一ID
                if person_name not in label_id:
                    label_id[person_name] = current_id
                    current_id += 1
                label = label_id[person_name]

                # 处理该人员的所有图像
                person_path = os.path.join(dataset_path, person_name)
                image_files = [f for f in os.listdir(person_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                for image_file in image_files:
                    image_path = os.path.join(person_path, image_file)

                    # 使用OpenCV读取图像并转换为灰度（替代PIL）
                    try:
                        # 读取图像
                        img = cv2.imread(image_path)
                        if img is None:
                            raise Exception("无法读取图像")

                        # 转换为灰度图
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # 确保图像尺寸正确
                        gray_img = cv2.resize(gray_img, (100, 100))

                        faces.append(gray_img)
                        labels.append(label)
                    except Exception as e:
                        print(f"处理图像 {image_path} 失败: {e}")

            # 检查是否有足够的数据
            if len(faces) < 2:
                QMessageBox.warning(self, "错误", "训练数据不足，请至少收集两个人的人脸数据")
                return

            # 创建并训练识别器
            progress.setLabelText("正在训练模型...")
            recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=self.radius.value(),
                neighbors=self.neighbors.value(),
                grid_x=self.grid_x.value(),
                grid_y=self.grid_y.value()
            )
            recognizer.train(faces, np.array(labels))

            # 保存模型和标签映射
            recognizer.save("face_recognizer.yml")
            with open("label_map.txt", "w", encoding="utf-8") as f:
                for name, id in label_id.items():
                    f.write(f"{id},{name}\n")

            progress.setValue(len(person_dirs))
            QMessageBox.information(self, "成功",
                                    f"模型训练完成，共训练 {len(label_id)} 个人\n模型已保存至 face_recognizer.yml")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败: {str(e)}")


class DataCollectionDialog(QDialog):
    """数据收集对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("收集人脸数据")
        self.setGeometry(300, 300, 300, 150)

        layout = QVBoxLayout()

        # 表单布局
        form_layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("请输入人员姓名")
        form_layout.addRow("人员姓名:", self.name_edit)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(5, 50)
        self.count_spin.setValue(20)
        form_layout.addRow("收集图像数量:", self.count_spin)

        # 按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始收集")
        self.cancel_btn = QPushButton("取消")

        self.start_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)

        # 添加到主布局
        layout.addLayout(form_layout)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_parameters(self):
        """返回收集参数"""
        return {
            "name": self.name_edit.text().strip(),
            "count": self.count_spin.value()
        }


class SettingsDialog(QDialog):
    """设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setGeometry(300, 300, 300, 250)

        # 获取父窗口的设置
        self.parent = parent
        self.capture_dir = "./captures"
        if parent and hasattr(parent, 'capture_dir'):
            self.capture_dir = parent.capture_dir

        layout = QVBoxLayout()

        # 人脸检测设置
        face_group = QGroupBox("人脸检测设置")
        face_layout = QFormLayout()

        self.scale_factor = QSpinBox()
        self.scale_factor.setRange(105, 200)  # 1.05到2.00
        self.scale_factor.setValue(110)  # 默认1.10
        self.scale_factor.setSuffix("%")
        face_layout.addRow("缩放因子:", self.scale_factor)

        self.min_neighbors = QSpinBox()
        self.min_neighbors.setRange(1, 10)
        self.min_neighbors.setValue(5)
        face_layout.addRow("最小邻居数:", self.min_neighbors)

        self.min_size = QSpinBox()
        self.min_size.setRange(10, 100)
        self.min_size.setValue(30)
        self.min_size.setSuffix("px")
        face_layout.addRow("最小人脸尺寸:", self.min_size)

        face_group.setLayout(face_layout)

        # 截图设置
        capture_group = QGroupBox("截图设置")
        capture_layout = QFormLayout()

        self.save_path_edit = QLineEdit(self.capture_dir)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_save_path)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.save_path_edit)
        path_layout.addWidget(browse_btn)
        capture_layout.addRow("截图保存路径:", path_layout)

        capture_group.setLayout(capture_layout)

        # 按钮
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.cancel_btn = QPushButton("取消")

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)

        # 添加到主布局
        layout.addWidget(face_group)
        layout.addWidget(capture_group)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def browse_save_path(self):
        """浏览选择保存路径"""
        path = QFileDialog.getExistingDirectory(self, "选择截图保存目录")
        if path:
            self.save_path_edit.setText(path)

    def get_settings(self):
        """返回设置值"""
        return {
            "scale_factor": self.scale_factor.value() / 100.0,
            "min_neighbors": self.min_neighbors.value(),
            "min_size": (self.min_size.value(), self.min_size.value()),
            "save_path": self.save_path_edit.text()
        }


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        # 初始化参数
        self.capture_dir = "./captures"
        self.records = []  # 存储识别记录
        self.detection_params = {
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": (30, 30)
        }

        # 初始化UI和工作线程
        self.init_ui()
        self.init_worker()

        # 创建截图保存目录
        os.makedirs(self.capture_dir, exist_ok=True)

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("人脸识别系统")
        self.setGeometry(100, 100, 1200, 800)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：人脸识别区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 状态标签
        self.status_label = QLabel("系统就绪")
        self.status_label.setFont(QFont("SimHei", 9))
        self.status_label.setStyleSheet("color: #3498db; padding: 4px;")

        # 人脸识别框
        self.face_frame = QLabel()
        self.face_frame.setAlignment(Qt.AlignCenter)
        self.face_frame.setMinimumSize(640, 480)
        self.face_frame.setStyleSheet("border: 2px solid #3498db; background-color: #f0f0f0;")
        self.face_frame.setText("等待摄像头启动...")

        # 按钮布局 - 第一行
        btn_layout1 = QHBoxLayout()

        # 开始识别按钮
        self.start_btn = QPushButton("开始识别")
        self.start_btn.setFont(QFont("SimHei", 10))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.start_btn.clicked.connect(self.start_recognition)

        # 截图按钮
        self.capture_btn = QPushButton("截图")
        self.capture_btn.setFont(QFont("SimHei", 10))
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.capture_btn.clicked.connect(self.capture_image)

        # 切换检测按钮
        self.toggle_detection_btn = QPushButton("关闭实时检测")
        self.toggle_detection_btn.setFont(QFont("SimHei", 10))
        self.toggle_detection_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        self.toggle_detection_btn.clicked.connect(self.toggle_detection)

        btn_layout1.addWidget(self.start_btn)
        btn_layout1.addWidget(self.capture_btn)
        btn_layout1.addWidget(self.toggle_detection_btn)

        # 按钮布局 - 第二行
        btn_layout2 = QHBoxLayout()

        # 收集数据按钮
        self.collect_data_btn = QPushButton("收集人脸数据")
        self.collect_data_btn.setFont(QFont("SimHei", 10))
        self.collect_data_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.collect_data_btn.clicked.connect(self.collect_face_data)

        # 训练模型按钮
        self.train_model_btn = QPushButton("训练识别模型")
        self.train_model_btn.setFont(QFont("SimHei", 10))
        self.train_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        self.train_model_btn.clicked.connect(self.train_model)

        # 设置按钮
        self.settings_btn = QPushButton("设置")
        self.settings_btn.setFont(QFont("SimHei", 10))
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #1abc9c;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #16a085;
            }
        """)
        self.settings_btn.clicked.connect(self.open_settings)

        btn_layout2.addWidget(self.collect_data_btn)
        btn_layout2.addWidget(self.train_model_btn)
        btn_layout2.addWidget(self.settings_btn)

        # 添加到左侧布局
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.face_frame)
        left_layout.addLayout(btn_layout1)
        left_layout.addLayout(btn_layout2)

        # 右侧：记录区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 记录标题
        records_title = QLabel("识别记录")
        records_title.setFont(QFont("SimHei", 12, QFont.Bold))
        records_title.setAlignment(Qt.AlignCenter)
        records_title.setStyleSheet("padding: 8px; background-color: #34495e; color: white;")

        # 记录列表
        self.records_list = QListWidget()
        self.records_list.setFont(QFont("SimHei", 10))
        self.records_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)

        # 添加到右侧布局
        right_layout.addWidget(records_title)
        right_layout.addWidget(self.records_list)

        # 添加到分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # 设置分割器初始大小
        splitter.setSizes([800, 400])

        main_layout.addWidget(splitter)

    def init_worker(self):
        """初始化工作线程"""
        self.worker = FaceRecognitionWorker()
        self.worker.frame_processed.connect(self.update_frame)
        self.worker.recognition_result.connect(self.handle_recognition_result)
        self.worker.status_updated.connect(self.update_status)
        self.worker.start()

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(f"状态: {message}")

    def update_frame(self, frame):
        """更新显示的帧"""
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.face_frame.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.face_frame.width(), self.face_frame.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.current_frame = frame  # 保存当前帧用于截图

    def start_recognition(self):
        """开始人脸识别"""
        self.worker.start_recognition()

    def toggle_detection(self):
        """切换实时检测状态"""
        state = self.worker.toggle_detection()
        if state:
            self.toggle_detection_btn.setText("关闭实时检测")
        else:
            self.toggle_detection_btn.setText("开启实时检测")

    def collect_face_data(self):
        """收集人脸数据"""
        # 显示数据收集对话框
        dialog = DataCollectionDialog(self)
        if dialog.exec_():
            params = dialog.get_parameters()
            name = params["name"]
            count = params["count"]

            if not name:
                QMessageBox.warning(self, "警告", "请输入人员姓名")
                return

            # 开始收集数据
            self.worker.start_data_collection(name, count)

    def train_model(self):
        """训练识别模型"""
        dialog = TrainModelDialog(self)
        if dialog.exec_():
            # 训练完成后重新加载模型
            self.worker.load_recognition_model()

    def handle_recognition_result(self, name, result):
        """处理识别结果"""
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存记录
        self.records.append({
            "time": current_time,
            "name": name,
            "result": result,
            "screenshot": ""  # 截图路径，初始为空
        })

        # 更新记录列表
        self.update_records_list()

        # 显示提示
        QMessageBox.information(self, "识别完成", f"识别时间: {current_time}\n姓名: {name}\n结果: {result}")

    def update_records_list(self):
        """更新记录列表"""
        self.records_list.clear()
        for record in self.records:
            item_text = f"时间: {record['time']}\n姓名: {record['name']}\n结果: {record['result']}"
            if record['screenshot']:
                item_text += f"\n截图: {os.path.basename(record['screenshot'])}"

            item = QListWidgetItem(item_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.records_list.addItem(item)

    def capture_image(self):
        """截图并保存"""
        if not hasattr(self, 'current_frame'):
            QMessageBox.warning(self, "警告", "没有可截图的画面")
            return

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(self.capture_dir, filename)

        # 保存图片
        try:
            # 转换回BGR格式保存
            bgr_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, bgr_frame)

            # 如果有选中的记录，将截图关联到该记录
            selected_items = [i for i in range(self.records_list.count())
                              if self.records_list.item(i).checkState() == Qt.Checked]

            if selected_items:
                self.records[selected_items[0]]['screenshot'] = filepath
                self.update_records_list()

            QMessageBox.information(self, "成功", f"截图已保存至:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存截图失败:\n{str(e)}")

    def open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self)
        if dialog.exec_():
            settings = dialog.get_settings()
            self.capture_dir = settings["save_path"]

            # 更新人脸检测参数
            self.detection_params = settings

            # 创建目录（如果不存在）
            os.makedirs(self.capture_dir, exist_ok=True)

            QMessageBox.information(self, "设置已保存", "设置已应用")

    def closeEvent(self, event):
        """关闭窗口时停止线程"""
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局字体，确保中文正常显示
    font = QFont("SimHei")
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
