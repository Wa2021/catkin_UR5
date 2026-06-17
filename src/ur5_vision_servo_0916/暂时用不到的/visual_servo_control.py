import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
import os


DEFAULT_YOLO_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../models/yolov8n.pt')
)

class VisualServoController:
    def __init__(self, model_path=DEFAULT_YOLO_MODEL_PATH, calibration_file='calibration_result1.2.1.json'):
        """
        初始化视觉伺服控制器
        :param model_path: YOLOv8模型路径
        :param calibration_file: 标定结果文件路径
        """
        self.model = YOLO(model_path)
        
        # 加载标定结果
        self.load_calibration(calibration_file)
        
        # 视觉伺服控制参数
        self.lambda_v = 0.5  # 速度增益
        self.desired_image_center = np.array([320, 240])  # 期望目标在图像中的位置
        self.z_desired = 0.3  # 期望的抓取距离（米）
        
    def load_calibration(self, calibration_file):
        """
        加载标定结果
        :param calibration_file: 标定结果文件路径
        """
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
            
        self.camera_matrix = np.array(calibration_data['camera_matrix'])
        self.dist_coeffs = np.array(calibration_data['dist_coeffs'])
        self.hand_eye_matrix = np.array(calibration_data['hand_eye_matrix'])
        
    def detect_objects(self, image):
        """
        使用YOLOv8检测目标
        :param image: RGB图像
        :return: 检测结果
        """
        # 图像去畸变
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        undist_image = cv2.undistort(image, self.camera_matrix, 
                                   self.dist_coeffs, None, newcameramtx)
        
        # 目标检测
        results = self.model(undist_image)
        return results[0]
        
    def compute_visual_servo_control(self, current_feature, desired_feature, z_current):
        """
        计算视觉伺服控制律
        :param current_feature: 当前特征点位置 [x, y]
        :param desired_feature: 期望特征点位置 [x, y]
        :param z_current: 当前深度值
        :return: 相机速度螺旋
        """
        # 将像素坐标转换为归一化相机坐标
        current_norm = self.pixel_to_normalized(current_feature)
        desired_norm = self.pixel_to_normalized(desired_feature)
        
        # 计算图像特征误差
        error = (current_norm - desired_norm).reshape(-1, 1)
        
        # 计算交互矩阵
        x, y = current_norm
        Z = z_current
        L = np.array([
            [-1/Z, 0, x/Z, x*y, -(1+x*x), y],
            [0, -1/Z, y/Z, 1+y*y, -x*y, -x]
        ])
        
        # 计算控制律
        v_camera = -self.lambda_v * np.linalg.pinv(L) @ error
        
        # 通过手眼矩阵转换到机器人基座标系
        v_robot = self.camera_velocity_to_robot_velocity(v_camera)
        return v_robot
        
    def pixel_to_normalized(self, pixel_coord):
        """
        将像素坐标转换为归一化相机坐标
        :param pixel_coord: 像素坐标 [u, v]
        :return: 归一化相机坐标 [x, y]
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        x = (pixel_coord[0] - cx) / fx
        y = (pixel_coord[1] - cy) / fy
        return np.array([x, y])
        
    def camera_velocity_to_robot_velocity(self, v_camera):
        """
        将相机速度转换为机器人末端执行器速度
        :param v_camera: 相机速度螺旋
        :return: 机器人末端执行器速度
        """
        # 提取旋转矩阵和平移向量
        R_ce = self.hand_eye_matrix[:3, :3]  # 相机到末端执行器的旋转
        t_ce = self.hand_eye_matrix[:3, 3]   # 相机到末端执行器的平移
        
        # 构建速度变换矩阵
        t_skew = np.array([
            [0, -t_ce[2], t_ce[1]],
            [t_ce[2], 0, -t_ce[0]],
            [-t_ce[1], t_ce[0], 0]
        ])
        
        V_ce = np.block([
            [R_ce, t_skew @ R_ce],
            [np.zeros((3, 3)), R_ce]
        ])
        
        # 转换速度
        v_robot = V_ce @ v_camera
        return v_robot
        
    def process_frame(self, rgb_image, depth_image):
        """
        处理一帧图像，返回视觉伺服控制命令
        :param rgb_image: RGB图像
        :param depth_image: 深度图像
        :return: 机器人控制命令, 目标边界框
        """
        # 目标检测
        results = self.detect_objects(rgb_image)
        
        if len(results.boxes) == 0:
            return None, None
            
        # 获取最大置信度的目标
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 计算目标中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        current_feature = np.array([center_x, center_y])
        
        # 获取深度信息
        depth = depth_image[center_y, center_x]
        z_current = depth / 1000.0  # 转换为米
        
        # 计算视觉伺服控制律
        v_robot = self.compute_visual_servo_control(
            current_feature,
            self.desired_image_center,
            z_current
        )
        
        return v_robot, (x1, y1, x2, y2) 
