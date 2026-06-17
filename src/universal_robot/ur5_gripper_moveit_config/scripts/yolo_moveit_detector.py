#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO+MoveIt集成检测器 - 眼在手避障抓取
在arm_grasp环境中运行，完成检测后自动停止相机数据流
"""

import rospy
import cv2
import numpy as np
import json
import threading
import struct
import os
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Empty
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

# YOLO导入
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("错误: ultralytics未安装，请在arm_grasp环境中安装")
    YOLO_AVAILABLE = False

class YOLOMoveItDetector:
    def __init__(self):
        rospy.init_node('yolo_moveit_detector')
        
        # 初始化
        self.bridge = CvBridge()
        self.detection_count = 0
        self.stable_detections = {}
        self.detection_lock = threading.Lock()
        
        # 参数配置
        default_model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../../models/yolov8n.pt')
        )
        model_path = rospy.get_param('~model_path', default_model_path)
        self.detection_confidence = rospy.get_param('~detection_confidence', 0.5)
        self.target_objects = rospy.get_param('~target_objects', ['keyboard'])
        self.obstacle_objects = rospy.get_param('~obstacle_objects', ['person', 'laptop', 'book', 'mouse', 'cell phone', 'cup', 'bottle', 'chair', 'tv', 'monitor'])
        self.enable_auto_freeze = rospy.get_param('~enable_auto_freeze', False)  # 默认关闭
        self.observation_time = rospy.get_param('~observation_time', 5.0)
        
        # 障碍物生成模式
        # "blacklist": 除了target_objects之外的所有检测物体都生成障碍物
        # "whitelist": 只有obstacle_objects列表中的物体才生成障碍物
        self.obstacle_mode = rospy.get_param('~obstacle_mode', 'blacklist')
        
        # YOLO模型初始化
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                rospy.loginfo(f"YOLO模型加载成功: {model_path}")
            except Exception as e:
                rospy.logerr(f"YOLO模型加载失败: {e}")
                self.model = None
        else:
            rospy.logerr("YOLO不可用，请确保在arm_grasp环境中运行")
            self.model = None
        
        # 相机数据存储
        self.latest_pointcloud = None
        self.camera_info = None
        self.is_observing = True
        self.start_time = rospy.Time.now()
        
        # ROS订阅者
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.pointcloud_sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        
        # ROS发布者
        self.detection_image_pub = rospy.Publisher('/yolo/detection_image', Image, queue_size=1)
        self.object_points_pub = rospy.Publisher('/yolo/object_points', String, queue_size=1)
        self.detection_results_pub = rospy.Publisher('/yolo/detection_results', String, queue_size=1)
        self.status_pub = rospy.Publisher('/yolo/status', String, queue_size=1)
        
        # 等待快照控制服务
        try:
            rospy.wait_for_service('/snapshot/freeze', timeout=5.0)
            self.freeze_service = rospy.ServiceProxy('/snapshot/freeze', Empty)
            rospy.loginfo("连接到快照控制服务成功")
        except rospy.ROSException:
            rospy.logwarn("快照控制服务不可用，将不会自动停止相机")
            self.freeze_service = None
        
        # 观察完成定时器
        if self.enable_auto_freeze:
            self.observation_timer = rospy.Timer(
                rospy.Duration(self.observation_time), 
                self.finish_observation, 
                oneshot=True
            )
        
        rospy.loginfo("YOLO-MoveIt检测器已启动")
        rospy.loginfo(f"障碍物模式: {self.obstacle_mode}")
        rospy.loginfo(f"目标物体: {self.target_objects}")
        if self.obstacle_mode == 'whitelist':
            rospy.loginfo(f"障碍物体: {self.obstacle_objects}")
        else:
            rospy.loginfo("障碍物体: 除目标物体外的所有检测物体")
        rospy.loginfo(f"自动冻结: {'启用' if self.enable_auto_freeze else '禁用'}")
        rospy.loginfo(f"观察时间: {self.observation_time}秒")
    
    def camera_info_callback(self, msg):
        """相机参数回调"""
        if self.is_observing:
            self.camera_info = msg
    
    def pointcloud_callback(self, msg):
        """点云数据回调"""
        if self.is_observing:
            # 验证点云数据完整性
            if self.validate_pointcloud(msg):
                self.latest_pointcloud = msg
            else:
                rospy.logdebug("点云数据验证失败，跳过此帧")
    
    def validate_pointcloud(self, pointcloud_msg):
        """验证点云数据的完整性"""
        try:
            # 检查基本信息
            if pointcloud_msg.width == 0 or pointcloud_msg.height == 0:
                return False
            
            # 检查数据大小是否匹配
            expected_size = pointcloud_msg.row_step * pointcloud_msg.height
            actual_size = len(pointcloud_msg.data)
            
            if actual_size < expected_size:
                rospy.logdebug(f"点云数据大小不匹配: 期望 {expected_size}, 实际 {actual_size}")
                return False
            
            # 检查点云字段
            required_fields = ['x', 'y', 'z']
            available_fields = [field.name for field in pointcloud_msg.fields]
            
            for field in required_fields:
                if field not in available_fields:
                    rospy.logdebug(f"缺少必需的点云字段: {field}")
                    return False
            
            return True
            
        except Exception as e:
            rospy.logdebug(f"点云验证错误: {e}")
            return False
    
    def image_callback(self, msg):
        """图像回调和YOLO检测"""
        if not self.is_observing or self.model is None:
            return
        
        try:
            # 转换ROS图像到OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # YOLO检测
            results = self.model(cv_image, verbose=False)
            
            # 处理检测结果
            detections = self.process_yolo_results(results[0], cv_image, msg.header)
            
            # 发布检测结果
            self.publish_detection_results(detections, msg.header)
            
            # 发布完整检测结果供点云过滤器使用
            self.publish_complete_detection_results(detections)
            
            # 发布标注图像
            if results[0].boxes is not None:
                annotated_img = results[0].plot()
                
                # 添加状态信息到图像
                self.add_status_overlay(annotated_img, detections)
                
                img_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
                img_msg.header = msg.header
                self.detection_image_pub.publish(img_msg)
            
            # 更新检测计数
            self.detection_count += 1
            
        except Exception as e:
            rospy.logerr(f"YOLO检测错误: {e}")
    
    def add_status_overlay(self, image, detections):
        """在图像上添加状态信息"""
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        remaining_time = max(0, self.observation_time - elapsed_time) if self.enable_auto_freeze else 0
        
        # 状态文本
        if self.enable_auto_freeze:
            if self.is_observing:
                status_text = f"观察中... 剩余时间: {remaining_time:.1f}s"
            else:
                status_text = "观察完成 - 相机已停止"
        else:
            status_text = "调试模式 - 自动冻结已禁用"
        
        # 统计信息
        target_count = sum(1 for d in detections if d['is_target'])
        obstacle_count = sum(1 for d in detections if d['is_obstacle'])
        
        # 绘制状态信息背景
        cv2.rectangle(image, (10, 10), (450, 120), (0, 0, 0), -1)
        
        # 绘制状态信息
        cv2.putText(image, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"模式: {self.obstacle_mode}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"目标物体: {target_count}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(image, f"障碍物: {obstacle_count}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(image, f"检测次数: {self.detection_count}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_yolo_results(self, result, image, header):
        """处理YOLO检测结果"""
        detections = []
        
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                # 获取检测信息
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # 创建所有感兴趣的类别列表（目标物体 + 障碍物体）
                interested_classes = set(self.target_objects + self.obstacle_objects)
                
                # 只处理感兴趣的类别，并且置信度足够高
                if class_name in interested_classes and confidence >= self.detection_confidence:
                    # 计算中心点
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # 获取3D坐标
                    world_point = self.get_3d_point(center_x, center_y)
                    
                    # 判断物体类型
                    is_target = class_name in self.target_objects
                    
                    # 根据模式判断是否为障碍物
                    if self.obstacle_mode == 'blacklist':
                        # 黑名单模式：除了目标物体外，其他都是障碍物
                        is_obstacle = not is_target
                    else:
                        # 白名单模式：只有在obstacle_objects列表中的才是障碍物
                        is_obstacle = class_name in self.obstacle_objects
                    
                    detection = {
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center_2d': [center_x, center_y],
                        'point_3d': world_point,
                        'is_target': is_target,
                        'is_obstacle': is_obstacle,
                        'header': {
                            'frame_id': header.frame_id,
                            'stamp': {
                                'secs': header.stamp.secs,
                                'nsecs': header.stamp.nsecs
                            }
                        }
                    }
                    
                    detections.append(detection)
                    
                    # 记录稳定的检测结果
                    with self.detection_lock:
                        obj_key = f"{class_name}_{i}"
                        self.stable_detections[obj_key] = detection
        
        return detections
    
    def get_3d_point(self, u, v):
        """从2D像素坐标获取3D世界坐标"""
        if self.latest_pointcloud is None:
            return None
        
        try:
            # 检查像素坐标是否在有效范围内
            if u < 0 or v < 0 or u >= self.latest_pointcloud.width or v >= self.latest_pointcloud.height:
                rospy.logdebug(f"像素坐标超出范围: ({u}, {v}), 点云尺寸: {self.latest_pointcloud.width}x{self.latest_pointcloud.height}")
                return None
            
            # 验证点云数据完整性
            expected_data_size = self.latest_pointcloud.row_step * self.latest_pointcloud.height
            actual_data_size = len(self.latest_pointcloud.data)
            
            if actual_data_size < expected_data_size:
                rospy.logdebug(f"点云数据不完整: 期望{expected_data_size}字节，实际{actual_data_size}字节")
                return None
            
            # 计算在点云数据中的索引位置
            point_step = self.latest_pointcloud.point_step
            row_step = self.latest_pointcloud.row_step
            data_offset = v * row_step + u * point_step
            
            # 检查数据偏移是否超出缓冲区范围
            if data_offset + 12 > actual_data_size:  # xyz各4字节，共12字节
                rospy.logdebug(f"数据偏移超出范围: {data_offset} + 12 > {actual_data_size}")
                return None
            
            # 安全地从点云中获取对应像素的3D坐标
            points = list(pc2.read_points(
                self.latest_pointcloud, 
                field_names=("x", "y", "z"), 
                skip_nans=True, 
                uvs=[(u, v)]
            ))
            
            if points and len(points) > 0:
                point = points[0]
                # 检查3D坐标是否有效（不是NaN或无穷大）
                if not (np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]) or
                        np.isinf(point[0]) or np.isinf(point[1]) or np.isinf(point[2])):
                    return {'x': float(point[0]), 'y': float(point[1]), 'z': float(point[2])}
                else:
                    rospy.logdebug(f"3D坐标无效: ({point[0]}, {point[1]}, {point[2]})")
                    return None
            else:
                rospy.logdebug(f"未找到有效的3D点: ({u}, {v})")
                return None
                
        except struct.error as e:
            rospy.logwarn(f"获取3D坐标时缓冲区错误: {e}")
            return None
        except Exception as e:
            rospy.logwarn(f"获取3D坐标失败: {e}")
            return None
    
    def publish_detection_results(self, detections, header):
        """发布检测结果"""
        # 发布障碍物坐标供MoveIt使用
        obstacle_points = []
        target_points = []
        
        for detection in detections:
            if detection['point_3d']:
                if detection['is_obstacle']:
                    obstacle_points.append({
                        'class_name': detection['class_name'],
                        'point_3d': detection['point_3d'],
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'header': detection['header']
                    })
                elif detection['is_target']:
                    target_points.append({
                        'class_name': detection['class_name'],
                        'point_3d': detection['point_3d'],
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'header': detection['header']
                    })
        
        # 发布障碍物点
        if obstacle_points:
            points_msg = String()
            points_msg.data = json.dumps(obstacle_points)
            self.object_points_pub.publish(points_msg)
        
        # 发布状态信息
        status_info = {
            'is_observing': self.is_observing,
            'detection_count': self.detection_count,
            'target_objects_found': len(target_points),
            'obstacles_found': len(obstacle_points),
            'elapsed_time': (rospy.Time.now() - self.start_time).to_sec(),
            'auto_freeze_enabled': self.enable_auto_freeze,
            'obstacle_mode': self.obstacle_mode,
            'remaining_time': max(0, self.observation_time - (rospy.Time.now() - self.start_time).to_sec()) if self.enable_auto_freeze else -1
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status_info)
        self.status_pub.publish(status_msg)
    
    def publish_complete_detection_results(self, detections):
        """发布完整的检测结果供点云过滤器使用"""
        complete_results = []
        
        for detection in detections:
            complete_result = {
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'center_2d': detection['center_2d'],
                'is_target': detection['is_target'],
                'is_obstacle': detection['is_obstacle'],
                'header': detection['header']
            }
            # 不包含3D坐标，因为点云过滤器会自己处理
            complete_results.append(complete_result)
        
        # 发布完整检测结果
        results_msg = String()
        results_msg.data = json.dumps(complete_results)
        self.detection_results_pub.publish(results_msg)
    
    def finish_observation(self, event):
        """完成观察，停止相机数据流"""
        if not self.is_observing or not self.enable_auto_freeze:
            return
        
        rospy.loginfo("观察时间结束，停止相机数据流...")
        
        # 停止观察
        self.is_observing = False
        
        # 发布最终检测结果摘要
        with self.detection_lock:
            final_summary = {
                'total_detections': len(self.stable_detections),
                'target_objects': [d for d in self.stable_detections.values() if d['is_target']],
                'obstacle_objects': [d for d in self.stable_detections.values() if d['is_obstacle']],
                'observation_completed': True,
                'obstacle_mode': self.obstacle_mode
            }
        
        rospy.loginfo(f"观察完成摘要: {len(final_summary['target_objects'])}个目标物体, {len(final_summary['obstacle_objects'])}个障碍物")
        
        # 调用快照冻结服务
        if self.freeze_service:
            try:
                self.freeze_service()
                rospy.loginfo("已通知快照控制器停止相机数据")
            except rospy.ServiceException as e:
                rospy.logwarn(f"调用快照冻结服务失败: {e}")
        
        # 发布完成状态
        completion_msg = String()
        completion_msg.data = json.dumps(final_summary)
        self.status_pub.publish(completion_msg)

if __name__ == '__main__':
    try:
        detector = YOLOMoveItDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
