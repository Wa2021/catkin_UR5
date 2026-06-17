#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoveIt障碍物管理节点 - 运行在ROS原生环境中
订阅YOLO检测结果，发布障碍物到MoveIt规划场景
"""

import rospy
import json
from std_msgs.msg import String, Header
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PointStamped
import tf2_ros
import tf2_geometry_msgs
import moveit_commander

class MoveItObstacleManager:
    def __init__(self):
        rospy.init_node('moveit_obstacle_manager')
        
        # 初始化MoveIt
        moveit_commander.roscpp_initialize([])
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # TF2变换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 配置参数
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        
        # 当前障碍物记录
        self.current_obstacles = {}
        
        # 订阅YOLO检测结果
        self.detection_sub = rospy.Subscriber('/yolo/object_points', String, self.detection_callback)
        
        # 发布规划场景
        self.planning_scene_pub = rospy.Publisher('/planning_scene', PlanningScene, queue_size=10)
        
        # 等待规划场景发布者连接
        rospy.loginfo("等待规划场景发布者连接...")
        while self.planning_scene_pub.get_num_connections() < 1:
            rospy.sleep(0.1)
        
        rospy.loginfo("MoveIt障碍物管理器已启动 (ROS原生环境)")
        rospy.loginfo(f"机器人坐标系: {self.robot_frame}")
        rospy.loginfo(f"相机坐标系: {self.camera_frame}")
    
    def detection_callback(self, msg):
        """处理YOLO检测结果"""
        try:
            object_points = json.loads(msg.data)
            
            # 清除旧的障碍物
            self.clear_old_obstacles()
            
            # 处理新检测到的物体
            for i, obj in enumerate(object_points):
                self.process_detected_object(obj, i)
                
        except Exception as e:
            rospy.logerr(f"处理检测结果错误: {e}")
    
    def process_detected_object(self, obj, index):
        """处理单个检测到的物体"""
        try:
            class_name = obj['class_name']
            point_3d = obj['point_3d']
            bbox = obj['bbox']
            confidence = obj['confidence']
            header_data = obj['header']
            
            # 重建Header
            header = Header()
            header.frame_id = header_data['frame_id']
            header.stamp.secs = header_data['stamp']['secs']
            header.stamp.nsecs = header_data['stamp']['nsecs']
            
            # 转换到机器人坐标系
            robot_point = self.transform_to_robot_frame(point_3d, header)
            
            if robot_point:
                # 发布障碍物
                obstacle_id = f"obstacle_{class_name}_{index}"
                self.publish_obstacle(obstacle_id, class_name, robot_point, bbox)
                
                rospy.loginfo(f"添加障碍物: {class_name} at ({robot_point['x']:.2f}, {robot_point['y']:.2f}, {robot_point['z']:.2f})")
            
        except Exception as e:
            rospy.logerr(f"处理物体错误: {e}")
    
    def transform_to_robot_frame(self, point_3d, header):
        """将点从相机坐标系转换到机器人坐标系"""
        try:
            # 创建点的几何消息
            point_stamped = PointStamped()
            point_stamped.header = header
            point_stamped.point.x = point_3d['x']
            point_stamped.point.y = point_3d['y']
            point_stamped.point.z = point_3d['z']
            
            # 转换到机器人坐标系
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame, header.frame_id, rospy.Time(), rospy.Duration(1.0)
            )
            
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            
            return {
                'x': transformed_point.point.x,
                'y': transformed_point.point.y,
                'z': transformed_point.point.z
            }
        except Exception as e:
            rospy.logwarn(f"坐标转换失败: {e}")
            return None
    
    def publish_obstacle(self, obstacle_id, class_name, robot_point, bbox):
        """发布障碍物到MoveIt规划场景"""
        try:
            # 创建规划场景消息
            planning_scene = PlanningScene()
            planning_scene.is_diff = True
            
            # 创建碰撞对象
            collision_object = CollisionObject()
            collision_object.header.frame_id = self.robot_frame
            collision_object.id = obstacle_id
            
            # 根据物体类型估算尺寸
            dimensions = self.estimate_object_dimensions(class_name, bbox)
            
            # 创建基本形状（圆柱体）
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [dimensions['height'], dimensions['radius']]
            
            # 设置位置和姿态
            pose = Pose()
            pose.position.x = robot_point['x']
            pose.position.y = robot_point['y']
            pose.position.z = robot_point['z']
            pose.orientation.w = 1.0
            
            # 添加到碰撞对象
            collision_object.primitives.append(primitive)
            collision_object.primitive_poses.append(pose)
            collision_object.operation = CollisionObject.ADD
            
            # 添加到规划场景
            planning_scene.world.collision_objects.append(collision_object)
            
            # 发布
            self.planning_scene_pub.publish(planning_scene)
            
            # 记录当前障碍物
            self.current_obstacles[obstacle_id] = {
                'class_name': class_name,
                'position': robot_point
            }
            
        except Exception as e:
            rospy.logerr(f"发布障碍物错误: {e}")
    
    def estimate_object_dimensions(self, class_name, bbox):
        """根据物体类型和检测框估算物体尺寸"""
        # 基于检测框大小估算
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        # 预定义尺寸（针对常见物体）
        predefined_sizes = {
            'person': {'height': 1.7, 'radius': 0.3},
            'laptop': {'height': 0.05, 'radius': 0.15},
            'book': {'height': 0.05, 'radius': 0.12},
            'chair': {'height': 0.8, 'radius': 0.25},
            'mouse': {'height': 0.03, 'radius': 0.06},
            'keyboard': {'height': 0.03, 'radius': 0.20},  # 87键键盘：长40cm，宽15cm，高3cm，半径取长轴的一半
            'cell phone': {'height': 0.01, 'radius': 0.08},
            'tv': {'height': 0.1, 'radius': 0.4},
            'monitor': {'height': 0.1, 'radius': 0.3},
            'backpack': {'height': 0.4, 'radius': 0.2},
            'handbag': {'height': 0.3, 'radius': 0.15},
            'suitcase': {'height': 0.6, 'radius': 0.3},
        }
        
        # 如果有预定义尺寸，使用预定义的
        if class_name in predefined_sizes:
            return predefined_sizes[class_name]
        
        # 否则根据检测框动态估算
        # 像素转米的估算系数（可以根据实际情况调整）
        pixel_to_meter = 0.001
        estimated_width = bbox_width * pixel_to_meter
        estimated_height = bbox_height * pixel_to_meter
        
        # 设置合理的最小和最大尺寸
        min_size = 0.05  # 最小5cm
        max_size = 1.0   # 最大1m
        
        estimated_radius = max(min_size, min(estimated_width * 0.5, max_size))
        estimated_obj_height = max(min_size, min(estimated_height, max_size))
        
        return {
            'height': estimated_obj_height,
            'radius': estimated_radius
        }
    
    def clear_old_obstacles(self):
        """清除旧的障碍物"""
        if not self.current_obstacles:
            return
        
        try:
            planning_scene = PlanningScene()
            planning_scene.is_diff = True
            
            for obstacle_id in self.current_obstacles.keys():
                collision_object = CollisionObject()
                collision_object.id = obstacle_id
                collision_object.operation = CollisionObject.REMOVE
                planning_scene.world.collision_objects.append(collision_object)
            
            self.planning_scene_pub.publish(planning_scene)
            self.current_obstacles.clear()
            
        except Exception as e:
            rospy.logerr(f"清除障碍物错误: {e}")

if __name__ == '__main__':
    try:
        manager = MoveItObstacleManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass