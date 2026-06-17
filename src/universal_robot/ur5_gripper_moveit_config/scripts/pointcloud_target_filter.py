#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云目标过滤器 - 基于YOLO检测结果移除目标物体的点云
为MoveIt的octomap提供清洁的障碍物点云数据
"""

import rospy
import numpy as np
import json
import threading
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Empty
from std_srvs.srv import Empty as EmptyService
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.point_cloud2 import create_cloud_xyz32

class PointCloudTargetFilter:
    def __init__(self):
        rospy.init_node('pointcloud_target_filter')
        
        # 参数配置
        self.filter_enabled = rospy.get_param('~filter_enabled', True)
        self.target_expand_factor = rospy.get_param('~target_expand_factor', 1.2)  # 目标区域扩展因子
        self.min_depth = rospy.get_param('~min_depth', 0.1)  # 最小深度（米）
        self.max_depth = rospy.get_param('~max_depth', 2.0)  # 最大深度（米）
        
        # 数据存储
        self.latest_pointcloud = None
        self.current_detections = []
        self.data_lock = threading.Lock()
        self.yolo_connected = False
        self.last_yolo_time = None
        
        # ROS订阅者
        self.pointcloud_sub = rospy.Subscriber(
            '/camera/depth/color/points_relay', 
            PointCloud2, 
            self.pointcloud_callback
        )
        
        self.detection_sub = rospy.Subscriber(
            '/yolo/detection_results', 
            String, 
            self.detection_callback
        )
        
        # ROS发布者
        self.filtered_pointcloud_pub = rospy.Publisher(
            '/camera/depth/color/points_filtered', 
            PointCloud2, 
            queue_size=1
        )
        
        self.status_pub = rospy.Publisher(
            '/pointcloud_filter/status', 
            String, 
            queue_size=1
        )
        
        rospy.loginfo("点云目标过滤器已启动")
        rospy.loginfo(f"过滤功能: {'启用' if self.filter_enabled else '禁用'}")
        rospy.loginfo(f"目标扩展因子: {self.target_expand_factor}")
        rospy.loginfo(f"深度范围: {self.min_depth}m - {self.max_depth}m")
        rospy.loginfo("🔄 当前模式: 转发模式 (等待YOLO连接)")
        
        # 初始化octomap清除服务
        try:
            rospy.wait_for_service('/clear_octomap', timeout=5.0)
            self.clear_octomap_service = rospy.ServiceProxy('/clear_octomap', EmptyService)
            rospy.loginfo("✅ 连接到octomap清除服务")
        except rospy.ROSException:
            rospy.logwarn("⚠️ 无法连接到octomap清除服务，障碍物可能不会实时更新")
            self.clear_octomap_service = None
        
        # 定期检查YOLO连接状态
        self.status_timer = rospy.Timer(rospy.Duration(10.0), self.check_yolo_status)
        
        # 用于跟踪之前是否有目标被检测到
        self.previous_target_detected = False
    
    def detection_callback(self, msg):
        """YOLO检测结果回调"""
        try:
            detections = json.loads(msg.data)
            with self.data_lock:
                self.current_detections = detections
                if not self.yolo_connected:
                    self.yolo_connected = True
                    rospy.loginfo("🎯 YOLO检测器已连接 - 切换到过滤模式")
                self.last_yolo_time = rospy.Time.now()
            rospy.logdebug(f"接收到检测结果: {len(detections)}个物体")
        except json.JSONDecodeError as e:
            rospy.logwarn(f"解析检测结果失败: {e}")
    
    def pointcloud_callback(self, msg):
        """点云数据回调和过滤处理"""
        if not self.filter_enabled:
            # 如果过滤功能禁用，直接转发原始点云
            self.filtered_pointcloud_pub.publish(msg)
            return
        
        try:
            with self.data_lock:
                detections = self.current_detections.copy()
            
            if not detections:
                # 没有检测结果时，发布原始点云（这是正常情况，不是错误）
                self.filtered_pointcloud_pub.publish(msg)
                if not self.yolo_connected:
                    rospy.logdebug("🔄 转发模式：等待YOLO检测结果，发布原始点云")
                else:
                    rospy.logdebug("📭 过滤模式：暂无检测目标，发布原始点云")
                return
            
            # 过滤点云
            filtered_cloud = self.filter_target_points(msg, detections)
            
            if filtered_cloud:
                self.filtered_pointcloud_pub.publish(filtered_cloud)
                
                # 发布状态信息
                self.publish_status(msg, filtered_cloud, detections)
            else:
                rospy.logwarn("点云过滤失败，发布原始点云")
                self.filtered_pointcloud_pub.publish(msg)
                
        except Exception as e:
            rospy.logerr(f"点云过滤错误: {e}")
            # 出错时发布原始点云，确保系统继续运行
            self.filtered_pointcloud_pub.publish(msg)
    
    def filter_target_points(self, pointcloud_msg, detections):
        """过滤点云中的目标物体"""
        try:
            # 提取所有3D点
            points = list(pc2.read_points(
                pointcloud_msg, 
                field_names=("x", "y", "z"), 
                skip_nans=True
            ))
            
            if not points:
                rospy.logwarn("点云中没有有效点")
                return None
            
            # 转换为numpy数组便于处理
            points_array = np.array(points)
            
            # 深度过滤
            valid_depth_mask = (
                (points_array[:, 2] >= self.min_depth) & 
                (points_array[:, 2] <= self.max_depth)
            )
            points_array = points_array[valid_depth_mask]
            
            if len(points_array) == 0:
                rospy.logwarn("深度过滤后没有有效点")
                return None
            
            # 获取目标物体的边界框
            target_bboxes = []
            target_count = 0
            for detection in detections:
                if detection.get('is_target', False) and 'bbox' in detection:
                    bbox = detection['bbox']
                    # 扩展边界框
                    # 使用相机图像尺寸而不是点云尺寸来扩展边界框
                    expanded_bbox = self.expand_bbox(bbox, 640, 480)  # RealSense相机尺寸
                    target_bboxes.append(expanded_bbox)
                    target_count += 1
                    rospy.loginfo(f"检测到目标物体: {detection.get('class_name', 'unknown')}, 原始bbox: {bbox}, 扩展后: {expanded_bbox}")
            
            if not target_bboxes:
                # 没有目标物体，返回所有点
                rospy.logdebug("没有检测到目标物体，保留所有点")
                
                # 如果之前有目标被检测到，现在没有了，清除octomap
                if self.previous_target_detected:
                    self.clear_octomap_if_needed("目标物体消失")
                    self.previous_target_detected = False
                    
                return self.create_filtered_pointcloud(points_array, pointcloud_msg)
            else:
                rospy.loginfo(f"总共检测到 {target_count} 个目标物体，将进行点云过滤")
                
                # 如果之前没有目标，现在有了，清除octomap以重新构建
                if not self.previous_target_detected:
                    self.clear_octomap_if_needed("检测到新目标")
                    self.previous_target_detected = True
            
            # 将3D点投影回2D像素坐标进行过滤
            filtered_points = self.filter_points_by_bboxes(
                points_array, target_bboxes, pointcloud_msg
            )
            
            rospy.loginfo(f"点云过滤完成: {len(points_array)} -> {len(filtered_points)} 点")
            
            return self.create_filtered_pointcloud(filtered_points, pointcloud_msg)
            
        except Exception as e:
            rospy.logerr(f"点云过滤处理错误: {e}")
            return None
    
    def expand_bbox(self, bbox, image_width, image_height):
        """扩展边界框"""
        x1, y1, x2, y2 = bbox
        
        # 计算扩展量
        width = x2 - x1
        height = y2 - y1
        expand_w = int(width * (self.target_expand_factor - 1) / 2)
        expand_h = int(height * (self.target_expand_factor - 1) / 2)
        
        # 应用扩展并确保在图像范围内
        expanded_x1 = max(0, x1 - expand_w)
        expanded_y1 = max(0, y1 - expand_h)
        expanded_x2 = min(image_width - 1, x2 + expand_w)
        expanded_y2 = min(image_height - 1, y2 + expand_h)
        
        rospy.logdebug(f"边界框扩展: 原始[{x1},{y1},{x2},{y2}] -> 扩展[{expanded_x1},{expanded_y1},{expanded_x2},{expanded_y2}] (图像尺寸: {image_width}x{image_height})")
        
        return [expanded_x1, expanded_y1, expanded_x2, expanded_y2]
    
    def filter_points_by_bboxes(self, points_3d, target_bboxes, pointcloud_msg):
        """根据边界框过滤3D点"""
        if len(target_bboxes) == 0:
            rospy.logdebug("没有目标边界框，保留所有点")
            return points_3d
        
        width = pointcloud_msg.width
        height = pointcloud_msg.height
        
        rospy.loginfo(f"开始过滤 {len(points_3d)} 个点，目标边界框数量: {len(target_bboxes)}")
        for i, bbox in enumerate(target_bboxes):
            rospy.loginfo(f"目标边界框 {i}: {bbox}")
        
        # 如果点云不是有序的，我们需要使用相机内参进行投影
        # 但是这里我们先尝试简化方法：使用深度信息过滤
        filtered_points = []
        
        # 方法1: 基于点云的有序结构（如果可用）
        # 检查是否为真正的有序点云 (height > 1表示2D网格结构)
        if width > 1 and height > 1 and len(points_3d) == width * height:
            rospy.loginfo("使用有序点云过滤方法")
            removed_count = 0
            
            for i, point in enumerate(points_3d):
                # 计算像素位置
                pixel_u = i % width
                pixel_v = i // width
                
                # 检查是否在目标区域内
                point_in_target = False
                for bbox in target_bboxes:
                    x1, y1, x2, y2 = bbox
                    if x1 <= pixel_u <= x2 and y1 <= pixel_v <= y2:
                        point_in_target = True
                        removed_count += 1
                        break
                
                # 如果不在目标区域，保留该点
                if not point_in_target:
                    filtered_points.append(point)
            
            rospy.loginfo(f"有序点云过滤完成: 移除了 {removed_count} 个点")
            
        else:
            # 方法2: 无序点云，使用3D到2D投影过滤
            rospy.loginfo(f"点云不是有序的 (width={width}, height={height}, points={len(points_3d)})，使用投影过滤")
            removed_count = 0
            
            # 真实的相机内参（从camera_info获取）
            fx = 615.2874145507812  # 焦距x
            fy = 615.3785400390625  # 焦距y  
            cx = 322.08758544921875  # 主点x
            cy = 234.27511596679688  # 主点y
            
            rospy.loginfo(f"使用相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            
            for point in points_3d:
                x, y, z = point
                
                # 检查深度有效性
                if z <= 0 or z > self.max_depth or z < self.min_depth:
                    filtered_points.append(point)  # 保留无效深度的点
                    continue
                
                # 3D到2D投影
                pixel_u = int(fx * x / z + cx)
                pixel_v = int(fy * y / z + cy)
                
                # 检查投影是否在图像范围内
                if pixel_u < 0 or pixel_v < 0 or pixel_u >= 640 or pixel_v >= 480:
                    filtered_points.append(point)  # 保留图像外的点
                    continue
                
                # 检查是否在目标区域内
                point_in_target = False
                for bbox in target_bboxes:
                    x1, y1, x2, y2 = bbox
                    if x1 <= pixel_u <= x2 and y1 <= pixel_v <= y2:
                        point_in_target = True
                        removed_count += 1
                        break
                
                # 如果不在目标区域，保留该点
                if not point_in_target:
                    filtered_points.append(point)
            
            rospy.loginfo(f"无序点云过滤完成: 移除了 {removed_count} 个点")
        
        result_array = np.array(filtered_points) if filtered_points else np.array([]).reshape(0, 3)
        rospy.loginfo(f"过滤结果: {len(points_3d)} -> {len(result_array)} 点")
        
        return result_array
    
    def create_filtered_pointcloud(self, points_array, original_msg):
        """创建过滤后的点云消息"""
        if len(points_array) == 0:
            # 创建空点云
            header = original_msg.header
            return create_cloud_xyz32(header, [])
        
        # 创建新的点云消息
        header = original_msg.header
        filtered_cloud = create_cloud_xyz32(header, points_array.tolist())
        
        return filtered_cloud
    
    def publish_status(self, original_cloud, filtered_cloud, detections):
        """发布过滤状态信息"""
        try:
            original_points = list(pc2.read_points(original_cloud, skip_nans=True))
            filtered_points = list(pc2.read_points(filtered_cloud, skip_nans=True))
            
            target_count = sum(1 for d in detections if d.get('is_target', False))
            
            status_info = {
                'filter_enabled': self.filter_enabled,
                'original_points': len(original_points),
                'filtered_points': len(filtered_points),
                'removed_points': len(original_points) - len(filtered_points),
                'target_objects_detected': target_count,
                'removal_ratio': (len(original_points) - len(filtered_points)) / max(1, len(original_points))
            }
            
            status_msg = String()
            status_msg.data = json.dumps(status_info)
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            rospy.logwarn(f"发布状态信息失败: {e}")
    
    def check_yolo_status(self, event):
        """定期检查YOLO连接状态"""
        with self.data_lock:
            if self.yolo_connected and self.last_yolo_time:
                time_since_last = (rospy.Time.now() - self.last_yolo_time).to_sec()
                if time_since_last > 30.0:  # 30秒没有数据认为断开
                    self.yolo_connected = False
                    rospy.logwarn("🔌 YOLO连接可能已断开 - 切换回转发模式")
            elif not self.yolo_connected:
                rospy.loginfo("⏳ 仍在等待YOLO检测器连接...")
    
    def clear_octomap_if_needed(self, reason=""):
        """在必要时清除octomap"""
        if self.clear_octomap_service:
            try:
                self.clear_octomap_service()
                rospy.loginfo(f"🧹 清除octomap - {reason}")
            except rospy.ServiceException as e:
                rospy.logwarn(f"清除octomap失败: {e}")
        else:
            rospy.logdebug(f"octomap清除服务不可用 - {reason}")

if __name__ == '__main__':
    try:
        filter_node = PointCloudTargetFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass