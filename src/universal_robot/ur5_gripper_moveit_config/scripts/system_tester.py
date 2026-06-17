#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云过滤系统测试脚本
验证新的YOLO+点云过滤避障系统是否正常工作
"""

import rospy
import json
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class SystemTester:
    def __init__(self):
        rospy.init_node('system_tester')
        
        # 数据收集
        self.yolo_status = None
        self.filter_status = None
        self.original_pointcloud_size = 0
        self.filtered_pointcloud_size = 0
        
        # 订阅关键话题
        rospy.Subscriber('/yolo/status', String, self.yolo_status_callback)
        rospy.Subscriber('/pointcloud_filter/status', String, self.filter_status_callback)
        rospy.Subscriber('/camera/depth/color/points_relay', PointCloud2, self.original_pointcloud_callback)
        rospy.Subscriber('/camera/depth/color/points_filtered', PointCloud2, self.filtered_pointcloud_callback)
        
        rospy.loginfo("系统测试器已启动，开始监控系统状态...")
        
        # 定期输出状态
        self.status_timer = rospy.Timer(rospy.Duration(5.0), self.print_status)
    
    def yolo_status_callback(self, msg):
        try:
            self.yolo_status = json.loads(msg.data)
        except:
            pass
    
    def filter_status_callback(self, msg):
        try:
            self.filter_status = json.loads(msg.data)
        except:
            pass
    
    def original_pointcloud_callback(self, msg):
        try:
            points = list(pc2.read_points(msg, skip_nans=True))
            self.original_pointcloud_size = len(points)
        except:
            pass
    
    def filtered_pointcloud_callback(self, msg):
        try:
            points = list(pc2.read_points(msg, skip_nans=True))
            self.filtered_pointcloud_size = len(points)
        except:
            pass
    
    def print_status(self, event):
        print("\n" + "="*60)
        print("系统状态报告")
        print("="*60)
        
        # YOLO状态
        if self.yolo_status:
            print(f"YOLO检测器:")
            print(f"  - 目标物体数量: {self.yolo_status.get('target_objects_found', 0)}")
            print(f"  - 障碍物数量: {self.yolo_status.get('obstacles_found', 0)}")
            print(f"  - 检测次数: {self.yolo_status.get('detection_count', 0)}")
            print(f"  - 观察状态: {'正在观察' if self.yolo_status.get('is_observing', False) else '已停止'}")
        else:
            print("YOLO检测器: 无状态数据")
        
        # 点云过滤状态
        if self.filter_status:
            print(f"点云过滤器:")
            print(f"  - 过滤功能: {'启用' if self.filter_status.get('filter_enabled', False) else '禁用'}")
            print(f"  - 原始点数: {self.filter_status.get('original_points', 0)}")
            print(f"  - 过滤后点数: {self.filter_status.get('filtered_points', 0)}")
            print(f"  - 移除点数: {self.filter_status.get('removed_points', 0)}")
            print(f"  - 移除比例: {self.filter_status.get('removal_ratio', 0):.2%}")
            print(f"  - 检测到目标: {self.filter_status.get('target_objects_detected', 0)}")
        else:
            print("点云过滤器: 无状态数据")
        
        # 点云大小
        print(f"点云数据:")
        print(f"  - 原始点云大小: {self.original_pointcloud_size}")
        print(f"  - 过滤后点云大小: {self.filtered_pointcloud_size}")
        
        # 系统评估
        print("系统评估:")
        if self.yolo_status and self.filter_status:
            if self.yolo_status.get('target_objects_found', 0) > 0:
                if self.filter_status.get('removed_points', 0) > 0:
                    print("  ✅ 系统正常：检测到目标物体并成功过滤点云")
                else:
                    print("  ⚠️  检测到目标但未过滤点云，检查过滤器配置")
            else:
                print("  ℹ️  未检测到目标物体，保持原始点云")
        else:
            print("  ⚠️  等待系统初始化完成...")
        
        print("="*60)

if __name__ == '__main__':
    try:
        tester = SystemTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass