#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云过滤调试脚本
检查YOLO检测结果和点云过滤的详细过程
"""

import rospy
import json
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class FilterDebugger:
    def __init__(self):
        rospy.init_node('filter_debugger')
        
        self.yolo_detections = None
        self.filter_status = None
        
        # 订阅关键话题
        rospy.Subscriber('/yolo/detection_results', String, self.yolo_callback)
        rospy.Subscriber('/pointcloud_filter/status', String, self.filter_status_callback)
        rospy.Subscriber('/camera/depth/color/points_relay', PointCloud2, self.original_pointcloud_callback)
        rospy.Subscriber('/camera/depth/color/points_filtered', PointCloud2, self.filtered_pointcloud_callback)
        
        rospy.loginfo("🔍 点云过滤调试器启动")
        
        # 定期输出详细状态
        self.debug_timer = rospy.Timer(rospy.Duration(5.0), self.print_debug_info)
    
    def yolo_callback(self, msg):
        try:
            self.yolo_detections = json.loads(msg.data)
        except:
            pass
    
    def filter_status_callback(self, msg):
        try:
            self.filter_status = json.loads(msg.data)
        except:
            pass
    
    def original_pointcloud_callback(self, msg):
        rospy.logdebug(f"收到原始点云: {msg.width}x{msg.height}, {len(msg.data)}字节")
    
    def filtered_pointcloud_callback(self, msg):
        rospy.logdebug(f"收到过滤点云: {msg.width}x{msg.height}, {len(msg.data)}字节")
    
    def print_debug_info(self, event):
        print("\n" + "="*60)
        print("🔍 点云过滤详细调试信息")
        print("="*60)
        
        # YOLO检测详情
        if self.yolo_detections:
            print("🎯 YOLO检测结果:")
            total_detections = len(self.yolo_detections)
            target_objects = [d for d in self.yolo_detections if d.get('is_target', False)]
            obstacle_objects = [d for d in self.yolo_detections if d.get('is_obstacle', False)]
            
            print(f"   总检测数: {total_detections}")
            print(f"   目标物体数: {len(target_objects)}")
            print(f"   障碍物数: {len(obstacle_objects)}")
            
            if target_objects:
                print("   目标物体详情:")
                for i, obj in enumerate(target_objects):
                    class_name = obj.get('class_name', 'unknown')
                    confidence = obj.get('confidence', 0)
                    bbox = obj.get('bbox', [])
                    print(f"     {i+1}. {class_name} (置信度: {confidence:.2f}, bbox: {bbox})")
            else:
                print("   ❌ 没有检测到目标物体！")
        else:
            print("🎯 YOLO检测结果: 无数据")
        
        # 过滤器状态
        if self.filter_status:
            print("✂️  点云过滤状态:")
            print(f"   过滤功能: {'启用' if self.filter_status.get('filter_enabled', False) else '禁用'}")
            print(f"   原始点数: {self.filter_status.get('original_points', 0)}")
            print(f"   过滤后点数: {self.filter_status.get('filtered_points', 0)}")
            print(f"   移除点数: {self.filter_status.get('removed_points', 0)}")
            print(f"   移除比例: {self.filter_status.get('removal_ratio', 0):.2%}")
        else:
            print("✂️  点云过滤状态: 无数据")
        
        # 问题诊断
        print("🔧 问题诊断:")
        if not self.yolo_detections:
            print("   ❌ YOLO检测器未连接或无数据")
        elif not any(d.get('is_target', False) for d in self.yolo_detections):
            print("   ⚠️  YOLO已连接但未检测到目标物体")
            print("   建议: 确保场景中有keyboard并且清晰可见")
        elif self.filter_status and self.filter_status.get('removed_points', 0) == 0:
            print("   ⚠️  检测到目标但点云未被过滤")
            print("   可能原因: 1) 点云结构问题 2) 边界框映射问题 3) 过滤算法问题")
        else:
            print("   ✅ 系统工作正常")
        
        print("="*60)

if __name__ == '__main__':
    try:
        debugger = FilterDebugger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass