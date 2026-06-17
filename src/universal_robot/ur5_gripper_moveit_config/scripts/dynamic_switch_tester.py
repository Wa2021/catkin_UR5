#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态切换测试脚本
验证点云过滤器是否能在YOLO启动前后正确切换模式
"""

import rospy
import json
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class DynamicSwitchTester:
    def __init__(self):
        rospy.init_node('dynamic_switch_tester')
        
        # 状态记录
        self.yolo_data_received = False
        self.last_yolo_time = None
        self.pointcloud_count = 0
        self.filtered_pointcloud_count = 0
        
        # 订阅者
        rospy.Subscriber('/yolo/detection_results', String, self.yolo_callback)
        rospy.Subscriber('/camera/depth/color/points_relay', PointCloud2, self.original_pointcloud_callback)
        rospy.Subscriber('/camera/depth/color/points_filtered', PointCloud2, self.filtered_pointcloud_callback)
        
        rospy.loginfo("🔍 动态切换测试器启动")
        rospy.loginfo("监控系统从'转发模式'到'过滤模式'的切换...")
        
        # 定期报告
        self.report_timer = rospy.Timer(rospy.Duration(3.0), self.report_status)
    
    def yolo_callback(self, msg):
        """YOLO数据回调"""
        if not self.yolo_data_received:
            rospy.loginfo("🎯 检测到YOLO数据！系统应该开始过滤模式")
            self.yolo_data_received = True
        
        self.last_yolo_time = rospy.Time.now()
        
        try:
            detections = json.loads(msg.data)
            target_count = sum(1 for d in detections if d.get('is_target', False))
            if target_count > 0:
                rospy.loginfo(f"🎯 检测到 {target_count} 个目标物体 - 点云应该被过滤")
        except:
            pass
    
    def original_pointcloud_callback(self, msg):
        """原始点云回调"""
        self.pointcloud_count += 1
    
    def filtered_pointcloud_callback(self, msg):
        """过滤后点云回调"""
        self.filtered_pointcloud_count += 1
    
    def report_status(self, event):
        """定期状态报告"""
        print("\n" + "="*50)
        print("🔍 动态切换状态报告")
        print("="*50)
        
        # YOLO状态
        if self.yolo_data_received:
            time_since_yolo = (rospy.Time.now() - self.last_yolo_time).to_sec()
            print(f"🎯 YOLO状态: 已连接 (最后数据: {time_since_yolo:.1f}秒前)")
        else:
            print("🎯 YOLO状态: 未连接 (等待启动)")
        
        # 数据流状态
        print(f"📊 数据流:")
        print(f"   原始点云: {self.pointcloud_count} 帧")
        print(f"   过滤点云: {self.filtered_pointcloud_count} 帧")
        
        # 系统模式判断
        if not self.yolo_data_received:
            print("🔄 当前模式: 转发模式 (等待YOLO)")
            print("   → 过滤器转发原始点云给MoveIt")
        else:
            print("✂️  当前模式: 过滤模式 (YOLO已连接)")
            print("   → 过滤器移除目标物体后发送给MoveIt")
        
        # 切换提示
        if not self.yolo_data_received:
            print("\n💡 测试提示:")
            print("   在另一个终端启动YOLO检测器:")
            print("   conda activate arm_grasp")
            print("   rosrun ur5_gripper_moveit_config yolo_moveit_detector.py")
        
        print("="*50)

if __name__ == '__main__':
    try:
        tester = DynamicSwitchTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass