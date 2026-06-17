#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2
import threading

class SnapshotController:
    def __init__(self):
        rospy.init_node('snapshot_controller')
        
        # 参数设置
        self.auto_freeze_enabled = rospy.get_param('~auto_freeze_enabled', True)
        self.freeze_delay = rospy.get_param('~freeze_delay', 10.0)  # 默认10秒后冻结
        
        # 等待MoveIt服务启动
        rospy.wait_for_service('/clear_octomap')
        self.clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
        
        # 控制点云订阅的开关
        self.pointcloud_enabled = True
        self.lock = threading.Lock()
        self.start_time = rospy.Time.now()
        self.auto_frozen = False
        
        # 原始点云topic
        self.original_topic = "/camera/depth/color/points"
        self.relay_topic = "/camera/depth/color/points_relay"
        
        # 订阅原始点云
        self.pointcloud_sub = rospy.Subscriber(
            self.original_topic, PointCloud2, self.pointcloud_callback
        )
        
        # 发布中继点云
        self.pointcloud_pub = rospy.Publisher(
            self.relay_topic, PointCloud2, queue_size=1
        )
        
        # 服务
        rospy.Service('/snapshot/freeze', Empty, self.freeze_callback)
        rospy.Service('/snapshot/unfreeze', Empty, self.unfreeze_callback)
        rospy.Service('/snapshot/clear', Empty, self.clear_callback)
        rospy.Service('/snapshot/reset_timer', Empty, self.reset_timer_callback)
        
        # 定时器 - 用于检查是否需要自动冻结
        if self.auto_freeze_enabled:
            self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        
        rospy.loginfo("Snapshot controller ready!")
        rospy.loginfo("Services available:")
        rospy.loginfo("  /snapshot/freeze - 手动冻结障碍物更新")
        rospy.loginfo("  /snapshot/unfreeze - 恢复障碍物更新") 
        rospy.loginfo("  /snapshot/clear - 清空当前障碍物")
        rospy.loginfo("  /snapshot/reset_timer - 重置自动冻结计时器")
        
        if self.auto_freeze_enabled:
            rospy.loginfo(f"Auto-freeze enabled: will freeze after {self.freeze_delay} seconds")
        else:
            rospy.loginfo("Auto-freeze disabled: manual control only")
    
    def timer_callback(self, event):
        """定时器回调 - 检查是否需要自动冻结"""
        if not self.auto_frozen and self.pointcloud_enabled:
            elapsed = (rospy.Time.now() - self.start_time).to_sec()
            if elapsed >= self.freeze_delay:
                with self.lock:
                    self.pointcloud_enabled = False
                    self.auto_frozen = True
                rospy.loginfo(f"自动冻结：{self.freeze_delay}秒后自动停止障碍物更新")
    
    def pointcloud_callback(self, msg):
        """点云回调函数，根据开关决定是否转发"""
        with self.lock:
            if self.pointcloud_enabled:
                self.pointcloud_pub.publish(msg)
    
    def freeze_callback(self, req):
        """手动冻结障碍物更新"""
        with self.lock:
            self.pointcloud_enabled = False
            self.auto_frozen = True  # 手动冻结也算作已冻结状态
        rospy.loginfo("手动冻结：障碍物地图已冻结 - 停止更新")
        return []
    
    def unfreeze_callback(self, req):
        """恢复障碍物更新"""
        with self.lock:
            self.pointcloud_enabled = True
            self.auto_frozen = False
        # 重置计时器
        self.start_time = rospy.Time.now()
        rospy.loginfo("障碍物地图已解冻 - 恢复更新，重置自动冻结计时器")
        return []
    
    def clear_callback(self, req):
        """清空当前障碍物"""
        try:
            self.clear_octomap()
            rospy.loginfo("障碍物地图已清空")
        except rospy.ServiceException as e:
            rospy.logerr(f"清空失败: {e}")
        return []
    
    def reset_timer_callback(self, req):
        """重置自动冻结计时器"""
        with self.lock:
            if not self.pointcloud_enabled:
                # 如果当前是冻结状态，先解冻
                self.pointcloud_enabled = True
                self.auto_frozen = False
        self.start_time = rospy.Time.now()
        rospy.loginfo(f"计时器已重置，将在{self.freeze_delay}秒后自动冻结")
        return []

if __name__ == '__main__':
    try:
        controller = SnapshotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass