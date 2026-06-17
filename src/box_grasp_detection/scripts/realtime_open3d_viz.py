#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open3D 实时可视化器 - Realtime Open3D Visualizer
展示 Open3D 如何在 ROS 中实现实时点云和抓取结果的渲染
"""

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
from box_grasp_detection.msg import BoxGraspArray
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation
import threading
import time

class RealtimeOpen3DVisualizer:
    def __init__(self):
        rospy.init_node('realtime_open3d_viz', anonymous=True)
        
        # 1. 初始化 Open3D 可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Realtime Open3D Grasp Viz', width=1280, height=720)
        
        # 设置渲染选项
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1]) # 深灰色背景
        opt.point_size = 2.0
        opt.show_coordinate_frame = True
        
        # 2. 初始化几何体
        # 点云容器
        self.pcd = o3d.geometry.PointCloud()
        # 添加一个虚拟点避免 "0 points" 警告
        self.pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        self.pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        self.vis.add_geometry(self.pcd)
        
        # 抓取坐标系容器 (列表)
        self.grasp_geometries = []
        
        # 3. 数据同步控制
        self.lock = threading.Lock()
        self.new_cloud = False
        self.new_grasps = False
        self.latest_grasps_msg = None
        self.processing_cloud = False  # 防止处理积压
        
        # 4. 订阅话题
        # 订阅点云 (注意：高分辨率点云转换可能会消耗CPU)
        self.cloud_sub = rospy.Subscriber(
            '/camera/depth/color/points', 
            PointCloud2, 
            self.cloud_callback, 
            queue_size=1,
            buff_size=2**24  # 增加缓冲区大小
        )
        
        # 订阅抓取结果
        # 订阅 /box_grasps 话题，这是 box_grasp_planner.py 发布的包含所有抓取的话题
        self.grasp_sub = rospy.Subscriber(
            '/box_grasps', 
            BoxGraspArray, 
            self.grasp_callback, 
            queue_size=1
        )
        
        self._has_received_cloud = False
        
        rospy.loginfo("🚀 Open3D 实时可视化器已启动")
        rospy.loginfo("   按 'Q' 键退出")
        rospy.loginfo("   等待点云数据...")

    def cloud_callback(self, msg):
        """点云回调：将ROS消息转换为Open3D格式"""
        # 简单的丢帧逻辑：如果正在处理上一帧，则丢弃当前帧
        if self.processing_cloud:
            return
            
        self.processing_cloud = True
        
        if not self._has_received_cloud:
            rospy.loginfo("✅ 收到第一帧点云数据！正在处理...")
            self._has_received_cloud = True
        
        # 为了保证实时性，我们在回调中做繁重的数据转换
        try:
            # 使用 sensor_msgs.point_cloud2 读取点云 (x, y, z, rgb)
            # 注意：这里假设点云是无序的或者我们只取有效点
            # 为了性能，可以设置 skip_nans=True
            gen = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
            
            # 转换为 numpy 数组
            points_list = list(gen)
            if not points_list:
                self.processing_cloud = False
                return
                
            points_np = np.array(points_list)
            
            # 降采样：每N个点取一个，减少渲染压力
            # 虚拟机环境下建议 N=5 或更大
            step = 5 
            points_np = points_np[::step]
            
            # 分离坐标和颜色
            xyz = points_np[:, 0:3]
            rgb_float = points_np[:, 3]
            
            # 确保内存连续，解决 "last axis must be contiguous" 错误
            rgb_float = np.ascontiguousarray(rgb_float)
            
            # 解析 RGB (float32 -> uint8 -> float 0-1)
            # 这是一个简化的转换，通常 RGB 被打包在一个 float 中
            # 这里使用一种快速近似或标准解包方法
            rgb_uint32 = rgb_float.view(np.uint32)
            r = ((rgb_uint32 >> 16) & 0xFF) / 255.0
            g = ((rgb_uint32 >> 8) & 0xFF) / 255.0
            b = (rgb_uint32 & 0xFF) / 255.0
            rgb = np.stack([r, g, b], axis=1)
            
            with self.lock:
                # 更新点云数据
                self.pcd.points = o3d.utility.Vector3dVector(xyz)
                self.pcd.colors = o3d.utility.Vector3dVector(rgb)
                self.new_cloud = True
                
                # 如果是第一帧，重置视角以确保能看到点云
                if not hasattr(self, '_view_initialized'):
                    self._view_initialized = True
                    self.vis.reset_view_point(True)
                
        except Exception as e:
            rospy.logwarn(f"点云转换错误: {e}")
        finally:
            self.processing_cloud = False

    def grasp_callback(self, msg):
        """抓取结果回调"""
        with self.lock:
            self.latest_grasps_msg = msg
            self.new_grasps = True

    def update_grasps(self):
        """更新抓取坐标系几何体"""
        if self.latest_grasps_msg is None:
            return

        # 1. 移除旧的几何体
        for geom in self.grasp_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.grasp_geometries.clear()
        
        # 2. 生成新的几何体
        for grasp in self.latest_grasps_msg.grasps:
            # 创建坐标系
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            
            # 设置位置
            pos = grasp.grasp_pose.pose.position
            mesh.translate([pos.x, pos.y, pos.z])
            
            # 设置姿态
            ori = grasp.grasp_pose.pose.orientation
            r = Rotation.from_quat([ori.x, ori.y, ori.z, ori.w])
            mesh.rotate(r.as_matrix(), center=[pos.x, pos.y, pos.z])
            
            # 添加到场景
            self.vis.add_geometry(mesh, reset_bounding_box=False)
            self.grasp_geometries.append(mesh)

    def run(self):
        """主渲染循环"""
        rate = rospy.Rate(30) # 30Hz 刷新率
        
        while not rospy.is_shutdown():
            updated = False
            
            with self.lock:
                # 更新点云
                if self.new_cloud:
                    self.vis.update_geometry(self.pcd)
                    # 强制重置包围盒，有时update_geometry不会自动更新包围盒导致渲染裁剪
                    # self.vis.update_renderer() # 移到下面统一调用
                    self.new_cloud = False
                    updated = True
                
                # 更新抓取
                if self.new_grasps:
                    self.update_grasps()
                    self.new_grasps = False
                    updated = True
            
            # Open3D 渲染步进
            keep_running = self.vis.poll_events()
            self.vis.update_renderer()
            
            if not keep_running:
                break
                
            rate.sleep()
            
        self.vis.destroy_window()

if __name__ == '__main__':
    try:
        viz = RealtimeOpen3DVisualizer()
        viz.run()
    except rospy.ROSInterruptException:
        pass
