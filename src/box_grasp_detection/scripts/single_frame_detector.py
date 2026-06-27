#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单帧抓取检测器 - Single Frame Grasp Detector
从相机获取一帧数据，输出最佳抓取位姿
"""

import rospy
import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from box_grasp_detection.msg import BoxGraspArray
import sensor_msgs.point_cloud2 as pc2
from threading import Lock, Event
import sys


class SingleFrameGraspDetector:
    """单帧抓取检测器"""
    
    def __init__(self, visualize=True):
        rospy.init_node('single_frame_grasp_detector', anonymous=True)
        
        self.visualize = visualize
        self.lock = Lock()
        self.frame_received = Event()
        self.grasps_received = Event()
        
        # 存储数据
        self.current_frame = None
        self.current_grasps = None
        
        # 订阅话题
        self.cloud_sub = rospy.Subscriber(
            '/camera/depth/color/points',
            PointCloud2,
            self.cloud_callback,
            queue_size=1
        )
        
        self.grasp_sub = rospy.Subscriber(
            '/box_grasps',
            BoxGraspArray,
            self.grasp_callback,
            queue_size=1
        )

        rospy.loginfo("=" * 70)
        rospy.loginfo("单帧抓取检测器已启动")
        rospy.loginfo("=" * 70)

    def cloud_callback(self, msg):
        """接收点云帧"""
        with self.lock:
            if not self.frame_received.is_set():
                self.current_frame = msg
                self.frame_received.set()
                rospy.loginfo("✓ 已接收点云帧")
    
    def grasp_callback(self, msg):
        """接收抓取结果"""
        with self.lock:
            if not self.grasps_received.is_set():
                self.current_grasps = msg
                self.grasps_received.set()

                if len(msg.grasps) > 0:
                    rospy.loginfo(f"✓ 已接收 {len(msg.grasps)} 个抓取候选")
    
    def wait_for_frame(self, timeout=5.0):
        """等待接收一帧数据"""
        rospy.loginfo("等待相机数据...")
        self.frame_received.clear()
        
        if not self.frame_received.wait(timeout):
            rospy.logerr("超时：未接收到相机数据！")
            rospy.logerr("请检查相机是否启动：rostopic hz /camera/depth/color/points")
            return False
        
        return True
    
    def wait_for_grasps(self, timeout=10.0):
        """等待检测结果"""
        rospy.loginfo("等待检测结果...")
        self.grasps_received.clear()
        
        if not self.grasps_received.wait(timeout):
            rospy.logerr("超时：未接收到检测结果！")
            rospy.logerr("请检查box_detector节点是否运行")
            return False
        
        return True
    
    def get_best_grasp(self):
        """获取最佳抓取位姿"""
        # 1. 等待一帧点云
        if not self.wait_for_frame():
            return None
        
        # 2. 等待检测结果
        if not self.wait_for_grasps():
            return None
        
        # 3. 返回最佳抓取
        with self.lock:
            if self.current_grasps is None or len(self.current_grasps.grasps) == 0:
                rospy.logwarn("未检测到抓取！")
                return None
            
            rospy.loginfo(f"🔍 检测到 {len(self.current_grasps.grasps)} 个物体，选择评分最高的:")
            for i, grasp in enumerate(self.current_grasps.grasps):
                rospy.loginfo(f"   物体{i+1}: 评分={grasp.score:.3f}, 位置=({grasp.box_pose.position.x:.3f}, {grasp.box_pose.position.y:.3f}, {grasp.box_pose.position.z:.3f})")

            # 选择评分最高的抓取
            best_grasp = max(self.current_grasps.grasps, key=lambda g: g.score)
            rospy.loginfo(f"✅ 选择物体: 评分={best_grasp.score:.3f}")
            
            return best_grasp
    
    def print_grasp_info(self, grasp):
        """打印抓取信息"""
        print("\n" + "=" * 70)
        print("📦 检测到药盒 - 最佳抓取位姿")
        print("=" * 70)
        
        # 盒子信息
        print("\n【盒子信息】")
        print(f"  尺寸: {grasp.length*100:.1f} × {grasp.width*100:.1f} × {grasp.height*100:.1f} cm")
        volume = grasp.length * grasp.width * grasp.height * 1e6
        print(f"  体积: {volume:.1f} cm³")
        
        # 抓取策略分析
        dimensions = [grasp.length, grasp.width, grasp.height]
        dim_labels = ['长度(L)', '宽度(W)', '高度(H)']
        min_idx = dimensions.index(min(dimensions))
        print(f"\n【智能抓取策略】")
        print(f"  检测到盒子朝向：")
        for i, (dim, label) in enumerate(zip(dimensions, dim_labels)):
            marker = " ← 最短边，抓取轴" if i == min_idx else ""
            print(f"    {label}: {dim*100:.1f} cm{marker}")
        
        grasp_axis_names = ['X轴（长度方向）', 'Y轴（宽度方向）', 'Z轴（高度方向）']
        print(f"  抓取策略: 沿{grasp_axis_names[min_idx]}夹取")
        print(f"  优势: 抓取力臂最短，稳定性最高")
        
        # 盒子位置
        print(f"\n【盒子位置】")
        print(f"  X: {grasp.box_pose.position.x:.4f} m")
        print(f"  Y: {grasp.box_pose.position.y:.4f} m")
        print(f"  Z: {grasp.box_pose.position.z:.4f} m  ← 盒子中心高度")
        
        # 抓取信息
        print(f"\n【抓取信息】")
        print(f"  类型: {grasp.grasp_type}")
        print(f"  评分: {grasp.score:.3f} / 1.000")
        
        # 抓取位姿（关键输出！）
        print(f"\n【抓取位姿 - 6D Pose】")
        print(f"  坐标系: {grasp.grasp_pose.header.frame_id}")
        
        pos = grasp.grasp_pose.pose.position
        print(f"\n  位置 (Position):")
        print(f"    X: {pos.x:.4f} m")
        print(f"    Y: {pos.y:.4f} m")
        print(f"    Z: {pos.z:.4f} m  ← 抓取点高度（应该高于盒子）")
        
        ori = grasp.grasp_pose.pose.orientation
        print(f"\n  姿态 (Orientation - Quaternion):")
        print(f"    x: {ori.x:.4f}")
        print(f"    y: {ori.y:.4f}")
        print(f"    z: {ori.z:.4f}")
        print(f"    w: {ori.w:.4f}")
        
        # 转换为欧拉角（XYZ+RPY格式，适合机械臂）
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([ori.x, ori.y, ori.z, ori.w])
        euler = r.as_euler('xyz', degrees=True)
        print(f"\n  ⭐ 机械臂姿态 (XYZ + RPY):")
        print(f"    X: {pos.x:.4f} m")
        print(f"    Y: {pos.y:.4f} m") 
        print(f"    Z: {pos.z:.4f} m")
        print(f"    Roll:  {euler[0]:7.2f}°")
        print(f"    Pitch: {euler[1]:7.2f}°")
        print(f"    Yaw:   {euler[2]:7.2f}°")
        
        print("\n" + "=" * 70)
        print("✓ 抓取位姿已输出")
        print("=" * 70 + "\n")
        
        return grasp
    
    def run_once(self):
        """运行一次检测"""
        rospy.loginfo("\n开始单帧检测...")
        
        # 获取最佳抓取
        best_grasp = self.get_best_grasp()
        
        if best_grasp is None:
            rospy.logerr("检测失败！")
            return None
        
        # 打印信息并返回抓取位姿
        selected_grasp = self.print_grasp_info(best_grasp)
        
        if self.visualize:
            self.visualize_grasp()
        
        return selected_grasp

    def visualize_grasp(self):
        """使用Open3D可视化点云和所有抓取"""
        viz_start_time = time.time()
        if self.current_frame is None or self.current_grasps is None:
            rospy.logwarn("没有数据可供可视化")
            return

        rospy.loginfo("🎨 正在启动Open3D可视化窗口...")
        rospy.loginfo("   - 显示 RGB 点云")
        rospy.loginfo(f"   - 显示 {len(self.current_grasps.grasps)} 个抓取坐标系")
        rospy.loginfo("   按 'Q' 键关闭窗口")

        geometries = []

        # 1. 处理点云 (确保是 RGB)
        try:
            # 尝试读取 RGB
            gen = pc2.read_points(self.current_frame, field_names=("x", "y", "z", "rgb"), skip_nans=True)
            points_list = list(gen)
            
            if points_list:
                # ⚠️ 关键修正：必须显式指定 dtype=np.float32
                # 否则 numpy 默认使用 float64，导致 view(np.uint32) 解析 RGB 错误
                points_np = np.array(points_list, dtype=np.float32)
                
                # 降采样
                points_np = points_np[::3] 
                
                xyz = points_np[:, 0:3]
                rgb_float = points_np[:, 3]
                rgb_float = np.ascontiguousarray(rgb_float)
                
                # 解析 RGB (ROS PointCloud2 中 RGB 存储在 float32 中)
                # 这种位运算方式适用于 Little Endian 系统
                rgb_uint32 = rgb_float.view(np.uint32)
                r = ((rgb_uint32 >> 16) & 0xFF) / 255.0
                g = ((rgb_uint32 >> 8) & 0xFF) / 255.0
                b = (rgb_uint32 & 0xFF) / 255.0
                rgb = np.stack([r, g, b], axis=1)
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                geometries.append(pcd)
                rospy.loginfo(f"✓ 已生成 RGB 彩色点云 (点数: {len(xyz)})")
            else:
                rospy.logwarn("点云为空")
        except Exception as e:
            rospy.logwarn(f"RGB 点云转换失败: {e}")
            # 尝试只读取 XYZ 并显示为灰色
            try:
                gen = pc2.read_points(self.current_frame, field_names=("x", "y", "z"), skip_nans=True)
                points_list = list(gen)
                if points_list:
                    xyz = np.array(points_list, dtype=np.float32)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # 灰色
                    geometries.append(pcd)
                    rospy.loginfo("✓ 已生成灰度点云 (无 RGB 信息)")
            except:
                pass

        # 2. 添加所有抓取坐标系 (恢复为坐标系显示)
        for i, grasp in enumerate(self.current_grasps.grasps):
            # 创建坐标系
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            
            # 设置位置
            pos = grasp.grasp_pose.pose.position
            mesh.translate([pos.x, pos.y, pos.z])
            
            # 设置姿态
            ori = grasp.grasp_pose.pose.orientation
            r = Rotation.from_quat([ori.x, ori.y, ori.z, ori.w])
            mesh.rotate(r.as_matrix(), center=[pos.x, pos.y, pos.z])
            
            geometries.append(mesh)

        # 3. 显示
        if geometries:
            viz_end_time = time.time()
            viz_time_ms = (viz_end_time - viz_start_time) * 1000
            print(f"\n{'='*40}")
            print(f"🎨 Open3D 可视化准备耗时: {viz_time_ms:.1f} ms")
            print(f"{'='*40}\n")
            
            o3d.visualization.draw_geometries(geometries, window_name="Single Frame Detection Result", width=1280, height=720)
        else:
            rospy.logwarn("没有几何体可显示")


    
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='单帧抓取检测器 - 从相机获取一帧，输出抓取位姿'
    )
    parser.add_argument(
        '--no-viz', 
        action='store_true',
        help='禁用Open3D可视化（仅输出文本）'
    )
    parser.add_argument(
        '--loop',
        action='store_true',
        help='循环模式：按Enter键继续下一次检测'
    )
    
    args = parser.parse_args(rospy.myargv()[1:])
    
    try:
        detector = SingleFrameGraspDetector(visualize=not args.no_viz)
        
        if args.loop:
            # 循环模式
            rospy.loginfo("\n【循环模式】按 Enter 开始检测，Ctrl+C 退出\n")
            while not rospy.is_shutdown():
                try:
                    input("按 Enter 开始检测...")
                    detector.run_once()
                except KeyboardInterrupt:
                    break
        else:
            # 单次模式
            grasp = detector.run_once()
            if grasp is not None:
                rospy.loginfo("✓ 检测成功完成")
            else:
                rospy.logerr("✗ 检测失败")
                sys.exit(1)
        
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
