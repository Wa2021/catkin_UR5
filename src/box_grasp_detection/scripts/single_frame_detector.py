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
from geometry_msgs.msg import PoseStamped
from box_grasp_detection.msg import BoxGraspArray, BoxGrasp
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
        
        # 为Open3D可视化器发布所有抓取
        self.all_grasps_pub = rospy.Publisher(
            '/all_box_grasps',
            BoxGraspArray,
            queue_size=1
        )
        
        rospy.loginfo("=" * 70)
        rospy.loginfo("单帧抓取检测器已启动")
        rospy.loginfo("=" * 70)

        # 性能统计
        self._last_cloud_stamp = None
        self._alg_start_time = None
        self._alg_end_time = None


    
    def cloud_callback(self, msg):
        """接收点云帧"""
        with self.lock:
            if not self.frame_received.is_set():
                self.current_frame = msg
                # 记录算法开始时间
                self._alg_start_time = time.time()
                self.frame_received.set()
                rospy.loginfo("✓ 已接收点云帧")
    
    def grasp_callback(self, msg):
        """接收抓取结果"""
        with self.lock:
            if not self.grasps_received.is_set():
                self.current_grasps = msg
                self.grasps_received.set()
                
                # 算法结束时间（收到抓取结果）
                self._alg_end_time = time.time()
          
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
            
            # 发布所有抓取给Open3D可视化器
            rospy.loginfo(f"📡 发布所有 {len(self.current_grasps.grasps)} 个抓取给Open3D可视化器...")
            # 注意：realtime_open3d_viz.py 订阅的是 /box_grasps，所以这里不需要额外发布
            # 如果需要专门发布给可视化器，可以使用 self.all_grasps_pub
            # 但目前系统架构中，box_grasp_planner.py 已经发布了 /box_grasps
            # 所以这里我们只需要确保可视化器订阅了正确的话题即可
            
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
        
        # 修正抓取位姿（如果需要）
        corrected_grasp = self.correct_grasp_pose(grasp)
        
        # 抓取位姿（关键输出！）
        print(f"\n【抓取位姿 - 6D Pose】")
        print(f"  坐标系: {corrected_grasp.grasp_pose.header.frame_id}")
        
        pos = corrected_grasp.grasp_pose.pose.position
        print(f"\n  位置 (Position):")
        print(f"    X: {pos.x:.4f} m")
        print(f"    Y: {pos.y:.4f} m")
        print(f"    Z: {pos.z:.4f} m  ← 抓取点高度（应该高于盒子）")
        
        ori = corrected_grasp.grasp_pose.pose.orientation
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
        
        # 使用修正后的抓取位姿进行可视化
        return corrected_grasp
    
    def correct_grasp_pose(self, grasp):
        """修正错误的抓取位姿"""
        # 🔧 修改：直接使用C++发送的姿态，不再强制修改
        # C++代码已经正确设置了抓取姿态
        rospy.loginfo("✓ 使用C++计算的抓取姿态（不做修正）")
        return grasp
    
    def visualize_grasp(self, grasp):
        """使用Open3D可视化（简化版）"""
        try:
            import open3d as o3d
            from scipy.spatial.transform import Rotation
        except ImportError:
            rospy.logwarn("未安装Open3D，跳过可视化")
            return
        
        rospy.loginfo("正在生成Open3D可视化...")
        
        # 转换点云
        points = []
        colors = []
        for point in pc2.read_points(self.current_frame, skip_nans=True):
            x, y, z = point[:3]
            points.append([x, y, z])
            if len(point) > 3:
                rgb = point[3]
                r = (int(rgb) >> 16) & 0xFF
                g = (int(rgb) >> 8) & 0xFF
                b = int(rgb) & 0xFF
                colors.append([r/255.0, g/255.0, b/255.0])
            else:
                colors.append([0.5, 0.5, 0.5])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='药盒抓取位姿检测', width=1280, height=720)
        vis.add_geometry(pcd)
        
        # 1. 相机坐标系（原点）
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.15, origin=[0, 0, 0]
        )
        vis.add_geometry(camera_frame)
        
        # 参考说明：相机坐标系定义
        # X轴（红色）：指向右侧
        # Y轴（绿色）：指向下方
        # Z轴（蓝色）：指向场景深度（远离相机）
        # 注意：这里没有添加参考平面，避免视觉混淆
        
        # 2. 药盒边界框（只显示选中的这一个）
        box_center = np.array([
            grasp.box_pose.position.x,
            grasp.box_pose.position.y,
            grasp.box_pose.position.z
        ])
        
        bbox = o3d.geometry.OrientedBoundingBox()
        bbox.center = box_center
        bbox.extent = np.array([grasp.length, grasp.width, grasp.height])
        
        # 盒子朝向
        box_quat = [grasp.box_pose.orientation.x, grasp.box_pose.orientation.y, 
                   grasp.box_pose.orientation.z, grasp.box_pose.orientation.w]
        try:
            bbox.R = Rotation.from_quat(box_quat).as_matrix()
        except:
            bbox.R = np.eye(3)  # 如果四元数无效，使用单位矩阵
            
        bbox.color = [1, 0, 0]  # 红色边界框
        vis.add_geometry(bbox)
        
        # 3. 抓取坐标系（使用C++发布的位姿）
        grasp_pos = np.array([
            grasp.grasp_pose.pose.position.x,
            grasp.grasp_pose.pose.position.y,
            grasp.grasp_pose.pose.position.z
        ])
        
        # 从消息中读取四元数姿态（C++计算的结果）
        grasp_quat = [
            grasp.grasp_pose.pose.orientation.x,
            grasp.grasp_pose.pose.orientation.y,
            grasp.grasp_pose.pose.orientation.z,
            grasp.grasp_pose.pose.orientation.w
        ]
        
        try:
            grasp_rot = Rotation.from_quat(grasp_quat).as_matrix()
        except:
            rospy.logwarn("四元数转换失败，使用单位矩阵")
            grasp_rot = np.eye(3)
        
        # 抓取坐标系（绿色）
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.12, origin=grasp_pos
        )
        grasp_frame.rotate(grasp_rot, center=grasp_pos)
        vis.add_geometry(grasp_frame)
        
        # 4. 抓取点标记
        grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        grasp_sphere.translate(grasp_pos)
        grasp_sphere.paint_uniform_color([0, 1, 0])  # 绿色球
        vis.add_geometry(grasp_sphere)
        
        # 调试信息 - 详细位置分析
        rospy.loginfo("=" * 50)
        rospy.loginfo("📍 坐标系位置分析:")
        rospy.loginfo(f"📷 相机坐标系: (0, 0, 0)")
        rospy.loginfo(f"🤖 抓取位置: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        rospy.loginfo(f"📦 药盒中心: ({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f})")
        rospy.loginfo(f"📏 Z轴分析:")
        rospy.loginfo(f"   - 相机到抓取: {grasp_pos[2]:.3f}m")
        rospy.loginfo(f"   - 相机到药盒: {box_center[2]:.3f}m") 
        rospy.loginfo(f"   - 差值: {box_center[2] - grasp_pos[2]:.3f}m")
        rospy.loginfo("")
        rospy.loginfo("🎯 关键理解:")
        rospy.loginfo("   - Z值越小 = 距离相机越近")
        rospy.loginfo("   - 抓取点应该在药盒和相机之间")
        rospy.loginfo(f"   - 抓取点Z({grasp_pos[2]:.3f}) vs 药盒Z({box_center[2]:.3f})")
        
        if grasp_pos[2] < box_center[2]:
            rospy.loginfo("✅ 位置关系正确：抓取点更靠近相机（在药盒上方）")
        else:
            rospy.logerr("❌ 位置关系错误：抓取点比药盒更远（错误！）")
        rospy.loginfo("=" * 50)
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        
        rospy.loginfo("=" * 70)
        rospy.loginfo("Open3D窗口已打开 - 按 Q 关闭")
        rospy.loginfo("")
        rospy.loginfo("🎨 颜色说明:")
        rospy.loginfo("  🔴 红色边界框 = 检测到的药盒")
        rospy.loginfo("  🟢 绿色坐标系和球 = 抓取位姿")
        rospy.loginfo("  ⚪ 白色坐标系 = 相机位置(原点)")
        rospy.loginfo("")
        rospy.loginfo("📐 坐标系说明:")
        rospy.loginfo("  - 相机在原点(0,0,0)，Z轴指向场景深度方向")
        rospy.loginfo("  - Z值越大 = 距离相机越远")
        rospy.loginfo("  - 在眼在手系统中，相机在机械臂末端，朝下看桌面")
        rospy.loginfo("  - 抓取点Z > 药盒Z = 抓取点在药盒上方（从机械臂角度）")
        rospy.loginfo("=" * 70)
        
        vis.run()
        vis.destroy_window()
        
        rospy.loginfo("可视化窗口已关闭")
    
    def run_once(self):
        """运行一次检测"""
        rospy.loginfo("\n开始单帧检测...")
        
        # 获取最佳抓取
        best_grasp = self.get_best_grasp()
        
        if best_grasp is None:
            rospy.logerr("检测失败！")
            return None
        
        # 打印信息并获取修正后的抓取位姿
        corrected_grasp = self.print_grasp_info(best_grasp)
        
        # 可视化修正后的抓取位姿
        if self.visualize:
            self.visualize_grasp(corrected_grasp)
        
        return corrected_grasp

    def visualize_grasp(self, best_grasp):
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
