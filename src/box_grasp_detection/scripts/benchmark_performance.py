#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试 - Benchmark Performance
- 订阅 /box_grasps，统计Open3D可视化时间（单帧）
- 不做实时循环，只测一帧的可视化耗时，便于与GraspNet的Open3D结果对比
"""

import rospy
import time
import numpy as np
from box_grasp_detection.msg import BoxGraspArray
import open3d as o3d
from scipy.spatial.transform import Rotation


def make_coordinate_frame(size=0.1, translation=None, rotation_quat=None, color=None):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if translation is not None:
        mesh.translate(translation)
    if rotation_quat is not None:
        R = Rotation.from_quat(rotation_quat).as_matrix()
        mesh.rotate(R, center=translation if translation is not None else np.zeros(3))
    if color is not None:
        # 坐标系对象不支持直接改色，这里忽略color
        pass
    return mesh


def build_geometries_from_grasps(msg: BoxGraspArray):
    geoms = []
    # 添加世界坐标系
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    # 为每个抓取添加抓取坐标系
    for i, g in enumerate(msg.grasps):
        t = np.array([
            g.grasp_pose.pose.position.x,
            g.grasp_pose.pose.position.y,
            g.grasp_pose.pose.position.z,
        ])
        q = np.array([
            g.grasp_pose.pose.orientation.x,
            g.grasp_pose.pose.orientation.y,
            g.grasp_pose.pose.orientation.z,
            g.grasp_pose.pose.orientation.w,
        ])
        geoms.append(make_coordinate_frame(size=0.08, translation=t, rotation_quat=q))
    return geoms


class Open3DBenchmarkOnce:
    def __init__(self):
        rospy.init_node('open3d_benchmark_once', anonymous=True)
        self._grasps = None
        self._sub = rospy.Subscriber('/box_grasps', BoxGraspArray, self._cb, queue_size=1)
        rospy.loginfo('⏱️ Benchmark节点已启动，等待/box_grasps 单帧数据…')

    def _cb(self, msg: BoxGraspArray):
        if self._grasps is None:
            self._grasps = msg
            rospy.loginfo(f'收到抓取结果: {len(msg.grasps)} 个候选，开始Open3D可视化计时…')
            self.measure_open3d_once()

    def measure_open3d_once(self):
        if self._grasps is None:
            return
        # 构建几何体（不包含点云，仅坐标系，和GraspNet对齐方式）
        geoms = build_geometries_from_grasps(self._grasps)

        # 计时：创建窗口 + 添加几何体 + 刷新一次 + 关闭
        t0 = time.time()
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Benchmark(Open3D single frame)', width=960, height=720, visible=True)
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        # 为保证渲染真正执行，短暂sleep
        time.sleep(0.05)
        vis.destroy_window()
        t1 = time.time()
        open3d_time = t1 - t0
        
        print(f"\n{'='*40}")
        print(f"⏱️  Open3D渲染耗时 (Rendering Time)")
        print(f"{'='*40}")
        print(f"  总耗时: {open3d_time*1000:.1f} ms")
        print(f"{'='*40}\n")
        
        rospy.loginfo(f'📊 Open3D可视化耗时(单帧): {open3d_time*1000:.1f} ms')

        # 输出总结后退出
        rospy.signal_shutdown('Benchmark complete')


if __name__ == '__main__':
    try:
        Open3DBenchmarkOnce()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
