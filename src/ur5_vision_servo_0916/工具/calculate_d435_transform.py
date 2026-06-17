#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 D435 相机的正确 URDF 变换参数
基于手眼标定结果和 D435 的内部坐标系偏移
用手眼标定矩阵和各个坐标系的偏移，结合起来综合计算挂载在 ee_link 上的矩阵
"""

import numpy as np
import json

def load_calibration_result(json_path):
    """加载手眼标定结果"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['hand_eye_matrix'])

def rpy_to_rotation_matrix(roll, pitch, yaw):
    """RPY角度转旋转矩阵（ZYX顺序）"""
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def rotation_matrix_to_rpy(rot_matrix):
    """旋转矩阵转RPY角度"""
    # 提取 roll, pitch, yaw
    sy = np.sqrt(rot_matrix[0,0]**2 + rot_matrix[1,0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(rot_matrix[2,1], rot_matrix[2,2])
        pitch = np.arctan2(-rot_matrix[2,0], sy)
        yaw = np.arctan2(rot_matrix[1,0], rot_matrix[0,0])
    else:
        roll = np.arctan2(-rot_matrix[1,2], rot_matrix[1,1])
        pitch = np.arctan2(-rot_matrix[2,0], sy)
        yaw = 0
    
    return np.array([roll, pitch, yaw])

def compute_d435_urdf_transform(T_ee_co):
    """
    计算 D435 相机的 URDF 变换参数
    
    参数:
        T_ee_co: 4x4 手眼标定矩阵 (ee_link -> camera_color_optical_frame)
    
    返回:
        xyz, rpy: 用于 URDF <origin> 标签的参数
    """
    
    print("=" * 70)
    print("RealSense D435 URDF 变换参数计算")
    print("=" * 70)
    print()
    
    # 1. 手眼标定矩阵
    print("【步骤 1】手眼标定结果 (ee_link -> camera_color_optical_frame):")
    print(f"平移: [{T_ee_co[0,3]:.8f}, {T_ee_co[1,3]:.8f}, {T_ee_co[2,3]:.8f}]")
    rpy_ee_co = rotation_matrix_to_rpy(T_ee_co[:3,:3])
    print(f"RPY: [{rpy_ee_co[0]:.8f}, {rpy_ee_co[1]:.8f}, {rpy_ee_co[2]:.8f}]")
    print()
    
    # 2. D435 内部坐标系偏移 (根据 _d435.urdf.xacro)
    print("【步骤 2】D435 内部坐标系偏移 (camera_bottom_screw_frame -> camera_color_optical_frame):")
    
    # D435 的内部参数（来自 _d435.urdf.xacro）
    d435_cam_mount_from_center_offset = 0.0149
    d435_glass_to_front = 0.0001
    d435_zero_depth_to_glass = 0.0042
    d435_mesh_x_offset = d435_cam_mount_from_center_offset - d435_glass_to_front - d435_zero_depth_to_glass
    
    d435_cam_depth_py = 0.0175
    d435_cam_height = 0.025
    d435_cam_depth_pz = d435_cam_height / 2.0
    
    d435_cam_depth_to_color_offset = 0.015
    
    # bottom_screw -> camera_link
    t1 = np.array([d435_mesh_x_offset, d435_cam_depth_py, d435_cam_depth_pz])
    
    # camera_link -> camera_color_frame (仅Y方向平移)
    t2 = np.array([0.0, d435_cam_depth_to_color_offset, 0.0])
    
    # camera_color_frame -> camera_color_optical_frame (旋转)
    # RPY = (-pi/2, 0, -pi/2)
    rpy_optical = np.array([-np.pi/2, 0, -np.pi/2])
    R_optical = rpy_to_rotation_matrix(*rpy_optical)
    
    # 合并变换: bottom_screw -> color_optical
    t_bs_co = t1 + t2  # 平移叠加
    
    print(f"  bottom_screw -> camera_link: [{t1[0]:.6f}, {t1[1]:.6f}, {t1[2]:.6f}]")
    print(f"  camera_link -> camera_color_frame: [{t2[0]:.6f}, {t2[1]:.6f}, {t2[2]:.6f}]")
    print(f"  camera_color_frame -> camera_color_optical_frame: RPY = [-π/2, 0, -π/2]")
    print(f"  合并后平移: [{t_bs_co[0]:.6f}, {t_bs_co[1]:.6f}, {t_bs_co[2]:.6f}]")
    print()
    
    # 构建 T_bs_co 矩阵
    T_bs_co = np.eye(4)
    T_bs_co[:3,:3] = R_optical
    T_bs_co[:3,3] = t_bs_co
    
    # 3. 计算 T_ee_bs = T_ee_co * inv(T_bs_co)
    print("【步骤 3】计算 URDF 变换参数 (ee_link -> camera_bottom_screw_frame):")
    T_bs_co_inv = np.linalg.inv(T_bs_co)
    T_ee_bs = T_ee_co @ T_bs_co_inv
    
    xyz = T_ee_bs[:3, 3]
    rpy = rotation_matrix_to_rpy(T_ee_bs[:3, :3])
    
    print(f"平移 (xyz): [{xyz[0]:.8f}, {xyz[1]:.8f}, {xyz[2]:.8f}]")
    print(f"旋转 (rpy): [{rpy[0]:.8f}, {rpy[1]:.8f}, {rpy[2]:.8f}]")
    print()
    
    # 4. 验证：T_ee_co_check = T_ee_bs * T_bs_co
    print("【步骤 4】验证计算结果:")
    T_ee_co_check = T_ee_bs @ T_bs_co
    xyz_check = T_ee_co_check[:3, 3]
    rpy_check = rotation_matrix_to_rpy(T_ee_co_check[:3, :3])
    
    print(f"重建的 ee_link -> camera_color_optical_frame:")
    print(f"  平移: [{xyz_check[0]:.8f}, {xyz_check[1]:.8f}, {xyz_check[2]:.8f}]")
    print(f"  RPY:  [{rpy_check[0]:.8f}, {rpy_check[1]:.8f}, {rpy_check[2]:.8f}]")
    
    diff_xyz = np.abs(xyz_check - T_ee_co[:3,3])
    diff_rpy = np.abs(rpy_check - rpy_ee_co)
    print(f"  平移误差: [{diff_xyz[0]*1000:.3f}, {diff_xyz[1]*1000:.3f}, {diff_xyz[2]*1000:.3f}] mm")
    print(f"  旋转误差: [{np.degrees(diff_rpy[0]):.3f}, {np.degrees(diff_rpy[1]):.3f}, {np.degrees(diff_rpy[2]):.3f}] deg")
    
    if np.all(diff_xyz < 0.001) and np.all(diff_rpy < 0.01):
        print("  ✅ 验证通过！误差在可接受范围内。")
    else:
        print("  ⚠️ 警告：验证误差较大，请检查计算。")
    print()
    
    return xyz, rpy

def generate_urdf_snippet(xyz, rpy):
    """生成 URDF 代码片段"""
    print("=" * 70)
    print("【URDF 代码片段】复制以下内容到你的 URDF 文件:")
    print("=" * 70)
    print()
    print(f'  <xacro:sensor_d435 parent="ee_link" use_nominal_extrinsics="true" add_plug="false" use_mesh="true">')
    print(f'    <origin xyz="{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}" rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"/>')
    print(f'  </xacro:sensor_d435>')
    print()
    print("=" * 70)

if __name__ == '__main__':
    # 加载手眼标定结果
    calib_file = '/home/xsh/catkin_UR5/src/ur5_vision_servo_0916/calibration_result1_4.json'
    
    print("\n正在加载手眼标定结果...")
    T_ee_co = load_calibration_result(calib_file)
    print(f"✅ 成功加载: {calib_file}\n")
    
    # 计算 URDF 变换参数
    xyz, rpy = compute_d435_urdf_transform(T_ee_co)
    
    # 生成 URDF 代码片段
    generate_urdf_snippet(xyz, rpy)
    
    print("\n💡 使用说明:")
    print("1. 复制上面的 URDF 代码片段")
    print("2. 替换 ur5_gripper_joint_limited_robot.urdf.xacro 中的相应部分")
    print("3. 重新编译: cd ~/catkin_UR5 && catkin_make")
    print("4. 重启 ROS: roslaunch ur_planning start_robot_only.launch")
    print("5. 验证: rosrun tf tf_echo ee_link camera_color_optical_frame")
    print("\n期望结果应该与手眼标定矩阵一致（误差 < 1mm, < 0.1°）\n")
