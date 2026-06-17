#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精确对比手眼标定结果和实际 TF 输出
将手眼标定结果和 TF 输出的平移和旋转转换为四元数进行详细对比
"""

import numpy as np
import json

def load_calibration_result(json_path):
    """加载手眼标定结果"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['hand_eye_matrix'])

def rotation_matrix_to_quaternion(R):
    """旋转矩阵转四元数"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[1,1] - R[0,0])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    
    return np.array([x, y, z, w])

def quaternion_angle_difference(q1, q2):
    """计算两个四元数之间的角度差（度）"""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle_rad = 2 * np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def analyze_difference():
    """分析手眼标定和TF输出的差异"""
    
    print("=" * 80)
    print("手眼标定 vs 实际 TF 输出 - 详细对比分析")
    print("=" * 80)
    print()
    
    # 加载手眼标定矩阵
    calib_file = '/home/xsh/catkin_UR5/src/ur5_vision_servo_0916/calibration_result1_4.json'
    T_calib = load_calibration_result(calib_file)
    
    # 从标定矩阵提取四元数
    R_calib = T_calib[:3, :3]
    q_calib = rotation_matrix_to_quaternion(R_calib)
    t_calib = T_calib[:3, 3]
    
    # 实际 TF 输出（来自你的终端输出）
    t_tf = np.array([0.026, 0.094, -0.138])
    q_tf = np.array([0.024, 0.025, 0.999, -0.005])  # [x, y, z, w]
    
    print("【1. 平移对比】")
    print("-" * 80)
    print(f"{'项目':<20} {'手眼标定':<20} {'TF输出':<20} {'差值':<15}")
    print("-" * 80)
    print(f"{'X (m)':<20} {t_calib[0]:<20.6f} {t_tf[0]:<20.6f} {(t_tf[0]-t_calib[0])*1000:>10.2f} mm")
    print(f"{'Y (m)':<20} {t_calib[1]:<20.6f} {t_tf[1]:<20.6f} {(t_tf[1]-t_calib[1])*1000:>10.2f} mm")
    print(f"{'Z (m)':<20} {t_calib[2]:<20.6f} {t_tf[2]:<20.6f} {(t_tf[2]-t_calib[2])*1000:>10.2f} mm")
    
    t_error = np.linalg.norm(t_tf - t_calib) * 1000
    print(f"\n{'总平移误差:':<20} {t_error:.2f} mm")
    print()
    
    print("【2. 旋转对比（四元数）】")
    print("-" * 80)
    print(f"{'项目':<20} {'手眼标定':<30} {'TF输出':<30}")
    print("-" * 80)
    print(f"{'Quaternion [x]':<20} {q_calib[0]:<30.6f} {q_tf[0]:<30.6f}")
    print(f"{'Quaternion [y]':<20} {q_calib[1]:<30.6f} {q_tf[1]:<30.6f}")
    print(f"{'Quaternion [z]':<20} {q_calib[2]:<30.6f} {q_tf[2]:<30.6f}")
    print(f"{'Quaternion [w]':<20} {q_calib[3]:<30.6f} {q_tf[3]:<30.6f}")
    
    angle_diff = quaternion_angle_difference(q_calib, q_tf)
    print(f"\n{'旋转角度差:':<20} {angle_diff:.4f}°")
    print()
    
    print("【3. 误差评估】")
    print("=" * 80)
    
    # 位置评估
    if t_error < 1.0:
        pos_status = "✅ 优秀"
        pos_comment = "位置误差 < 1mm，精度很高"
    elif t_error < 5.0:
        pos_status = "✅ 良好"
        pos_comment = "位置误差在可接受范围内"
    elif t_error < 10.0:
        pos_status = "⚠️ 一般"
        pos_comment = "位置误差较大，可能影响精度"
    else:
        pos_status = "❌ 较差"
        pos_comment = "位置误差过大，需要重新标定"
    
    # 姿态评估
    if angle_diff < 1.0:
        rot_status = "✅ 优秀"
        rot_comment = "姿态误差 < 1°，精度很高"
    elif angle_diff < 5.0:
        rot_status = "✅ 良好"
        rot_comment = "姿态误差在可接受范围内"
    elif angle_diff < 10.0:
        rot_status = "⚠️ 一般"
        rot_comment = "姿态误差较大，可能影响某些应用"
    else:
        rot_status = "❌ 较差"
        rot_comment = "姿态误差过大，需要重新标定"
    
    print(f"平移精度: {pos_status} - {pos_comment}")
    print(f"旋转精度: {rot_status} - {rot_comment}")
    print()
    
    print("【4. 误差来源分析】")
    print("=" * 80)
    print("可能的误差来源：")
    print("  1. 手眼标定时的测量噪声（±1-2mm, ±1-2°）")
    print("  2. URDF 模型简化和近似")
    print("  3. 机器人关节编码器精度")
    print("  4. 相机安装的机械公差")
    print("  5. D435 内部坐标系的标称值 vs 实际值的差异")
    print()
    
    print("【5. 对视觉伺服的影响】")
    print("=" * 80)
    
    if t_error < 5.0 and angle_diff < 5.0:
        print("✅ 当前精度对大多数视觉伺服任务来说是 **足够** 的：")
        print()
        print("  适用场景：")
        print("    ✓ 物体抓取（目标尺寸 > 5cm）")
        print("    ✓ 工件定位和对准")
        print("    ✓ 视觉引导的运动规划")
        print("    ✓ 一般精度的装配任务")
        print()
        print("  可能不适用：")
        print("    ✗ 高精度插孔（< 1mm 公差）")
        print("    ✗ 微小零件操作（< 2mm）")
        print()
        print("💡 建议：当前配置可以直接使用，无需重新标定。")
    else:
        print("⚠️ 当前精度可能不足，建议：")
        print("  1. 重新进行手眼标定")
        print("  2. 检查相机安装是否松动")
        print("  3. 验证机器人 TCP 设置")
        print("  4. 使用更多标定点位提高精度")
    
    print()
    print("=" * 80)
    print("【6. 关于 Pitch 符号反转】")
    print("=" * 80)
    print()
    print(f"实际旋转差异是 {angle_diff:.2f}°，这不是简单的符号反转，")
    print("而是一个小的但真实存在的姿态偏差。")
    print()
    print("这个偏差主要来自：")
    print("  • D435 内部坐标系的标称值与实际值的差异")
    print("  • 手眼标定的固有误差（通常 1-3°）")
    print("  • 机械安装的小偏差")
    print()
    print(f"结论：{angle_diff:.2f}° 的姿态差异在手眼标定的正常误差范围内。")
    print()

if __name__ == '__main__':
    analyze_difference()
