#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 MoveIt 计算给定位姿 (x,y,z, rx,ry,rz) 对应的UR5 关节角。
前提：已启动 move_group（例如 ur5_gripper_moveit_config/launch/demo.launch 或你的实际机器人 bringup）。
"""

import sys
import math
import time
import rospy
import numpy as np
import cv2

from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
import tf.transformations as tf_trans


def rotvec_to_quat(rx, ry, rz):
    """将 UR 的旋转向量 (Rodrigues) 转四元数 [x,y,z,w]"""
    rvec = np.array([[rx], [ry], [rz]], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    qx, qy, qz, qw = tf_trans.quaternion_from_matrix(T)
    return qx, qy, qz, qw


def compute_ik(x, y, z, rx, ry, rz, group_name='manipulator', ee_link='ee_link', frame_id='base_link'):
    roscpp_initialize(sys.argv)
    rospy.init_node('compute_ik_pose', anonymous=True)

    robot = RobotCommander()
    group = MoveGroupCommander(group_name)
    try:
        group.set_end_effector_link(ee_link)
    except Exception:
        pass  # 若 SRDF 已设置为 ee_link，可忽略
    group.set_pose_reference_frame(frame_id)

    # 目标位姿
    qx, qy, qz, qw = rotvec_to_quat(rx, ry, rz)
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw

    # 调用 /compute_ik 服务
    rospy.loginfo("等待 /compute_ik 服务...")
    rospy.wait_for_service('/compute_ik', timeout=10)
    ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)

    req = GetPositionIKRequest()
    req.ik_request.group_name = group_name
    req.ik_request.ik_link_name = ee_link
    req.ik_request.pose_stamped = pose
    # 某些 MoveIt 版本无 attempts 字段，做兼容并延长超时
    req.ik_request.timeout = rospy.Duration(5.0)
    if hasattr(req.ik_request, 'attempts'):
        req.ik_request.attempts = 5
    # 明确启用避障（可按需关闭）
    if hasattr(req.ik_request, 'avoid_collisions'):
        req.ik_request.avoid_collisions = True
    req.ik_request.robot_state = robot.get_current_state()

    resp = ik_srv(req)
    if resp.error_code.val != resp.error_code.SUCCESS:
        rospy.logerr("IK 失败，错误码: %d", resp.error_code.val)
        sys.exit(1)

    # 只提取本规划组的关节
    active_joints = group.get_active_joints()
    name_to_pos = {n: p for n, p in zip(resp.solution.joint_state.name, resp.solution.joint_state.position)}
    joints_rad = [name_to_pos[n] for n in active_joints if n in name_to_pos]
    joints_deg = [math.degrees(j) for j in joints_rad]

    print("\n目标位姿 (base -> {}):".format(ee_link))
    print("  位置 [m]:  x={:.6f}, y={:.6f}, z={:.6f}".format(x, y, z))
    print("  朝向 [rx,ry,rz]:  [{:.6f}, {:.6f}, {:.6f}]".format(rx, ry, rz))

    print("\n规划组: {}".format(group_name))
    print("末端链接: {}".format(group.get_end_effector_link()))

    print("\n关节顺序:")
    for i, n in enumerate(active_joints):
        print("  {}: {}".format(i+1, n))

    print("\n解 (弧度):")
    print([round(v, 6) for v in joints_rad])
    print("解 (度):")
    print([round(v, 3) for v in joints_deg])

    return active_joints, joints_rad, joints_deg


if __name__ == '__main__':
    # 你的习惯位姿 (x, y, z, rx, ry, rz)
    x, y, z = -0.478, -0.0678, 0.336
    rx, ry, rz = 2.222, -2.22, -0.140

    try:
        compute_ik(x, y, z, rx, ry, rz)
    except rospy.ROSException as e:
        print("ROS 异常:", e)
        print("请先启动 move_group，例如:\n  roslaunch ur5_gripper_moveit_config demo.launch use_gui:=true")
        sys.exit(1)


# 目标位姿 (base -> ee_link):
#   位置 [m]:  x=-0.478000, y=-0.067800, z=0.336000
#   朝向 [rx,ry,rz]:  [2.222000, -2.220000, -0.140000]

# 规划组: manipulator
# 末端链接: ee_link

# 关节顺序:
#   1: shoulder_pan_joint
#   2: shoulder_lift_joint
#   3: elbow_joint
#   4: wrist_1_joint
#   5: wrist_2_joint
#   6: wrist_3_joint

# 解 (弧度):
# [3.083827, -1.631521, 1.521992, -1.526241, -1.631839, 3.080942]
# 解 (度):
# [176.69, -93.479, 87.204, -87.447, -93.497, 176.525]