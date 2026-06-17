#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机械臂姿态单轴与视觉姿态误差映射验证脚本

核心思路：
1. 不再直接尝试把“图像里的 pitch/roll”硬映射到 base 的 rx/ry。
2. 直接使用 ArUco 求出的平面法向量，经过
   相机(虚拟) -> 相机(真实) -> tool -> base
   的旋转链路，得到 base 坐标系下的法向量。
3. 用 base 系中的法向误差向量来判断 rx/ry 的控制律：
      tilt_err_base = normal_base x target_normal_base

如果这个量在单轴微扰测试下接近：
    d(tilt_err_x)/d(roll_cmd)  ~= -1
    d(tilt_err_y)/d(pitch_cmd) ~= -1
那就说明视觉伺服里直接用 base 系法向误差做 wx/wy 控制是合理的，
而不需要再去猜“图像 pitch 对应 wx 还是 wy”。
"""

import json
import math
import os
import time

import cv2
import cv2.aruco as aruco
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from UR_Robot import UR_Robot


def normalize_angle_to_half_pi(angle):
    while angle > math.pi / 2:
        angle -= math.pi
    while angle <= -math.pi / 2:
        angle += math.pi
    return angle


def get_normal_from_rvec(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    normal_in_cam = rotation_matrix[:, 2]

    # 统一为“朝向工作台上方”的那一支法向，便于后续和 base Z 比较。
    if normal_in_cam[2] < 0:
        normal_in_cam = -normal_in_cam

    return normal_in_cam


def vector_to_str(vec):
    return "[" + ", ".join(f"{x:+.3f}" for x in vec) + "]"


class PostureVisionMappingTest:
    def __init__(self):
        rospy.init_node('test_posture_vision_mapping', anonymous=True)
        self.bridge = CvBridge()

        if hasattr(aruco, "getPredefinedDictionary"):
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        else:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

        if hasattr(aruco, "DetectorParameters_create"):
            self.parameters = aruco.DetectorParameters_create()
        else:
            self.parameters = aruco.DetectorParameters()

        self._load_calibration()

        self.aruco_marker_size = 0.05
        self.target_normal_base = np.array([0.0, 0.0, 1.0], dtype=float)

        self.latest_ex = 0.0
        self.latest_ey = 0.0
        self.latest_yaw = 0.0
        self.latest_normal_cam = None
        self.detected = False

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.loginfo("Subscribed to camera, waiting for images...")
        time.sleep(2)

        self.robot = UR_Robot(tcp_host_ip="192.168.0.1", is_use_robotiq85=False, is_use_camera=False)

    def _load_calibration(self):
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_result1_4.json')
        with open(json_path, 'r') as f:
            calib_data = json.load(f)

        self.camera_matrix = np.array(calib_data["camera_matrix"], dtype=float)
        self.dist_coeffs = np.array(calib_data["dist_coeffs"], dtype=float)
        self.R_cam2tool = np.array(calib_data["hand_eye_matrix"], dtype=float)[:3, :3]

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except Exception:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        height, width = cv_image.shape[:2]
        u0, v0 = width / 2.0, height / 2.0

        if ids is not None and len(corners) > 0:
            c = corners[0][0]
            u = np.mean(c[:, 0])
            v = np.mean(c[:, 1])

            self.latest_ex = u - u0
            self.latest_ey = v - v0

            dx = c[1][0] - c[0][0]
            dy = c[1][1] - c[0][1]
            raw_angle = math.atan2(dy, dx)
            self.latest_yaw = normalize_angle_to_half_pi(raw_angle)

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, self.aruco_marker_size, self.camera_matrix, self.dist_coeffs
            )

            normal_virtual_cam = get_normal_from_rvec(rvec[0])

            # 图像被 rotate(180) 后，PnP 解出的姿态属于“虚拟相机坐标系”，
            # 相对真实相机光学系，X/Y 需要翻回去。
            self.latest_normal_cam = np.array(
                [-normal_virtual_cam[0], -normal_virtual_cam[1], normal_virtual_cam[2]],
                dtype=float
            )
            self.latest_normal_cam /= np.linalg.norm(self.latest_normal_cam)
            self.detected = True

            text = (
                f"ex={self.latest_ex:.1f} ey={self.latest_ey:.1f} "
                f"yaw={math.degrees(self.latest_yaw):.1f}dg "
                f"n_cam={vector_to_str(self.latest_normal_cam)}"
            )
            cv2.putText(cv_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.drawFrameAxes(
                cv_image, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], self.aruco_marker_size * 0.5
            )
        else:
            self.detected = False
            self.latest_normal_cam = None
            cv2.putText(cv_image, "No ArUco detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Posture Vision Mapping Test", cv_image)
        cv2.waitKey(1)

    def _build_measurement_from_current_pose(self):
        if not self.detected or self.latest_normal_cam is None:
            return None

        tcp_pose = self.robot.get_current_tcp_pose()
        R_tool2base, _ = cv2.Rodrigues(np.array(tcp_pose[3:6], dtype=float))

        normal_base = R_tool2base @ self.R_cam2tool @ self.latest_normal_cam
        normal_base /= np.linalg.norm(normal_base)

        tilt_err_base = np.cross(normal_base, self.target_normal_base)

        return {
            "tcp_pose": np.array(tcp_pose, dtype=float),
            "ex": float(self.latest_ex),
            "ey": float(self.latest_ey),
            "yaw": float(self.latest_yaw),
            "normal_cam": self.latest_normal_cam.copy(),
            "normal_base": normal_base,
            "tilt_err_base": tilt_err_base,
        }

    def print_current_visual_state(self):
        measurement = self._build_measurement_from_current_pose()
        if measurement is None:
            print("No ArUco detected")
            return

        print(
            f"yaw={measurement['yaw']:.3f} rad ({math.degrees(measurement['yaw']):+.1f}°), "
            f"ex={measurement['ex']:+.1f}, ey={measurement['ey']:+.1f}, "
            f"normal_base={vector_to_str(measurement['normal_base'])}, "
            f"tilt_err_base={vector_to_str(measurement['tilt_err_base'])}"
        )

    def run_observe_mode(self):
        print("\n" + "=" * 60)
        print("进入阶段 A: 纯观测模式 (按 Ctrl+C 退出)")
        print("建议手动做以下动作，然后观察 normal_base 和 tilt_err_base：")
        print("1. 让 ArUco 平放")
        print("2. 分别沿物理 x / y 方向轻微翘起")
        print("3. 绕法向做少量 yaw 旋转")
        print("=" * 60 + "\n")

        try:
            while not rospy.is_shutdown():
                self.print_current_visual_state()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    def move_and_measure(self, name, rpy_pose, k_acc_param, k_vel_param):
        print(f"\n[{name}] 正在移动...")
        self.robot.move_j_p_1(rpy_pose, k_acc=k_acc_param, k_vel=k_vel_param)

        print(f"[{name}] 等待 3 秒以确保机械臂运动完毕并获取最新图像...")
        time.sleep(3.0)

        measurement = self._build_measurement_from_current_pose()
        if measurement is None:
            print(f"[{name}] 警告: 未检测到 ArUco 码！")
            return None

        print(
            f"[{name}] yaw={math.degrees(measurement['yaw']):+.2f}°, "
            f"normal_base={vector_to_str(measurement['normal_base'])}, "
            f"tilt_err_base={vector_to_str(measurement['tilt_err_base'])}"
        )
        return measurement

    def _print_axis_delta(self, label, plus_measurement, minus_measurement, base_measurement, step_rad):
        if plus_measurement is None or minus_measurement is None or base_measurement is None:
            print(f"{label}: 数据不足，无法估计导数")
            return

        plus_delta = plus_measurement["tilt_err_base"] - base_measurement["tilt_err_base"]
        minus_delta = minus_measurement["tilt_err_base"] - base_measurement["tilt_err_base"]
        central_diff = (plus_measurement["tilt_err_base"] - minus_measurement["tilt_err_base"]) / (2.0 * step_rad)

        print(label)
        print(f"  +2° 时 tilt_err_base 变化: {vector_to_str(plus_delta)}")
        print(f"  -2° 时 tilt_err_base 变化: {vector_to_str(minus_delta)}")
        print(f"  中心差分导数 d(err)/d(cmd): {vector_to_str(central_diff)}  (单位: rad/rad)")

    def run_robot_test(self):
        print("\n" + "=" * 60)
        print("进入阶段 B: 机械臂单轴姿态测试")
        print("这里不再直接看图像 pitch/roll，而是直接看 base 坐标系下的法向误差。")
        print("如果结果接近单位轴响应，就说明你的 wx/wy 该直接用 base 系法向误差。")
        print("=" * 60)

        x_base, y_base, z_base = -0.478, -0.0678, 0.336
        rx, ry, rz = 2.222, -2.22, -0.140

        try:
            rpy_tuple = self.robot.rotvec_to_rpy(x_base, y_base, z_base, rx, ry, rz)
            base_pose_rpy = list(rpy_tuple)
        except Exception as e:
            rospy.logerr(f"位姿转换失败: {e}")
            return

        print(f"基准位姿 (x,y,z,r,p,y): {[round(x, 4) for x in base_pose_rpy]}")
        input("按下 Enter 键开始移动到基准位置（请确保安全，机械臂将开始缓慢移动）...")

        k_acc = 0.2
        k_vel = 0.2
        step_rad = math.radians(2.0)

        base_m = self.move_and_measure("1. Base Position", base_pose_rpy, k_acc, k_vel)

        pose_r_pos = base_pose_rpy.copy()
        pose_r_pos[3] += step_rad
        roll_pos_m = self.move_and_measure("2. Roll +2° (绕 X)", pose_r_pos, k_acc, k_vel)
        self.move_and_measure("Return to Base", base_pose_rpy, k_acc, k_vel)

        pose_r_neg = base_pose_rpy.copy()
        pose_r_neg[3] -= step_rad
        roll_neg_m = self.move_and_measure("3. Roll -2° (绕 X)", pose_r_neg, k_acc, k_vel)
        self.move_and_measure("Return to Base", base_pose_rpy, k_acc, k_vel)

        pose_p_pos = base_pose_rpy.copy()
        pose_p_pos[4] += step_rad
        pitch_pos_m = self.move_and_measure("4. Pitch +2° (绕 Y)", pose_p_pos, k_acc, k_vel)
        self.move_and_measure("Return to Base", base_pose_rpy, k_acc, k_vel)

        pose_p_neg = base_pose_rpy.copy()
        pose_p_neg[4] -= step_rad
        pitch_neg_m = self.move_and_measure("5. Pitch -2° (绕 Y)", pose_p_neg, k_acc, k_vel)
        self.move_and_measure("Return to Base", base_pose_rpy, k_acc, k_vel)

        pose_y_pos = base_pose_rpy.copy()
        pose_y_pos[5] += step_rad
        yaw_pos_m = self.move_and_measure("6. Yaw +2° (绕 Z)", pose_y_pos, k_acc, k_vel)
        self.move_and_measure("Return to Base", base_pose_rpy, k_acc, k_vel)

        pose_y_neg = base_pose_rpy.copy()
        pose_y_neg[5] -= step_rad
        yaw_neg_m = self.move_and_measure("7. Yaw -2° (绕 Z)", pose_y_neg, k_acc, k_vel)
        self.move_and_measure("Return to Base (Final)", base_pose_rpy, k_acc, k_vel)

        print("\n" + "=" * 60)
        print("测 试 结 果 总 结")
        print("=" * 60)

        if base_m is None:
            print("未能获取基准姿态数据，无法生成总结。")
            return

        print(f"基准 yaw: {math.degrees(base_m['yaw']):+.2f}°")
        print(f"基准 normal_base: {vector_to_str(base_m['normal_base'])}")
        print(f"基准 tilt_err_base: {vector_to_str(base_m['tilt_err_base'])}")
        print("-" * 60)

        self._print_axis_delta("Roll 轴测试", roll_pos_m, roll_neg_m, base_m, step_rad)
        print("-" * 60)
        self._print_axis_delta("Pitch 轴测试", pitch_pos_m, pitch_neg_m, base_m, step_rad)
        print("-" * 60)

        if yaw_pos_m is not None and yaw_neg_m is not None:
            yaw_gain = (yaw_pos_m["yaw"] - yaw_neg_m["yaw"]) / (2.0 * step_rad)
            print(f"Yaw 轴中心差分导数 d(yaw_vis)/d(yaw_cmd): {yaw_gain:+.3f}  (单位: rad/rad)")
        else:
            print("Yaw 轴测试: 数据不足，无法估计导数")

        print("-" * 60)
        print("推荐你在 visual_servo.py 里采用的姿态控制律是：")
        print("  tilt_err_base = np.cross(normal_base, np.array([0, 0, 1]))")
        print("  w_tilt_base   = k * tilt_err_base")
        print("如果单轴测试显示中心差分接近 [-1, 0, 0] / [0, -1, 0]，")
        print("说明这个控制律的方向已经是对的，不需要再人工猜 wx/wy 对应关系。")
        print("=" * 60)


if __name__ == '__main__':
    try:
        tester = PostureVisionMappingTest()

        mode = input("请选择模式 (A: 纯观测模式, B: 机械臂单轴测试模式): ").strip().upper()
        if mode == 'A':
            tester.run_observe_mode()
        elif mode == 'B':
            tester.run_robot_test()
        else:
            print("输入无效，退出。")

    except rospy.ROSInterruptException:
        pass
