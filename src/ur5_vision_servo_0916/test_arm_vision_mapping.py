#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机械臂基坐标轴与图像误差映射验证测试脚本
验证机械臂沿基坐标系 X、Y 运动时，图像误差 ex 和 ey 的变化规律。
"""

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import os
import json
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from UR_Robot import UR_Robot

class ArmVisionMappingTest:
    def __init__(self):
        rospy.init_node('test_arm_vision_mapping', anonymous=True)
        self.bridge = CvBridge()
        
        # 兼容不同版本的 OpenCV ArUco
        if hasattr(aruco, "getPredefinedDictionary"):
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        else:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            
        if hasattr(aruco, "DetectorParameters_create"):
            self.parameters = aruco.DetectorParameters_create()
        else:
            self.parameters = aruco.DetectorParameters()

        # 存储最新的误差，以备每次读取
        self.latest_ex = None
        self.latest_ey = None
        
        # 订阅 Realsense 图像
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.loginfo("Subscribed to camera, waiting for images...")
        time.sleep(2)  # 等待图像话题接收稳定

        # 初始化机械臂通讯
        # 我们使用 use_gripper=False 因为这个测试不需要控制夹爪
        # 使用 is_use_camera=False 因为我们通过 ROS 节点订阅图像，不需要 UR_Robot 内部初始化 RealSense
        self.robot = UR_Robot(tcp_host_ip="192.168.0.1", is_use_robotiq85=False, is_use_camera=False)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 同样应用旋转180度的修正
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except Exception as e:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        height, width = cv_image.shape[:2]
        u0, v0 = width / 2.0, height / 2.0

        # 画图像中心
        img_center = (int(u0), int(v0))
        cv2.drawMarker(cv_image, img_center, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        if ids is not None and len(corners) > 0:
            c = corners[0][0]
            u = np.mean(c[:, 0])
            v = np.mean(c[:, 1])
            self.latest_ex = u - u0
            self.latest_ey = v - v0

            # 画 ArUco 中心和误差线，方便观察
            marker_center = (int(u), int(v))
            cv2.circle(cv_image, marker_center, 5, (0, 0, 255), -1)
            cv2.arrowedLine(cv_image, img_center, marker_center, (255, 0, 0), 2, tipLength=0.1)
            cv2.putText(cv_image, f"ex:{self.latest_ex:.1f} ey:{self.latest_ey:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            aruco.drawDetectedMarkers(cv_image, corners, ids)
        else:
            # 没检测到时置空，以防误读旧数据
            self.latest_ex = None
            self.latest_ey = None
            cv2.putText(cv_image, "No ArUco detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Arm Vision Mapping Test", cv_image)
        cv2.waitKey(1)

    def move_and_measure(self, name, target_pose, k_acc, k_vel):
        """
        封装的移动和测量函数
        :param name: 当前移动的描述
        :param target_pose: 目标位姿 [x, y, z, roll, pitch, yaw]
        :param k_acc: 加速度系数
        :param k_vel: 速度系数
        """
        print(f"\n[{name}] 正在移动...")
        # 改为使用 move_j_p_1 (非阻塞发送)，避免 recv() 循环超时死锁
        self.robot.move_j_p_1(target_pose, k_acc=k_acc, k_vel=k_vel)
        
        # 稳定等待
        print(f"[{name}] 指令已发送，等待 3 秒以确保机械臂运动完毕并获取最新图像...")
        time.sleep(3.0)
        
        ex, ey = self.latest_ex, self.latest_ey
        if ex is not None and ey is not None:
            print(f"[{name}] 视觉误差: ex = {ex:.2f}, ey = {ey:.2f}")
            return ex, ey
        else:
            print(f"[{name}] 警告: 未检测到 ArUco 码！")
            return None, None

    def run_test(self):
        print("\n" + "="*50)
        print("启动手眼坐标系真实映射测试 (Camera Frame Kinematics)")
        print("="*50)
        
        # 读取手眼标定矩阵（这是将相机动作转为真实机械臂基坐标动作的核心）
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_result1_4.json')
        try:
            with open(json_path, 'r') as f:
                calib_data = json.load(f)
                R_cam2tool = np.array(calib_data["hand_eye_matrix"])[:3, :3]
                print(f"成功读取手眼标定矩阵: \n{np.array2string(R_cam2tool, formatter={'float_kind':lambda x: '%.3f' % x})}")
        except Exception as e:
            print(f"读取标定矩阵失败: {e}")
            return
        
        # 预抓取位姿的旋转向量形式 (rx, ry, rz)
        x_base, y_base, z_base = -0.478, -0.0678, 0.336
        rx, ry, rz = 2.222, -2.22, -0.140
        
        # 核心：计算 相机到基坐标系 (Base_link) 的真实旋转映射矩阵
        R_tool2base, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=float))
        R_cam2base = R_tool2base @ R_cam2tool
        
        # 需要将其转换成 rpy 形式以给 move_j_p 使用
        try:
            rpy_tuple = self.robot.rotvec_to_rpy(x_base, y_base, z_base, rx, ry, rz)
            base_pose = list(rpy_tuple)
        except Exception as e:
            rospy.logerr(f"位姿转换失败: {e}")
            return
            
        print(f"转换后的基准位姿 (x,y,z,r,p,y): {[round(x,3) for x in base_pose]}")
        input("按下 Enter 键开始移动到基准位置（请确保安全，机械臂将真实移动!）...")

        def get_cam_offset_pose(dx_c, dy_c, dz_c):
            """将相机想要移动的独立距离，映射为基坐标系该移动的距离"""
            delta_base = R_cam2base @ np.array([dx_c, dy_c, dz_c])
            pose = base_pose.copy()
            pose[0] += delta_base[0]
            pose[1] += delta_base[1]
            pose[2] += delta_base[2]
            return pose

        # 慢速，低加速度
        k_acc = 0.2
        k_vel = 0.2
        step = 0.01  # 每次移动 10mm 测试，以放大画面上的像素变化

        # 1. 移动到起始点
        ex_b, ey_b = self.move_and_measure("0. 回到起始基准点", base_pose, k_acc, k_vel)
        
        # --- Camera 完全独立的运动测试 ---
        ex_cx, ey_cx = self.move_and_measure("1. 相机沿其自身的 +X 轴右移 (10mm)", get_cam_offset_pose(step, 0, 0), k_acc, k_vel)
        self.move_and_measure("Return", base_pose, k_acc, k_vel)
        
        ex_cy, ey_cy = self.move_and_measure("2. 相机沿其自身的 +Y 轴下移 (10mm)", get_cam_offset_pose(0, step, 0), k_acc, k_vel)
        self.move_and_measure("Return", base_pose, k_acc, k_vel)
        
        ex_cz, ey_cz = self.move_and_measure("3. 相机沿其自身的 +Z 轴前进 (10mm)", get_cam_offset_pose(0, 0, step), k_acc, k_vel)
        self.move_and_measure("Return", base_pose, k_acc, k_vel)

        # 总结输出
        print("\n\n" + "="*50)
        print("====== 手眼坐标系解耦测试：纯运动视觉映射结果 ======")
        print("="*50)
        if ex_b is not None and ey_b is not None:
            print(f"【基准画面误差】: ex = {ex_b:.2f} px, ey = {ey_b:.2f} px\n")
            
            if ex_cx is not None:
                dx = ex_cx - ex_b
                dy = ey_cx - ey_b
                print(f"[相机 +X 轴运动] 引起的画面像素变化:")
                print(f"   -> ex (U坐标) 变化了 {dx:+.2f} px")
                print(f"   -> ey (V坐标) 变化了 {dy:+.2f} px")
                if abs(dx) > abs(dy):
                    trend = "ex 增大 (+)" if dx > 0 else "ex 减小 (-)"
                    print(f"   结论: X 轴确实主导了 U坐标，当相机向+X走时，{trend}。")
                else:
                    print("   ！！！警告: Camera X 运动反而导致 ey 变化更大！平移映射错乱或镜头装歪了！")
            
            print("-" * 40)
            if ex_cy is not None:
                dx = ex_cy - ex_b
                dy = ey_cy - ey_b
                print(f"[相机 +Y 轴运动] 引起的画面像素变化:")
                print(f"   -> ex (U坐标) 变化了 {dx:+.2f} px")
                print(f"   -> ey (V坐标) 变化了 {dy:+.2f} px")
                if abs(dy) > abs(dx):
                    trend = "ey 增大 (+)" if dy > 0 else "ey 减小 (-)"
                    print(f"   结论: Y 轴确实主导了 V坐标，当相机向+Y走时，{trend}。")
                else:
                    print("   ！！！警告: Camera Y 运动反而导致 ex 变化更大！平移映射错乱或镜头装歪了！")
        else:
            print("未能在基准位置获取到有效的视觉误差，无法生成对比总结。")
        print("="*50)
        print("【关于控制律中正负号的定论】")
        print("如果在上面看到: 相机向 +X 运动 -> ex 变化量为正 (+)：")
        print("说明：如果要减小正的 ex (目标在视野右侧)，我们必须向 -X 去修正。此时伺服 v_x_cam = -k * ex (需要加负号)。")
        print("--------------------------------------------------")
        print("如果在上面看到: 相机向 +X 运动 -> ex 变化量为负 (-)：")
        print("说明：如果要减小正的 ex，我们必须向 +X 去修正。此时伺服 v_x_cam = k * ex (不要加负号)。")
        print("==================================================")


if __name__ == '__main__':
    try:
        tester = ArmVisionMappingTest()
        tester.run_test()
    except rospy.ROSInterruptException:
        pass
