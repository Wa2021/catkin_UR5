#!/usr/bin/env python3
"""
IBVS 视觉伺服节点：使用 ArUco 模拟 YOLO-OBB 检测输入，
包含坐标映射、死区和误差滤波。
"""
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import os
import json
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from collections import deque
from tf.transformations import quaternion_matrix


def normalize_angle_to_half_pi(angle):
    """
    将角度归一化到 [-π/2, π/2) 范围
    
    原因：
    - ArUco 的 atan2 输出范围是 (-π, π]
    - YOLO-OBB 的角度通常在 [-π/2, π/2) 之间
    - 抓取长方形时，转 180° 和 0° 等效（抓反了也可以）
    
    这样做可以保证 ArUco 和 YOLO 的控制律一致
    """
    while angle > math.pi / 2:
        angle -= math.pi
    while angle <= -math.pi / 2:
        angle += math.pi
    return angle


def get_normal_from_rvec(rvec):
    """
    从旋转向量提取物体的平面法向量 (Z轴方向)
    这是完美模拟后续 "RANSAC 平面拟合" 给出的法线 nx, ny, nz
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    normal_in_cam = rotation_matrix[:, 2]
    
    # 统一法线方向朝向相机 (Z值为正)
    if normal_in_cam[2] < 0:
        normal_in_cam = -normal_in_cam
        
    return normal_in_cam


class ErrorFilter:
    """
    误差滤波器：使用滑动窗口平均减少检测噪声
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffers = {}  # 每个通道一个缓冲区
    
    def filter(self, name, value):
        """对指定通道的值进行滤波"""
        if name not in self.buffers:
            self.buffers[name] = deque(maxlen=self.window_size)
        
        self.buffers[name].append(value)
        return np.mean(self.buffers[name])
    
    def reset(self):
        """重置所有缓冲区"""
        self.buffers.clear()


def apply_deadzone(value, deadzone):
    """
    应用死区：误差小于阈值时返回0
    改进：使用平滑过渡死区，避免误差越过死区边界时发生突变导致急加速/急刹车
    """
    if abs(value) < deadzone:
        return 0.0
    # 减去死区大小，保证离开死区时值为 0 并连续增长
    if value > 0:
        return value - deadzone
    else:
        return value + deadzone


def smoothstep01(value):
    """将 [0, 1] 区间内的值平滑映射到 [0, 1]。"""
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def read_bool_param(name, default):
    """兼容 launch/YAML/命令行中不同形式的布尔参数。"""
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class ArucoServoNode:
    def __init__(self):
        rospy.init_node('aruco_servo_node')

        self.command_topic = rospy.get_param("~command_topic", "/servo_server/delta_twist_cmds")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.command_frame = rospy.get_param("~command_frame", "base_link")
        self.rotate_image_180 = read_bool_param("~rotate_image_180", True)
        self.debug_step_mode = read_bool_param("~debug_step_mode", True)

        self.enable_linear_x = read_bool_param("~enable_linear_x", True)
        self.enable_linear_y = read_bool_param("~enable_linear_y", True)
        self.enable_linear_z = read_bool_param("~enable_linear_z", True)
        self.enable_angular_x = read_bool_param("~enable_angular_x", True)
        self.enable_angular_y = read_bool_param("~enable_angular_y", True)
        self.enable_angular_z = read_bool_param("~enable_angular_z", True)

        enabled_axes = []
        if self.enable_linear_x:
            enabled_axes.append("x")
        if self.enable_linear_y:
            enabled_axes.append("y")
        if self.enable_linear_z:
            enabled_axes.append("z")
        if self.enable_angular_x:
            enabled_axes.append("rx")
        if self.enable_angular_y:
            enabled_axes.append("ry")
        if self.enable_angular_z:
            enabled_axes.append("rz")
        self.enabled_axes_text = ", ".join(enabled_axes) if enabled_axes else "none"
        self.window_name = "Servo View (XY)" if enabled_axes == ["x", "y"] else "Servo View"
        
        # 1. 发布速度指令给 MoveIt Servo
        self.vel_pub = rospy.Publisher(self.command_topic, TwistStamped, queue_size=1)
        
        # 2. 订阅相机图像
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        
        # 3. TF 监听器 (用于获取真实 Z 高度)
        self.tf_listener = tf.TransformListener()
        
        # =====================================================
        # 4. 伺服目标参数
        # =====================================================
        self.target_u = 640 / 2      # 图像中心 X
        self.target_v = 480 / 2      # 图像中心 Y
        self.target_z_height = 0.30  # 期望 TCP 高度 (米) - 替代面积控制
        self.target_angle = 0        # 期望角度
        
        # =====================================================
        # 5. 控制增益
        # =====================================================
        self.lambda_tilt = 0.2       # 倾角控制增益
        self.lambda_z = 0.2          # Z 轴高度控制增益
        self.kx = 0.0003             # X 轴平移增益
        self.ky = 0.0003             # Y 轴平移增益
        self.lambda_yaw = 0.2        # Yaw 控制增益
        
        self.max_linear = 0.02       # 线速度极限制降至 2cm/s
        self.max_angular = 0.03      # 角速度极限制降至 0.03rad/s
        self.deadzone_px = 10.0      # 像素死区
        self.deadzone_z = 0.01       # 高度死区：1cm
        self.deadzone_angle = 0.05   # 角度死区：0.05 rad (约3度)
        self.deadzone_tilt = 0.05    # 法向量倾角死区
        
        # =====================================================
        # 5.2 误差滤波器
        # =====================================================
        self.error_filter = ErrorFilter(window_size=8)

        # =====================================================
        # 5.3 速度指令低通滤波器 (加速度限制机制)
        # =====================================================
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.current_wx = 0.0
        self.current_wy = 0.0
        self.current_wz = 0.0
        self.alpha_v = 0.2     # 目标速度低通滤波系数
        self.stop_decay = 0.95 # 空格松开后的平滑减速系数
        self.debug_alpha_v = 0.18
        self.debug_stop_decay = 0.98

        # =====================================================
        # 5. 读取手眼标定数据
        # =====================================================
        default_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_result1_4.json')
        json_path = rospy.get_param("~hand_eye_calibration_json", default_json_path)
        try:
            with open(json_path, 'r') as f:
                calib_data = json.load(f)
                self.camera_matrix = np.array(calib_data["camera_matrix"], dtype=float)
                self.dist_coeffs = np.array(calib_data["dist_coeffs"], dtype=float)
                # 提取手眼标定矩阵的旋转部分 (R_cam2tool)
                self.R_cam2tool = np.array(calib_data["hand_eye_matrix"])[:3, :3]
                rospy.loginfo("成功读取手眼标定矩阵: %s", json_path)
        except Exception as e:
            rospy.logwarn("未能读取手眼标定文件 %s: %s", json_path, e)
            rospy.logwarn("将退回到单位阵 R_cam2tool=I；这适合链路联调，不适合认真验证坐标映射。")
            self.camera_matrix = np.eye(3)
            self.dist_coeffs = np.zeros(5)
            self.R_cam2tool = np.eye(3)
            
        # ArUco 码物理尺寸 (米)，用于姿态估计
        self.aruco_marker_size = 0.05  # 5cm

        # =====================================================
        # 6. 调试模式开关
        # =====================================================
        self.waiting_for_input = False
        self.debug_hold_duration = 0.15
        self.debug_ramp_up_duration = 0.04
        self.debug_ramp_down_duration = 0.12
        self.debug_max_linear = 0.006
        self.debug_max_angular = 0.012
        self.debug_pulse_start = rospy.Time(0)
        self.active_until = rospy.Time(0)
        self.last_status_print = rospy.Time(0)
        self.last_cmd_print = rospy.Time(0)

        rospy.loginfo(
            "ArUco Visual Servo Node Started. topic=%s image=%s frame=%s axes=%s",
            self.command_topic,
            self.image_topic,
            self.command_frame,
            self.enabled_axes_text,
        )
        if self.debug_step_mode:
            rospy.logwarn(">>> 单步调试模式已开启！请在弹出的图像窗口按空格键执行每一步移动 <<<")
        
        # 初始化误差记录
        self.last_errors = {'ex': 0.0, 'ey': 0.0}

    def get_debug_pulse_scale(self, now):
        """为空格触发的单步调试生成平滑速度包络。"""
        if now >= self.active_until:
            return 0.0

        elapsed = (now - self.debug_pulse_start).to_sec()
        remaining = (self.active_until - now).to_sec()

        ramp_up = 1.0
        ramp_down = 1.0
        if self.debug_ramp_up_duration > 1e-6:
            ramp_up = smoothstep01(elapsed / self.debug_ramp_up_duration)
        if self.debug_ramp_down_duration > 1e-6:
            ramp_down = smoothstep01(remaining / self.debug_ramp_down_duration)

        return min(ramp_up, ramp_down)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.rotate_image_180:
                # 修正相机安装旋转180度的问题
                cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except Exception as e:
            rospy.logerr(e)
            return

        vel_msg = TwistStamped()
        vel_msg.header.stamp = rospy.Time.now()
        # 你的标定实验是在基坐标系(base_link)下做的动作！为了映射完全成立，速度指令也必须放在 base_link！
        vel_msg.header.frame_id = self.command_frame

        # =====================================================
        # 1. 感知层：ArUco 检测 → 伪装成 YOLO + 平面拟合
        # =====================================================
        features = self.get_aruco_features(cv_image)

        if features is None:
            # 没看到码，让速度逐渐降低为0而不是瞬间急刹车
            decay = self.debug_stop_decay if self.debug_step_mode else self.stop_decay
            self.current_vx *= decay
            self.current_vy *= decay
            self.current_vz *= decay
            self.current_wx *= decay
            self.current_wy *= decay
            self.current_wz *= decay
            
            vel_msg.twist.linear.x = self.current_vx
            vel_msg.twist.linear.y = self.current_vy
            vel_msg.twist.linear.z = self.current_vz
            vel_msg.twist.angular.x = self.current_wx
            vel_msg.twist.angular.y = self.current_wy
            vel_msg.twist.angular.z = self.current_wz

            self._enforce_axis_mask()
            vel_msg.twist.linear.x = self.current_vx
            vel_msg.twist.linear.y = self.current_vy
            vel_msg.twist.linear.z = self.current_vz
            vel_msg.twist.angular.x = self.current_wx
            vel_msg.twist.angular.y = self.current_wy
            vel_msg.twist.angular.z = self.current_wz
            
            self.vel_pub.publish(vel_msg)
            self.error_filter.reset()  # 重置滤波器，避免下次检测到时使用旧数据
            now = rospy.Time.now()
            if not self.debug_step_mode and (now - self.last_cmd_print).to_sec() > 1.0:
                rospy.logwarn("未检测到 ArUco，速度平滑递减中...")
                self.last_cmd_print = now
            if self.debug_step_mode:
                cv2.imshow(self.window_name, cv_image)
                cv2.waitKey(1)
            return

        # 解包视觉特征
        u, v, area, angle, normal_in_cam = features

        # =====================================================
        # 2. 真实坐标系映射
        # =====================================================
        try:
            (trans_tf, rot_tf) = self.tf_listener.lookupTransform('/base_link', '/tool0', rospy.Time(0))
            R_tool2base = quaternion_matrix(rot_tf)[:3, :3]
        except Exception as e:
            rospy.logwarn_throttle(2.0, "【离线模式】未读取到机械臂TF，使用 pre-grasp 基准位姿模拟 base_link -> tool0")
            rvec_home = np.array([2.222, -2.22, -0.140], dtype=float)
            R_tool2base, _ = cv2.Rodrigues(rvec_home)
            
        R_cam2tool = getattr(self, 'R_cam2tool', np.eye(3))
        R_cam2base = R_tool2base @ R_cam2tool

        # =====================================================
        # 3. 控制层：计算误差并映射到 base_link 坐标系
        # =====================================================
        
        ex = u - self.target_u
        ey = v - self.target_v
        
        # 滤波全方位参数
        ex_filtered = self.error_filter.filter('ex', ex)
        ey_filtered = self.error_filter.filter('ey', ey)
        error_angle_filtered = self.error_filter.filter('yaw', self.target_angle - angle)
        
        # 应用死区
        ex_deadzone = apply_deadzone(ex_filtered, self.deadzone_px)
        ey_deadzone = apply_deadzone(ey_filtered, self.deadzone_px)
        error_angle_deadzone = apply_deadzone(error_angle_filtered, self.deadzone_angle)
        
        # Z 轴高度控制 (由于需要精确的 Z 轴控制，我们先独立在 base 里面求误差)
        current_z = None
        if self.enable_linear_z:
            try:
                (trans, _) = self.tf_listener.lookupTransform('/base_link', '/ee_link', rospy.Time(0))
                current_z = trans[2]
                error_z_height = self.target_z_height - current_z
                error_z_filtered = self.error_filter.filter('z', error_z_height)
                error_z_deadzone = apply_deadzone(error_z_filtered, self.deadzone_z)
            except Exception:
                rospy.logwarn_throttle(1, "无法获取 TF 数据，Z 轴保持静止")
                error_z_filtered = 0.0
                error_z_deadzone = 0.0
        else:
            error_z_filtered = 0.0
            error_z_deadzone = 0.0

        # 平移控制先在相机系构造速度，再映射到 base_link
        v_virt_cam = np.array([
            ex_deadzone * self.kx,
            ey_deadzone * self.ky,
            -error_z_deadzone * self.lambda_z
        ])
        v_base = R_cam2base @ v_virt_cam
        vx = v_base[0]
        vy = v_base[1]
        vz = v_base[2]

        err_base_rot_x = 0.0
        err_base_rot_y = 0.0
        vroll = 0.0
        vpitch = 0.0
        vyaw = 0.0

        if self.enable_angular_x or self.enable_angular_y or self.enable_angular_z:
            # 图像经过 180 度旋转后，法向量的 X/Y 需要翻回真实相机坐标系
            corrected_normal_in_cam = np.array([-normal_in_cam[0], -normal_in_cam[1], normal_in_cam[2]])

            normal_base = R_cam2base @ corrected_normal_in_cam
            target_normal_base = np.array([0.0, 0.0, 1.0])
            w_tilt_base = np.cross(target_normal_base, normal_base)

            err_base_rot_x = self.error_filter.filter('rot_x', w_tilt_base[0])
            err_base_rot_y = self.error_filter.filter('rot_y', w_tilt_base[1])

            err_base_rot_x_deadzone = apply_deadzone(err_base_rot_x, self.deadzone_tilt)
            err_base_rot_y_deadzone = apply_deadzone(err_base_rot_y, self.deadzone_tilt)

            vroll = -err_base_rot_x_deadzone * self.lambda_tilt
            vpitch = -err_base_rot_y_deadzone * self.lambda_tilt

            # Yaw 误差先在相机系 Z 轴构造，再映射到 base_link
            w_yaw_cam = np.array([0.0, 0.0, -error_angle_deadzone * self.lambda_yaw])
            w_yaw_base = R_cam2base @ w_yaw_cam

            # 汇总完整解耦角速度
            vroll += w_yaw_base[0]
            vpitch += w_yaw_base[1]
            vyaw = w_yaw_base[2]
        
        # 目标速度限幅计算
        linear_limit = self.max_linear
        angular_limit = self.max_angular
        if self.debug_step_mode:
            linear_limit = min(linear_limit, self.debug_max_linear)
            angular_limit = min(angular_limit, self.debug_max_angular)

        target_vx = np.clip(vx, -linear_limit, linear_limit)
        target_vy = np.clip(vy, -linear_limit, linear_limit)
        target_vz = np.clip(vz, -linear_limit, linear_limit)
        
        target_vroll = np.clip(vroll, -angular_limit, angular_limit)
        target_vpitch = np.clip(vpitch, -angular_limit, angular_limit)
        target_vyaw = np.clip(vyaw, -angular_limit, angular_limit)

        if not self.enable_linear_x:
            target_vx = 0.0
        if not self.enable_linear_y:
            target_vy = 0.0
        if not self.enable_linear_z:
            target_vz = 0.0
        if not self.enable_angular_x:
            target_vroll = 0.0
        if not self.enable_angular_y:
            target_vpitch = 0.0
        if not self.enable_angular_z:
            target_vyaw = 0.0
        
        # 低通滤波形成平滑启停
        is_active = True
        pulse_scale = 1.0
        now = rospy.Time.now()
        if self.debug_step_mode:
            pulse_scale = self.get_debug_pulse_scale(now)
            is_active = pulse_scale > 0.0
            target_vx *= pulse_scale
            target_vy *= pulse_scale
            target_vz *= pulse_scale
            target_vroll *= pulse_scale
            target_vpitch *= pulse_scale
            target_vyaw *= pulse_scale

        if not is_active:
            target_vx, target_vy, target_vz = 0.0, 0.0, 0.0
            target_vroll, target_vpitch, target_vyaw = 0.0, 0.0, 0.0
            decay = self.debug_stop_decay if self.debug_step_mode else self.stop_decay
            a = 1.0 - decay
        else:
            a = self.debug_alpha_v if self.debug_step_mode else self.alpha_v

        self.current_vx = a * target_vx + (1 - a) * self.current_vx
        self.current_vy = a * target_vy + (1 - a) * self.current_vy
        self.current_vz = a * target_vz + (1 - a) * self.current_vz
        
        self.current_wx = a * target_vroll + (1 - a) * self.current_wx
        self.current_wy = a * target_vpitch + (1 - a) * self.current_wy
        self.current_wz = a * target_vyaw + (1 - a) * self.current_wz

        self._enforce_axis_mask()
        
        vel_msg.twist.linear.x = self.current_vx
        vel_msg.twist.linear.y = self.current_vy
        vel_msg.twist.linear.z = self.current_vz
        
        vel_msg.twist.angular.x = self.current_wx
        vel_msg.twist.angular.y = self.current_wy
        vel_msg.twist.angular.z = self.current_wz
        
        # Z轴末端保护 (防撞地)
        try:
            if current_z is not None and current_z < 0.10: # 高度一旦跌破 10cm 硬限位
                if vel_msg.twist.linear.z < 0: # 如果速度方向还是朝下
                    vel_msg.twist.linear.z = 0.0
                    self.current_vz = 0.0  # 重置当前缓降速度
                    rospy.logwarn_throttle(0.5, f"接近地面(Z={current_z:.3f}m)，强制关闭下降速度！")
        except:
            pass

        # 保存并打印
        self.last_errors = {'ex': ex_filtered, 'ey': ey_filtered, 'z': error_z_filtered, 'yaw': error_angle_filtered, 'tx': err_base_rot_x, 'ty': err_base_rot_y}
        
        rospy.loginfo_throttle(0.5, f"【视觉误差角速度】RotX_base_err={err_base_rot_x:5.2f} | RotY_base_err={err_base_rot_y:5.2f} || "
                                    f"【机械臂补偿指令】RotX={vel_msg.twist.angular.x:6.3f} | RotY={vel_msg.twist.angular.y:6.3f} | Yaw={vel_msg.twist.angular.z:6.3f}")

        # =====================================================
        # 5. 可视化死区圈（用以直观判断为何没触发 ALIGNED）
        # =====================================================
        # 绘制收敛容忍度圆圈 (绿色实线表示只要红点进去了就算 ALIGNED)
        cv2.circle(cv_image, (int(self.target_u), int(self.target_v)), 
                   int(self.deadzone_px), (0, 100, 0), 2)
                   
        # 绘制目标中心十字
        cv2.line(cv_image, (int(self.target_u)-15, int(self.target_v)), 
                 (int(self.target_u)+15, int(self.target_v)), (0, 255, 0), 2)
        cv2.line(cv_image, (int(self.target_u), int(self.target_v)-15), 
                 (int(self.target_u), int(self.target_v)+15), (0, 255, 0), 2)
        
        # 绘制当前检测点和连线
        cv2.circle(cv_image, (int(u), int(v)), 5, (0, 0, 255), -1)
        cv2.line(cv_image, (int(self.target_u), int(self.target_v)), 
                 (int(u), int(v)), (255, 0, 255), 2)
        
        # 显示误差信息
        y_offset = 30
        cv2.putText(cv_image, f"Error XY: ({self.last_errors['ex']:.1f}, {self.last_errors['ey']:.1f}) px", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(cv_image, f"Vel XY: ({vel_msg.twist.linear.x:.4f}, {vel_msg.twist.linear.y:.4f})", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
        
        # 收敛状态指示
        y_offset += 40
        converged_xy = abs(self.last_errors['ex']) < self.deadzone_px and abs(self.last_errors['ey']) < self.deadzone_px
        status_color = (0, 255, 0) if converged_xy else (0, 165, 255)
        status_text = "ALIGNED" if converged_xy else "TRACKING"
        cv2.putText(cv_image, f"Status: {status_text}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # =====================================================
        # 5. 单步调试逻辑
        # =====================================================
        if self.debug_step_mode:
            now = rospy.Time.now()
            if (now - self.last_status_print).to_sec() > 0.5:
                state = "ACTIVE" if is_active else "IDLE"
                print(f"[ {state} ] pulse={pulse_scale:.2f} vx={vel_msg.twist.linear.x:.4f}, vy={vel_msg.twist.linear.y:.4f}, wz={vel_msg.twist.angular.z:.4f}")
                self.last_status_print = now

            cv2.putText(cv_image, "DEBUG STEP: Press 'Space' for one short pulse", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(cv_image, f"Pulse={self.debug_hold_duration:.2f}s Lin<{self.debug_max_linear:.3f} Ang<{self.debug_max_angular:.3f}",
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            cv2.putText(cv_image, f"Axes: {self.enabled_axes_text}", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(self.window_name, cv_image)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # Space key
                self.debug_pulse_start = rospy.Time.now()
                self.active_until = rospy.Time.now() + rospy.Duration(self.debug_hold_duration)
                print(">>> 单步脉冲触发：持续 {:.2f}s，线速度上限 {:.3f} m/s，角速度上限 {:.3f} rad/s".format(
                    self.debug_hold_duration, self.debug_max_linear, self.debug_max_angular
                ))

            self.vel_pub.publish(vel_msg)
        else:
            self.vel_pub.publish(vel_msg)
            cv2.putText(cv_image, f"Axes: {self.enabled_axes_text}", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(self.window_name, cv_image)
            cv2.waitKey(1)

    def _enforce_axis_mask(self):
        """禁用的轴始终发布为 0，避免旧状态残留。"""
        if not self.enable_linear_x:
            self.current_vx = 0.0
        if not self.enable_linear_y:
            self.current_vy = 0.0
        if not self.enable_linear_z:
            self.current_vz = 0.0
        if not self.enable_angular_x:
            self.current_wx = 0.0
        if not self.enable_angular_y:
            self.current_wy = 0.0
        if not self.enable_angular_z:
            self.current_wz = 0.0

    def get_aruco_features(self, img):
        """
        核心函数：把 ArUco 数据转换成 YOLO-OBB + 平面拟合 格式
        
        返回 5 个特征值，对应当前控制器使用的视觉量：
        - u, v: 目标中心（像素）
        - area: 目标面积（像素²）
        - angle: 目标平面内角度（弧度）
        - normal_in_cam: 目标平面法向量（相机系）
        
        
        切换到 YOLO / 平面拟合时的对应关系：
        - u, v     ← result.obb.xywhr[0:2]
        - area     ← result.obb.xywhr[2] * result.obb.xywhr[3]
        - angle    ← result.obb.xywhr[4] (已归一化)
        - normal_in_cam ← 平面拟合法向量
        """
        # ArUco 检测使用灰度图即可。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OpenCV 4.7+ 去掉 Dictionary_get，使用兼容写法
        if hasattr(aruco, "getPredefinedDictionary"):
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        else:
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

        # OpenCV 4.10+ 去掉 DetectorParameters_create，改用构造函数
        if hasattr(aruco, "DetectorParameters_create"):
            parameters = aruco.DetectorParameters_create()
        else:
            parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # corners 的结构是 [N][1][4][2]，这里只取第一个码
        if ids is not None:
            c = corners[0][0]
            
            u = np.mean(c[:, 0])
            v = np.mean(c[:, 1])
            
            area = cv2.contourArea(c)
            
            dx = c[1][0] - c[0][0]
            dy = c[1][1] - c[0][1]
            raw_angle = math.atan2(dy, dx)
            
            angle = normalize_angle_to_half_pi(raw_angle)
            
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, self.aruco_marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            normal_in_cam = get_normal_from_rvec(rvec[0])
            
            cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs, 
                             rvec[0], tvec[0], self.aruco_marker_size * 1.5, 3)
            
            return u, v, area, angle, normal_in_cam
            
        return None


if __name__ == '__main__':
    try:
        node = ArucoServoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
