#coding=utf8
import time
import copy
import socket
import struct
import numpy as np
import math
import os
from real.robotiq_gripper import RobotiqGripper
import cv2

class UR_Robot:
    def __init__(self, tcp_host_ip="192.168.0.1", tcp_port=30003, workspace_limits=None,
                 is_use_robotiq85=True, is_use_camera=True):
        # Basic connection and device options
        if workspace_limits is None:
            workspace_limits = [[-0.7, 0.7], [-0.7, 0.7], [0.00, 0.6]]
        self.workspace_limits = workspace_limits
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port
        self.is_use_robotiq85 = is_use_robotiq85
        self.is_use_camera = is_use_camera


        # UR5 robot configuration
        # Default joint/tool speed configuration
        self.joint_acc = 1.4  # Safe: 1.4   8
        self.joint_vel = 1.05  # Safe: 1.05  3

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 0.5  # Safe: 0.5
        self.tool_vel = 0.2  # Safe: 0.2

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # robotiq85 gripper configuration
        if(self.is_use_robotiq85):
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()#复位
            print("Activating gripper...")
            self.gripper.activate()
            time.sleep(1.5)
        
        # realsense configuration
        if(self.is_use_camera):
            from real.realsenseD435_new1 import Camera
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera()
            #self.cam_intrinsics = self.camera.intrinsics  # get camera intrinsics
        self.cam_intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3)
        # # Load camera pose (from running calibrate.py), intrinsics and depth scale
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cam_pose_path = os.path.join(script_dir, 'real/cam_pose/camera_pose.txt')
        cam_depth_path = os.path.join(script_dir, 'real/cam_pose/camera_depth_scale.txt')
        self.cam_pose = np.loadtxt(cam_pose_path, delimiter=' ')
        self.cam_depth_scale = np.loadtxt(cam_depth_path, delimiter=' ')


        # Default robot home joint configuration (the robot is up to air)
        self.home_joint_config = [-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             -(0 / 360.0) * 2 * np.pi, 0.0]

        # self.testRobot()

    def testRobot(self):
        """Manual robot motion smoke test. Keep calls commented unless testing on hardware."""
        try:
            print("Test for robot...")
            # self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  -(0 / 360.0) * 2 * np.pi, 0.0])
            # self.move_j([(57.04 / 360.0) * 2 * np.pi, (-65.26/ 360.0) * 2 * np.pi,
            #                  (73.52/ 360.0) * 2 * np.pi, (-100.89/ 360.0) * 2 * np.pi,
            #                  (-86.93/ 360.0) * 2 * np.pi, (-0.29/360)*2*np.pi])
            # self.open_gripper()
            # self.move_j([(57.03 / 360.0) * 2 * np.pi, (-56.67 / 360.0) * 2 * np.pi,
            #                   (88.72 / 360.0) * 2 * np.pi, (-124.68 / 360.0) * 2 * np.pi,
            #                   (-86.96/ 360.0) * 2 * np.pi, (-0.3/ 360) * 2 * np.pi])
            # self.close_gripper()
            # self.move_j([(57.04 / 360.0) * 2 * np.pi, (-65.26 / 360.0) * 2 * np.pi,
            #                   (73.52 / 360.0) * 2 * np.pi, (-100.89 / 360.0) * 2 * np.pi,
            #                   (-86.93 / 360.0) * 2 * np.pi, (-0.29 / 360) * 2 * np.pi])
            # self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  -(0 / 360.0) * 2 * np.pi, 0.0])
            # self.move_j_p([0.3,0,0.3,np.pi/2,0,0],0.5,0.5)
            # for i in range(10):
            #     self.move_j_p([0.3, 0, 0.3, np.pi, 0, i*0.1], 0.5, 0.5)
            #     time.sleep(1)
            # self.move_j_p([0.3, 0, 0.3, -np.pi, 0, 0],0.5,0.5)
            # self.move_p([0.3, 0.3, 0.3, -np.pi, 0, 0],0.5,0.5)
            # self.move_l([0.2, 0.2, 0.3, -np.pi, 0, 0],0.5,0.5)
            # self.plane_grasp([0.3, 0.3, 0.1])
            # self.plane_push([0.3, 0.3, 0.1])
        except:
            print("Test fail! ")
    
    def move_j(self, joint_configuration,k_acc=1,k_vel=1,t=0,r=0):
        """Blocking joint-space move. joint_configuration contains six joint angles in radians."""
        joint_configuration = np.asarray(joint_configuration, dtype=np.float64).reshape(-1)
        if joint_configuration.size != 6:
            raise ValueError("joint_configuration must contain 6 joint angles")

        tcp_command = "movej([%f" % joint_configuration[0]  #"movej([]),a=,v=,\n"
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f,t=%f,r=%f)\n" % (k_acc*self.joint_acc, k_vel*self.joint_vel,t,r)

        with self._connect_robot_socket() as tcp_socket:
            self.tcp_socket = tcp_socket
            tcp_socket.sendall(str.encode(tcp_command))#把拼好的字符串编码成字节流，发送给机械臂。

            # Block until the target joint configuration is reached.
            state_data = tcp_socket.recv(1500)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
                state_data = tcp_socket.recv(1500)
                actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
                time.sleep(0.01)

    def _to_float_pose(self, pose):
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.size != 6:
            raise ValueError("TCP pose must contain 6 values: [x, y, z, rx, ry, rz] or [x, y, z, r, p, y]")
        return pose

    def _pose_to_rotvec_pose(self, tool_configuration, pose_format='rpy'):
        """
        Convert a TCP target to UR native pose [x, y, z, rx, ry, rz].

        pose_format='rpy' keeps compatibility with the original APIs.
        pose_format='rotvec'/'rxryrz' passes UR axis-angle orientation through unchanged.
        """
        pose = self._to_float_pose(tool_configuration)
        fmt = pose_format.lower()
        if fmt in ('rotvec', 'rxryrz', 'axis_angle', 'axis-angle'):
            return pose
        if fmt in ('rpy', 'xyzrpy', 'euler'):
            rotvec = self.rpy2rotating_vector(pose[3:6])
            return np.concatenate([pose[:3], rotvec])
        raise ValueError("pose_format must be 'rpy' or 'rotvec'")

    def _format_ur_pose(self, pose):
        return "p[%f,%f,%f,%f,%f,%f]" % tuple(pose)

    def _format_ur_vector(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        return "[" + ",".join("%f" % value for value in values) + "]"

    def _send_urscript(self, program, recv_size=0, timeout=2.0):
        """
        Send a small URScript program and optionally receive one response packet.

        This is useful for setup/stop/speed commands where we do not need the
        blocking pose wait loop used by move_j_p and move_l.
        """
        if not program.endswith("\n"):
            program += "\n"

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
            tcp_socket.settimeout(timeout)
            tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_socket.sendall(program.encode("utf-8"))
            if recv_size:
                return tcp_socket.recv(recv_size)
        return None

    def _connect_robot_socket(self, timeout=None):
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if timeout is not None:
            tcp_socket.settimeout(timeout)
        tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        return tcp_socket

    def _wait_for_tool_position(self, tcp_socket, target_pose, recv_size=1500, max_attempts=None):
        actual_tool_positions = None
        attempt = 0
        while actual_tool_positions is None or not all(
                [np.abs(actual_tool_positions[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]
        ):
            if max_attempts is not None and attempt >= max_attempts:
                print("[警告] 超过最大尝试次数，跳出等待循环")
                break
            state_data = tcp_socket.recv(recv_size)
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            attempt += 1
            time.sleep(0.01)
        return actual_tool_positions

    def move_j_pose_rotvec(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
        """Blocking movej to a UR native TCP pose [x, y, z, rx, ry, rz]."""
        return self.move_j_p(tool_configuration, k_acc, k_vel, t, r, pose_format='rotvec')

    def move_j_p(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0, pose_format='rpy'):
        target_pose = self._pose_to_rotvec_pose(tool_configuration, pose_format)
        print(f"movej_p({pose_format} -> rotvec: {target_pose.tolist()})")

        # 构造 URScript 命令
        tcp_command = "def process():\n"
        tcp_command += "movej(get_inverse_kin(%s),a=%f,v=%f,t=%f,r=%f)\n" % (
            self._format_ur_pose(target_pose),
            k_acc * self.joint_acc, k_vel * self.joint_vel, t, r)
        tcp_command += "end\n"

        with self._connect_robot_socket(timeout=5.0) as tcp_socket:
            self.tcp_socket = tcp_socket
            # 发送 URScript 指令
            tcp_socket.sendall(str.encode(tcp_command))

            # 等待机器人开始返回状态数据。
            time.sleep(0.2)  # 等待机器人执行命令

            try:
                state_data = tcp_socket.recv(2048)
                actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            except Exception as e:
                print(f"[错误] 初次解析TCP数据失败: {e}")
                actual_tool_positions = None

            # 限制等待次数，避免通信异常时一直阻塞。
            max_attempts = 100
            attempt = 0

            while actual_tool_positions is None or not all(
                    [np.abs(actual_tool_positions[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]
            ):
                if attempt >= max_attempts:
                    print("[警告] 超过最大尝试次数，跳出循环")
                    break
                try:
                    time.sleep(0.05)
                    state_data = tcp_socket.recv(2048)
                    actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
                except Exception as e:
                    print(f"[警告] 第 {attempt + 1} 次尝试解析失败: {e}")
                    actual_tool_positions = None
                attempt += 1

            time.sleep(1.5)  # 等待移动稳定

    def move_j_p_1_rotvec(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
        """Non-blocking movej to a UR native TCP pose [x, y, z, rx, ry, rz]."""
        return self.move_j_p_1(tool_configuration, k_acc, k_vel, t, r, pose_format='rotvec')

    def move_j_p_1(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0, pose_format='rpy'):
        """
        一个【非阻塞】的 move_j_p 函数，但每次调用都独立处理网络连接。
        适用于需要函数自包含、不希望在外部管理socket的场景。

        工作流程: 连接 -> 发送指令 -> 立即关闭并返回。
        """
        target_pose = self._pose_to_rotvec_pose(tool_configuration, pose_format)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
                tcp_socket.settimeout(2.0)
                tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

                tcp_command = "def process():\n"
                tcp_command += "    movej(get_inverse_kin(%s),a=%f,v=%f,t=%f,r=%f)\n" % (
                    self._format_ur_pose(target_pose),
                    k_acc, k_vel,
                    t, r
                )
                tcp_command += "end\n"

                tcp_socket.sendall(str.encode(tcp_command))

        except socket.timeout:
            print(f"[错误] 连接机器人 {self.tcp_host_ip} 超时。请检查网络和IP地址。")
        except Exception as e:
            print(f"[错误] 在 move_j_p 函数中发生异常: {e}")

    def movep_pose_rotvec(self, tool_configuration, k_acc=1, k_vel=1, r=0):
        """Blend-friendly movep to a UR native TCP pose [x, y, z, rx, ry, rz]."""
        target_pose = self._pose_to_rotvec_pose(tool_configuration, pose_format='rotvec')
        tcp_command = "movep(%s,a=%f,v=%f,r=%f)\n" % (
            self._format_ur_pose(target_pose), k_acc * self.tool_acc, k_vel * self.tool_vel, r)
        self._send_urscript(tcp_command)

    def move_l_pose_rotvec(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
        """Blocking movel to a UR native TCP pose [x, y, z, rx, ry, rz]."""
        return self.move_l(tool_configuration, k_acc, k_vel, t, r, pose_format='rotvec')

    def move_l(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0, pose_format='rpy'):
        target_pose = self._pose_to_rotvec_pose(tool_configuration, pose_format)
        print(f"movel({pose_format} -> rotvec: {target_pose.tolist()})")
        tcp_command = "def process():\n"
        tcp_command += "movel(%s,a=%f,v=%f,t=%f,r=%f)\n" % (
            self._format_ur_pose(target_pose),
            k_acc * self.joint_acc, k_vel * self.joint_vel,t,r)
        tcp_command += "end\n"

        with self._connect_robot_socket() as tcp_socket:
            self.tcp_socket = tcp_socket
            tcp_socket.sendall(str.encode(tcp_command))
            # Block until the target TCP position is reached.
            self._wait_for_tool_position(tcp_socket, target_pose)
            time.sleep(1.5)

    # Circular TCP motion.
    # mode 0: Unconstrained mode. Interpolate orientation from current pose to target pose (pose_to)
    #      1: Fixed mode. Keep orientation constant relative to the tangent of the circular arc (starting from current pose)
    def move_c_pose_rotvec(self, pose_via, tool_configuration, k_acc=1, k_vel=1, r=0, mode=0):
        """Circular move using UR native TCP poses [x, y, z, rx, ry, rz]."""
        return self.move_c(pose_via, tool_configuration, k_acc, k_vel, r, mode, pose_format='rotvec')

    def move_c(self,pose_via,tool_configuration,k_acc=1,k_vel=1,r=0,mode=0, pose_format='rpy'):
        via_pose = self._pose_to_rotvec_pose(pose_via, pose_format)
        target_pose = self._pose_to_rotvec_pose(tool_configuration, pose_format)

        print(f"movec({pose_format} -> rotvec: {via_pose.tolist()}, {target_pose.tolist()})")
        tcp_command = "def process():\n"
        tcp_command += " movec(%s,%s,a=%f,v=%f,r=%f,mode=%d)\n" % (
            self._format_ur_pose(via_pose),
            self._format_ur_pose(target_pose),
            k_acc * self.tool_acc, k_vel * self.tool_vel, r, mode)
        tcp_command += "end\n"

        with self._connect_robot_socket() as tcp_socket:
            self.tcp_socket = tcp_socket
            tcp_socket.sendall(str.encode(tcp_command))
            # Block until the target TCP position is reached.
            self._wait_for_tool_position(tcp_socket, target_pose)
            time.sleep(1.5)

    def translate_base(self, delta_xyz, k_acc=1, k_vel=1, wait=True):
        """
        Move TCP by delta_xyz in the robot base frame while keeping orientation.

        delta_xyz is in meters. The target is sent as a UR native rotvec pose.
        """
        delta_xyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
        current_pose = self.get_current_tcp_pose()
        target_pose = np.array(current_pose, dtype=np.float64)
        target_pose[:3] += delta_xyz

        if wait:
            self.move_l_pose_rotvec(target_pose, k_acc=k_acc, k_vel=k_vel)
        else:
            tcp_command = "movel(%s,a=%f,v=%f)\n" % (
                self._format_ur_pose(target_pose), k_acc * self.tool_acc, k_vel * self.tool_vel)
            self._send_urscript(tcp_command)
        return target_pose

    def translate_tool(self, delta_xyz_tool, k_acc=1, k_vel=1, wait=True):
        """
        Move TCP by delta_xyz_tool expressed in the current tool frame.

        This is handy for small camera/tool-relative nudges, e.g. move forward
        along the camera/tool z axis while keeping the same orientation.
        """
        delta_xyz_tool = np.asarray(delta_xyz_tool, dtype=np.float64).reshape(3)
        current_pose = self.get_current_tcp_pose()
        tool_rotation, _ = cv2.Rodrigues(np.asarray(current_pose[3:6], dtype=np.float64))
        delta_xyz_base = tool_rotation @ delta_xyz_tool
        return self.translate_base(delta_xyz_base, k_acc=k_acc, k_vel=k_vel, wait=wait)

    def set_tcp(self, tcp_pose):
        """Set TCP offset on the robot controller. tcp_pose is [x, y, z, rx, ry, rz]."""
        tcp_pose = self._pose_to_rotvec_pose(tcp_pose, pose_format='rotvec')
        self._send_urscript("set_tcp(%s)\n" % self._format_ur_pose(tcp_pose))

    def set_payload(self, mass, cog=None):
        """Set payload mass and optional center of gravity."""
        if cog is None:
            self._send_urscript("set_payload(%f)\n" % float(mass))
            return
        cog = np.asarray(cog, dtype=np.float64).reshape(3)
        self._send_urscript("set_payload(%f,%s)\n" % (float(mass), self._format_ur_vector(cog)))

    def stop_l(self, acc=2.0):
        """Stop linear TCP motion."""
        self._send_urscript("stopl(%f)\n" % float(acc))

    def stop_j(self, acc=2.0):
        """Stop joint motion."""
        self._send_urscript("stopj(%f)\n" % float(acc))

    def speed_l(self, tool_speed, acc=0.25, t=0):
        """Send speedl([vx,vy,vz,wx,wy,wz], a, t)."""
        tool_speed = self._to_float_pose(tool_speed)
        self._send_urscript("speedl(%s,%f,%f)\n" % (self._format_ur_vector(tool_speed), float(acc), float(t)))

    def speed_j(self, joint_speeds, acc=0.5, t=0):
        """Send speedj([q_speed...], a, t)."""
        joint_speeds = self._to_float_pose(joint_speeds)
        self._send_urscript("speedj(%s,%f,%f)\n" % (self._format_ur_vector(joint_speeds), float(acc), float(t)))

    def go_home(self):
        self.move_j(self.home_joint_config)

    def restartReal(self):
        self.go_home()
        # robotiq85 gripper configuration
        if (self.is_use_robotiq85):
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()
            print("Activating gripper...")
            self.gripper.activate()
            time.sleep(1.5)

        # realsense configuration
        if (self.is_use_camera):
            from real.realsenseD435_new1 import Camera
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera()
            # self.cam_intrinsics = self.camera.intrinsics  # get camera intrinsics
            self.cam_intrinsics = self.camera.color_intr          #相机内参
            # # Load camera pose (from running calibrate.py), intrinsics and depth scale
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cam_pose_path = os.path.join(script_dir, 'real/camera_pose.txt')
            cam_depth_path = os.path.join(script_dir, 'real/camera_depth_scale.txt')
            self.cam_pose = np.loadtxt(cam_pose_path, delimiter=' ')
            self.cam_depth_scale = np.loadtxt(cam_depth_path, delimiter=' ')

    # get robot current state and information
    def get_state(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
            tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            return tcp_socket.recv(1500)
    
    # get robot current joint angles and cartesian pose
    def parse_tcp_state_data(self, data, subpasckage):
        dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
               'I target': '6d',
               'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
               'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
               'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
               'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
               'Tool Accelerometer values': '3d',
               'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
               'softwareOnly2': 'd',
               'V main': 'd',
               'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
               'Elbow position': 'd', 'Elbow velocity': '3d'}
        ii = range(len(dic))
        for key, i in zip(dic, ii):
            fmtsize = struct.calcsize(dic[key])
            if len(data) < fmtsize:
                raise ValueError(f"Incomplete TCP state packet while reading {key}")
            data1, data = data[0:fmtsize], data[fmtsize:]
            fmt = "!" + dic[key]
            dic[key] = dic[key], struct.unpack(fmt, data1)

        if subpasckage == 'joint_data':  # get joint data
            q_actual_tuple = dic["q actual"]
            joint_data= np.array(q_actual_tuple[1])
            return joint_data
        elif subpasckage == 'cartesian_info':
            Tool_vector_actual = dic["Tool vector actual"]  # get x y z rx ry rz
            cartesian_info = np.array(Tool_vector_actual[1])
            return cartesian_info

    def rotvec_to_rpy(self,x, y, z, rx, ry, rz):
        """
        将 UR 机械臂的位姿表示（x, y, z, rx, ry, rz）中的旋转向量 (rx, ry, rz)
        转换为 Roll-Pitch-Yaw（r, p, y）欧拉角，返回 (x, y, z, roll, pitch, yaw)。

        参数:
          x, y, z : 浮点数，末端在基坐标系下的平移 (单位: 米)
          rx, ry, rz : 浮点数，旋转向量的三个分量 (Axis‐Angle 表示, 单位: 弧度)

        返回:
          (x, y, z, roll, pitch, yaw)
            roll, pitch, yaw 以弧度为单位，旋转顺序假定为:
              1. 先绕 X 轴旋转 roll
              2. 再绕 Y 轴旋转 pitch
              3. 最后绕 Z 轴旋转 yaw
        """
        # 将旋转向量转换为旋转矩阵
        rot_vec = np.array([rx, ry, rz], dtype=np.float64).reshape(3, 1)
        R_mat, _ = cv2.Rodrigues(rot_vec)

        # 依据 R = Rz(yaw) · Ry(pitch) · Rx(roll) 的顺序反求 r, p, y
        #   R_mat[2,0] = -sin(pitch)
        #   R_mat[2,1] = cos(pitch) * sin(roll)
        #   R_mat[2,2] = cos(pitch) * cos(roll)
        #   R_mat[1,0] = sin(yaw) * cos(pitch)
        #   R_mat[0,0] = cos(yaw) * cos(pitch)
        #
        # 因此:
        #   pitch = asin(-R_mat[2,0])
        #   roll  = atan2(R_mat[2,1], R_mat[2,2])
        #   yaw   = atan2(R_mat[1,0], R_mat[0,0])

        # 计算 pitch（弧度）
        pitch = np.arcsin(-R_mat[2, 0])

        # 处理 Gimbal Lock 情况
        if np.isclose(np.cos(pitch), 0.0, atol=1e-6):
            # 当 cos(pitch) ≈ 0 时，roll 和 yaw 可能无法唯一确定
            # 此处将 roll 设为 0，yaw 按照下面公式近似计算
            roll = 0.0
            yaw = np.arctan2(-R_mat[0, 1], R_mat[1, 1])
        else:
            roll = np.arctan2(R_mat[2, 1], R_mat[2, 2])
            yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])

        return x, y, z, roll, pitch, yaw

    def rpy2rotating_vector(self,rpy):
        # rpy to R
        R = self.rpy2R(rpy)#先将rpy欧拉角转换成旋转矩阵R
        # R to rotating_vector
        rotvec, _ = cv2.Rodrigues(R)
        return rotvec.reshape(3)#再将旋转R转换成旋转向量

    def rpy2R(self,rpy): # [r,p,y] 单位rad
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                          [0, math.sin(rpy[0]), math.cos(rpy[0])]])
        rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                          [0, 1, 0],
                          [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
        rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                          [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                          [0, 0, 1]])
        R = np.dot(rot_z, np.dot(rot_y, rot_x))
        return R

    def R2rotating_vector(self,R):
        rotvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
        return rotvec.reshape(3)

    def R2rpy(self,R):
    # assert (isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    ## robotiq85 gripper
    # get gripper position [0-255]  open:0 ,close:255获得夹爪的开合程度
    def get_current_tool_pos(self):
        return self.gripper.get_current_position()       

    def log_gripper_info(self):
        print(f"Pos: {str(self.gripper.get_current_position())}")

    def close_gripper(self,speed=255,force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(255, speed, force)
        print("gripper had closed!")
        time.sleep(1.2)
        self.log_gripper_info()

    def open_gripper(self,speed=255,force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(0, speed, force)
        print("gripper had opened!")
        time.sleep(1.2)
        self.log_gripper_info()

    def get_current_tcp_pose(self):

        #获取当前 TCP（末端）位姿 [x, y, z, rx, ry, rz]，单位：米和弧度
        #获得的是末端 TCP 在机器人基坐标系（base_link）下的位置
        with self._connect_robot_socket() as tcp_socket:
            self.tcp_socket = tcp_socket
            # 接收状态数据包
            state_data = tcp_socket.recv(1500)
            # 解析 TCP 位姿
            return self.parse_tcp_state_data(state_data, 'cartesian_info')

    ## get camera data
    def get_camera_data(self):
        color_img, depth_img = self.camera.get_data()
        return color_img, depth_img

    def shutdown(self):
        """
        安全地关闭所有与机器人相关的资源，包括相机和夹爪。
        这个函数被设计为可以被安全地多次调用，并且能处理资源未初始化的情况。
        """
        print("\n" + "=" * 20 + " 开始关闭 UR_Robot 资源 " + "=" * 20)

        # --- 1. 关闭相机 ---
        # 检查 is_use_camera 标志位，并且确认 self.camera 对象确实存在
        if self.is_use_camera and hasattr(self, 'camera') and self.camera is not None:
            print("[SHUTDOWN] 正在关闭相机...")
            try:
                # 调用相机自己的 stop 方法，它会处理线程和pipeline
                self.camera.stop()
                print("  ✅ 相机已成功关闭。")
            except Exception as e:
                # 即使关闭失败，也只打印错误，不让整个程序崩溃
                print(f"  ❌ 关闭相机时发生错误: {e}")
            # 将对象置为None，防止被重复关闭
            self.camera = None

            # --- 2. 关闭夹爪 ---
        # 检查 is_use_robotiq85 标志位，并且确认 self.gripper 对象确实存在
        if self.is_use_robotiq85 and hasattr(self, 'gripper') and self.gripper is not None:
            print("[SHUTDOWN] 正在关闭夹爪连接...")
            try:
                # 调用夹爪的 disconnect 方法
                self.gripper.disconnect()
                print("  ✅ 夹爪连接已成功关闭。")
            except Exception as e:
                print(f"  ❌ 关闭夹爪时发生错误: {e}")
            # 将对象置为None
            self.gripper = None

        print("=" * 20 + " UR_Robot 资源已全部处理 " + "=" * 20 + "\n")

    # Note: must be preceded by close_gripper()
    def check_grasp(self):
        # if the robot grasp unsuccessfully ,then the gripper close
        return self.get_current_tool_pos()>220

    def plane_grasp(self, position, yaw=0, open_size=0.65, k_acc=0.8,k_vel=0.8,speed=255, force=125):
        rpy = [-np.pi, 0, 1.57 - yaw]
        # 判定抓取的位置是否处于工作空间
        for i in range(3):
            position[i] = min(max(position[i],self.workspace_limits[i][0]),self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [-pi,pi]
        for i in range(3):
            if rpy[i] > np.pi:
                rpy[i] -= 2*np.pi
            elif rpy[i] < -np.pi:
                rpy[i] += 2*np.pi
        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2],rpy[0],rpy[1],rpy[2]))

        # pre work
        grasp_home = [0.4,0,0.4,-np.pi,0,0]  # you can change me
        self.move_j_p(grasp_home,k_acc,k_vel)
        open_pos = int(-258*open_size +230)  # open size:0~0.85cm --> open pos:230~10是对应的控制信号，0-255  0是闭合 255是张开
        self.gripper.move_and_wait_for_pos(open_pos, speed, force)
        print("gripper open size:")
        self.log_gripper_info()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis 加上10cm
        # print(pre_position)
        self.move_j_p(pre_position + rpy,k_acc,k_vel)

        # Second，achieve grasp position
        self.move_l(position+rpy,0.6*k_acc,0.6*k_vel)
        self.close_gripper(speed,force)
        self.move_l(pre_position + rpy, 0.6*k_acc,0.6*k_vel)
        if(self.check_grasp()):
            print("Check grasp fail! ")
            self.move_j_p(grasp_home)
            return False
        # Third,put the object into box
        box_position = [0.63,0,0.25,-np.pi,0,0]  # you can change me!
        self.move_j_p(box_position,k_acc,k_vel)
        box_position[2] = 0.1  # down to the 10cm
        self.move_j_p(box_position, k_acc, k_vel)
        self.open_gripper(speed,force)
        box_position[2] = 0.25
        self.move_j_p(box_position, k_acc, k_vel)
        self.move_j_p(grasp_home)
        print("grasp success!")
        return True

    def plane_push(self, position, move_orientation=0, length=0.1):
        for i in range(2):
            position[i] = min(max(position[i],self.workspace_limits[i][0]+0.1),self.workspace_limits[i][1]-0.1)
        position[2] = min(max(position[2],self.workspace_limits[2][0]),self.workspace_limits[2][1])
        print('Executing: push at (%f, %f, %f) and the orientation is %f' % (position[0], position[1], position[2],move_orientation))

        push_home = [0.4, 0, 0.4, -np.pi, 0, 0]
        self.move_j_p(push_home,k_acc=1, k_vel=1)  # pre push position(push home)
        # self.close_gripper()

        self.move_j_p([position[0],position[1],position[2]+0.1,-np.pi,0,0],k_acc=1,k_vel=1)
        self.move_j_p([position[0], position[1], position[2], -np.pi, 0, 0], k_acc=0.6, k_vel=0.6)

        # compute the destination pos
        destination_pos = [position[0] + length * math.cos(move_orientation),position[1] + length * math.sin(move_orientation),position[2]]
        self.move_l(destination_pos+[-np.pi, 0, 0], k_acc=0.5, k_vel=0.5)
        self.move_j_p([destination_pos[0],destination_pos[1],destination_pos[2]+0.1,-np.pi,0,0],k_acc=0.6, k_vel=0.6)

        # go back push-home
        self.move_j_p(push_home, k_acc=1, k_vel=1)

    def grasp(self, position, rpy=None, open_size=0.85, k_acc=0.8, k_vel=0.8, speed=255, force=125):

        # 判定抓取的位置是否处于工作空间
        if rpy is None:
            rpy = [-np.pi, 0, 0]
        for i in range(3):
            position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [0.5*pi,1.5*pi]
        for i in range(3):
            if rpy[i] > np.pi:
                rpy[i] -= 2 * np.pi
            elif rpy[i] < -np.pi:
                rpy[i] += 2 * np.pi
        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]))

        # pre work
        grasp_home = [0.4, 0, 0.4, -np.pi, 0, 0]  # you can change me
        self.move_j_p(grasp_home, k_acc, k_vel)
        open_pos = int(-300 * open_size + 255)  # open size:0~0.85cm --> open pos:255~0
        self.gripper.move_and_wait_for_pos(open_pos, speed, force)
        self.log_gripper_info()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis
        print(pre_position)
        self.move_j_p(pre_position + rpy, k_acc, k_vel)

        # Second，achieve grasp position
        self.move_l(position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        self.close_gripper(speed, force)
        self.move_l(pre_position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        if (self.check_grasp()):
            print("Check grasp fail! ")
            self.move_j_p(grasp_home)
            return False
        # Third,put the object into box
        box_position = [0.63, 0, 0.25, -np.pi, 0, 0]  # you can change me!
        self.move_j_p(box_position, k_acc, k_vel)
        box_position[2] = 0.1  # down to the 10cm
        self.move_l(box_position, k_acc, k_vel)
        self.open_gripper(speed, force)
        box_position[2] = 0.25
        self.move_l(box_position, k_acc, k_vel)
        self.move_j_p(grasp_home)
        print("grasp success!")

if __name__ =="__main__":
    ur_robot = UR_Robot()

