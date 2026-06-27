# 本次修正说明：
#     相比于1.2改了一下tcp，然后初始坐标也改了一下
import numpy as np
import cv2
import time
from UR_Robot import UR_Robot
import json


class HandEyeCalibrator:
    def __init__(self, robot_ip="192.168.0.1", pattern_size=(8, 5), square_size=0.027, use_gripper=False):
        """
        初始化手眼标定器
        :param robot_ip: UR机器人IP地址
        :param pattern_size: 标定板角点数 (宽, 高)
        :param square_size: 标定板方格大小(米)
        :param use_gripper: 是否使用机械夹爪
        """
        # 安全限制
        self.safe_height = 0.3  # 安全高度（米）
        self.max_speed = 0.3  # 最大速度限制
        self.max_acc = 0.3  # 最大加速度限制
        self.safe_zone = {  # 安全工作区域限制
            'x': [-0.5, 0.5],  # x轴范围（米）
            'y': [-0.5, 0.5],  # y轴范围（米）
            'z': [0.1, 0.6]  # z轴范围（米）
        }

        print("\n=== 安全提示 ===")
        print("1. 确保工作区域内无人和障碍物")
        print("2. 标定过程中保持观察，随时准备按下急停按钮")
        print("3. 机器人移动速度和加速度已限制")
        print("4. 工作区域已限制在安全范围内")
        print(f"   X: {self.safe_zone['x']}米")
        print(f"   Y: {self.safe_zone['y']}米")
        print(f"   Z: {self.safe_zone['z']}米")
        print("5. 每次移动前都会进行确认")
        print("6. 按Ctrl+C可随时终止程序")
        print("==============\n")

        self.robot = UR_Robot(tcp_host_ip=robot_ip, is_use_robotiq85=use_gripper)
        self.pattern_size = pattern_size
        self.square_size = square_size

        # 初始化标定数据存储
        self.object_points = []  # 世界坐标系中的点
        self.image_points = []  # 图像坐标系中的点
        self.robot_poses = []  # 机器人位姿

        # 生成标定板世界坐标
        self.pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.pattern_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.pattern_points *= square_size

        self.base_pose_rpy = np.array(
            self.robot.rotvec_to_rpy(
                -0.478, -0.0678, 0.336,  # x, y, z
                2.222, -2.22, -0.140 ) # rx, ry, rz
            )


    def check_pose_safety(self, pose):
        """检查位姿是否在安全范围内"""
        if not (self.safe_zone['x'][0] <= pose[0] <= self.safe_zone['x'][1]):
            raise Exception(f"X轴位置 {pose[0]} 超出安全范围 {self.safe_zone['x']}")
        if not (self.safe_zone['y'][0] <= pose[1] <= self.safe_zone['y'][1]):
            raise Exception(f"Y轴位置 {pose[1]} 超出安全范围 {self.safe_zone['y']}")
        if not (self.safe_zone['z'][0] <= pose[2] <= self.safe_zone['z'][1]):
            raise Exception(f"Z轴位置 {pose[2]} 超出安全范围 {self.safe_zone['z']}")
        return True

    def generate_calibration_poses(self):
        """
        生成标定位姿序列，确保所有位姿都在安全范围内
        返回: 基座标系下的标定位姿列表
        """
        base_pose = self.base_pose_rpy.copy()
        print(f"[DEBUG] base_pose = {base_pose}")

        poses = []

        # -----------------------------
        # 2) 按高度（Z）做微调
        # -----------------------------
        heights = [-0.02, 0, 0.02]  # 在基准高度上下各 2cm
        for h in heights:
            pose = base_pose.copy()
            pose[2] += h
            if self.check_pose_safety(pose):
                poses.append(pose)
                print(f"[DEBUG] 加入合法高度 Z 位姿: {pose}")
            else:
                print(f"[DEBUG] 被丢弃的高度 Z 位姿 (不安全): {pose}")

        # -----------------------------
        # 3) 按 X、Y 平面做微调
        # -----------------------------
        xy_offsets = [ 0, 0.02,0.04]  # X/Y ±2cm
        # X 方向偏移
        for dx in xy_offsets:
            pose = base_pose.copy()
            pose[0] += dx
            if self.check_pose_safety(pose):
                poses.append(pose)
                print(f"[DEBUG] 加入合法 X 方向偏移位姿: {pose}")
            else:
                print(f"[DEBUG] 被丢弃的 X 方向偏移位姿 (X={pose[0]:.3f} 超出): {pose}")
        # Y 方向偏移
        for dy in xy_offsets:
            pose = base_pose.copy()
            pose[1] += dy
            if self.check_pose_safety(pose):
                poses.append(pose)
                print(f"[DEBUG] 加入合法 Y 方向偏移位姿: {pose}")
            else:
                print(f"[DEBUG] 被丢弃的 Y 方向偏移位姿 (Y={pose[1]:.3f} 超出): {pose}")

        # -----------------------------
        # 4) 大范围旋转变化 (15-30度)
        # -----------------------------
        # Roll(rx)大范围旋转 - 约±25度
        roll_angles = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45]
        for dr in roll_angles:
            pose = base_pose.copy()
            pose[3] += dr
            if self.check_pose_safety(pose):
                poses.append(pose)
                print(f"[DEBUG] 加入大范围 Roll(rx) 旋转位姿: {pose}")
            else:
                print(f"[DEBUG] 被丢弃的 Roll(rx) 旋转位姿: {pose}")

        # Pitch(ry)大范围旋转 - 约±25度
        pitch_angles = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45]
        for dp in pitch_angles:
            pose = base_pose.copy()
            pose[4] += dp
            if self.check_pose_safety(pose):
                poses.append(pose)
                print(f"[DEBUG] 加入大范围 Pitch(ry) 旋转位姿: {pose}")
            else:
                print(f"[DEBUG] 被丢弃的 Pitch(ry) 旋转位姿: {pose}")



        # -----------------------------
        # 5) 打印最终 summary，注意先判断是否为空
        # -----------------------------
        print("\n=== 生成标定位姿 ===")
        print(f"基准位姿: {[round(x, 4) for x in base_pose]}")
        print(f"生成位姿数量: {len(poses)}")
        if len(poses) > 0:
            xs = [p[0] for p in poses]
            ys = [p[1] for p in poses]
            zs = [p[2] for p in poses]
            print("位姿范围:")
            print(f"  X: {min(xs):.3f} ～ {max(xs):.3f}")
            print(f"  Y: {min(ys):.3f} ～ {max(ys):.3f}")
            print(f"  Z: {min(zs):.3f} ～ {max(zs):.3f}")
        else:
            print("（警告：所有位姿都被安全检测过滤，poses 为空！请检查偏移量或安全范围设置。）")
        print("====================\n")

        return poses



    def collect_calibration_data(self, num_poses=None):
        """
        自动采集标定数据
        :param num_poses: 如果指定，则只采集前num_poses个位姿的数据
        """
        try:
            print("\n=== 开始标定数据采集 ===")
            print("注意事项：")
            print("1. 机器人将在基准位置附近移动")
            print("2. 每次移动前都会请求确认")
            print("3. 可以随时按Ctrl+C终止程序")
            print("4. 确保标定板在相机视野内\n")

            # 生成标定位姿
            poses = self.generate_calibration_poses()
            if num_poses is not None:
                poses = poses[:num_poses]

            print(f"计划采集 {len(poses)} 组数据")

            # 首先移动到基准位置
            base_pose = self.base_pose_rpy.copy()

            print("\n移动到基准位置...")
            print(f"基准位置: {[round(x, 4) for x in base_pose]}")

            confirm = input("是否移动到基准位置? (y/n): ")
            if confirm.lower() != 'y':
                print("用户取消操作")
                return

            self.robot.move_j_p(base_pose, k_acc=0.3, k_vel=0.3)
            time.sleep(2)  # 等待机器人稳定

            # 如果从未记录过图像尺寸，在第一次 get_camera_data 时保存下来
            if not hasattr(self, 'image_size'):
                self.image_size = None

            for i, pose in enumerate(poses):
                print(f"\n=== 第 {i + 1}/{len(poses)} 个位姿 ===")
                print(f"目标位姿: {pose}")
                print("请确认:")
                print("1. 工作区域内无障碍物")
                print("2. 标定板位置正确")
                print("3. 相机视野正常")

                confirm = input("是否移动到这个位置? (y/no/stop): ")
                if confirm.lower() == 'stop':
                    print("用户请求停止，程序终止")
                    break
                if confirm.lower() != 'y':
                    print("已跳过当前位姿")
                    continue

                try:
                    # 移动到标定位姿
                    self.robot.move_j_p(pose, k_acc=self.max_acc, k_vel=self.max_speed)
                    time.sleep(2)  # 等待机器人稳定

                    # 获取一帧图像并在第一次时记录尺寸
                    rgb_image, _ = self.robot.get_camera_data()
                    if self.image_size is None:
                        self.image_size = (rgb_image.shape[1], rgb_image.shape[0])
                        print(f"[DEBUG] 已记录图像尺寸: {self.image_size}")

                    # 显示实时图像
                    cv2.imshow('Calibration', rgb_image)
                    cv2.waitKey(500)

                    # 转灰度并检测角点
                    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCorners(
                        gray,
                        self.pattern_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                    )

                    if found:
                        # 亚像素角点检测
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                        # 绘制角点
                        cv2.drawChessboardCorners(rgb_image, self.pattern_size, corners, found)
                        cv2.imshow('Calibration', rgb_image)
                        cv2.waitKey(3000)

                        # 确认是否保存当前数据
                        save = input("检测到标定板，是否保存当前数据? (y/no): ")
                        if save.lower() == 'y':
                            # 存储数据
                            self.object_points.append(self.pattern_points)
                            self.image_points.append(corners)
                            self.robot_poses.append(self.robot.get_current_tcp_pose())
                            print(f"成功采集第 {len(self.robot_poses)} 组数据")
                        else:
                            print("已取消保存，继续下一个位姿")
                    else:
                        print("未检测到标定板，继续下一个位姿")
                        cv2.imshow('Calibration', rgb_image)
                        cv2.waitKey(3000)

                except Exception as e:
                    print(f"移动过程出错: {e}")
                    retry = input("是否重试当前位姿? (yes/no): ")
                    if retry.lower() == 'yes':
                        i -= 1  # 重试当前位姿
                        continue

                # 每次采集后回到安全高度
                safe_pose = pose.copy()
                safe_pose[2] = self.safe_height
                self.robot.move_j_p(safe_pose, k_acc=self.max_acc, k_vel=self.max_speed)

        except KeyboardInterrupt:
            print("\n用户中断，程序终止")
            # 确保机器人回到安全位置
            self.robot.move_j_p(base_pose, k_acc=0.3, k_vel=0.3)
        finally:
            cv2.destroyAllWindows()
            print("\n数据采集完成或终止!")
            print(f"共采集 {len(self.robot_poses)} 组有效数据")

            # 检查数据是否足够
            if len(self.robot_poses) < 4:
                print("警告：有效数据少于4组，可能影响标定精度")
                if input("是否继续采集更多数据? (yes/no): ").lower() == 'yes':
                    self.collect_calibration_data(num_poses=len(poses))

    def calibrate(self):
        """
        执行手眼标定
        :return: 相机内参矩阵, 畸变系数, 手眼变换矩阵
        """
        if len(self.object_points) < 4:
            raise ValueError("标定数据不足，至少需要4组数据")

        if not hasattr(self, 'image_size') or self.image_size is None:
            raise RuntimeError(
                "未记录图像尺寸，无法进行相机标定。请先运行 collect_calibration_data() 并采集至少一帧图像。")

        img_size = self.image_size

        # 相机标定：获取内参 + T_target_to_cam（每帧）
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points,
            img_size,
            None, None
        )
        # 返回参数解释：
        # ret           —— 重投影误差（re-projection error），衡量标定精度，越小越好
        # camera_matrix —— 相机内参矩阵
        # dist_coeffs   —— 相机畸变系数（包括径向和切向畸变）
        # rvecs         —— 每幅图像的旋转向量（相机坐标系到标定板坐标系）
        # tvecs         —— 每幅图像的平移向量（相机坐标系到标定板坐标系）

        # 保存以供后续误差计算使用
        self.rvecs = rvecs
        self.tvecs = tvecs

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for i in range(len(rvecs)):
            # 直接使用 OpenCV 返回的 T_target_to_cam
            R_tc, _ = cv2.Rodrigues(rvecs[i])
            t_tc = tvecs[i].reshape(3, 1)

            R_target2cam.append(R_tc)
            t_target2cam.append(t_tc)

            # 同步构建 base → gripper（机器人数据）
            pose = self.robot_poses[i]
            R_gb, t_gb = self._pose_to_matrix(pose)
            R_gripper2base.append(R_gb)
            t_gripper2base.append(t_gb)

        # 执行手眼标定：输出 T_cam_to_gripper
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # 构建齐次变换矩阵
        hand_eye_matrix = np.eye(4)
        hand_eye_matrix[:3, :3] = R_cam2gripper
        hand_eye_matrix[:3, 3] = t_cam2gripper.reshape(-1)

        # 重投影误差评估
        mean_error = self._compute_calibration_error(
            camera_matrix, dist_coeffs, hand_eye_matrix,
            self.object_points, self.image_points, self.robot_poses
        )

        print(f"\n标定完成!")
        print(f"平均重投影误差: {mean_error:.3f} 像素")

        return camera_matrix, dist_coeffs, hand_eye_matrix

    def _compute_calibration_error(self, camera_matrix, dist_coeffs, hand_eye_matrix,
                                   object_points, image_points, robot_poses):
        """
        计算标定误差：评估相机内参与手眼变换矩阵在投影精度上的表现。
        使用 cv2.calibrateCamera() 返回的 T_target_to_cam（rvecs, tvecs）直接进行投影。
        """
        total_error = 0
        total_points = 0

        for i in range(len(object_points)):
            # 使用标定阶段的旋转向量和平移向量（T_target_to_cam）
            rvec = self.rvecs[i]
            tvec = self.tvecs[i].reshape(3, 1)

            projected_points, _ = cv2.projectPoints(
                object_points[i], rvec, tvec,
                camera_matrix, dist_coeffs
            )

            projected_points = projected_points.reshape(-1, 2).astype(np.float32)
            image_pts = image_points[i].reshape(-1, 2).astype(np.float32)

            error = cv2.norm(image_pts, projected_points, cv2.NORM_L2)
            total_error += error
            total_points += len(object_points[i])

        return total_error / total_points

    def save_calibration(self, filename="calibration_result1_4.json"):
        """
        保存标定结果
        """
        camera_matrix, dist_coeffs, hand_eye_matrix = self.calibrate()

        calibration_data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "hand_eye_matrix": hand_eye_matrix.tolist()
        }

        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=4)

        print(f"标定结果已保存到 {filename}")

    def _pose_to_matrix(self, pose):
        """
        将位姿转换为旋转矩阵和平移向量
        :param pose: [x, y, z, rx, ry, rz]
        :return: R, t
        """
        R = cv2.Rodrigues(np.array(pose[3:]))[0]
        t = np.array(pose[:3]).reshape(3, 1)
        return R, t


def main():
    # 获取机器人IP地址
    robot_ip = input("请输入UR机器人的IP地址 (默认为192.168.0.1): ").strip()
    if not robot_ip:
        robot_ip = "192.168.0.1"

    print(f"\n正在尝试连接到机器人 ({robot_ip})...")
    print("请确保：")
    print("1. 机器人已开机并正常运行")
    print("2. 机器人和电脑在同一网段")
    print("3. 可以ping通机器人IP")
    print("4. 相机已正确连接")
    print("5. 标定板已放置在机器人工作空间内\n")

    try:
        # 创建标定器实例
        calibrator = HandEyeCalibrator(
            robot_ip=robot_ip,
            pattern_size=(8, 5),  # 根据实际标定板修改
            square_size=0.027,  # 根据实际标定板修改(米)
            use_gripper=True  # 使用/不使用夹爪进行标定
        )

        # 自动采集标定数据
        calibrator.collect_calibration_data(num_poses=23)

        # 执行标定并保存结果
        calibrator.save_calibration()

    except ConnectionRefusedError:
        print("\n错误：无法连接到机器人！")
        print("请检查：")
        print("1. 机器人IP地址是否正确")
        print("2. 机器人是否已开机")
        print("3. 网络连接是否正常")
        print(f"4. 是否可以ping通 {robot_ip}")
    except Exception as e:
        print(f"\n发生错误：{e}")
        print("请检查上述错误信息并重试")


if __name__ == "__main__":
    main()
