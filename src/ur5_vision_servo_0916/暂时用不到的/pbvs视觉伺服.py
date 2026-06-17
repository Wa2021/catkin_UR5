
#左转90度
import numpy as np
import cv2
import json
import time
from UR_Robot import UR_Robot
from scipy.spatial.transform import Rotation as R

def load_hand_eye_result(path='calibration_result1_4.json'):
    with open(path, 'r') as f:
        calib = json.load(f)
    camera_matrix = np.array(calib['camera_matrix'])
    dist_coeffs = np.array(calib['dist_coeffs'])
    T_cam2tool = np.array(calib['hand_eye_matrix'])
    return camera_matrix, dist_coeffs, T_cam2tool

def solve_pnp_pose(color_img, camera_matrix, dist_coeffs, pattern_size=(8, 5), square_size=0.027):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None

    # 亚像素优化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 构造目标点
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not success:
        return None

    return rvec, tvec
    #rvec是旋转向量，tvec是平移向量

def pbvs_control():
    robot = UR_Robot("192.168.0.1")
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result()

    DESIRED_DISTANCE = 0.25  # 目标与 TCP 之间保持25cm
    THRESHOLD = 0.005        # 到达判定阈值（米）
    TRACKING_INTERVAL = 0.2  # 控制帧间隔（秒）
    MAX_ERROR_TO_MOVE = 0.002  # 若小于该值则不移动，防止抖动

    print("[启动] PBVS 追踪已开始，按 Ctrl+C 停止...")

    try:
        while True:
            color_img, _ = robot.get_camera_data()
            result = solve_pnp_pose(color_img, camera_matrix, dist_coeffs)
            if result is None:
                print("[警告] 未检测到棋盘格，等待中...")
                time.sleep(0.1)
                continue

            rvec, tvec = result
            R_obj2cam, _ = cv2.Rodrigues(rvec)
            T_obj2cam = np.eye(4)
            T_obj2cam[:3, :3] = R_obj2cam
            T_obj2cam[:3, 3] = tvec.flatten()

            # 插入偏移距离（Z轴后退25cm）
            offset = np.eye(4)
            offset[2, 3] = -DESIRED_DISTANCE
            T_target2cam = offset @ T_obj2cam
            '''
            无论标定板怎么倾斜，你的机械臂总能移动到垂直于标定板上方25cm的那个点。这才是我们通常想要的“悬停”操作。

            总结：直接减法是在**相机坐标系**里移动，矩阵乘法是在**物体（棋盘格）自己的坐标系**里移动。
            '''

            # 相机 → 工具 → 基坐标
            T_tool2base = np.eye(4)
            tcp_pose = robot.get_current_tcp_pose()
            R_base_tool = R.from_rotvec(tcp_pose[3:]).as_matrix()
            T_tool2base[:3, :3] = R_base_tool
            T_tool2base[:3, 3] = tcp_pose[:3]

            T_target2tool = T_cam2tool @ T_target2cam
            T_target2base = T_tool2base @ T_target2tool

            pos = T_target2base[:3, 3]
            rpy = robot.R2rpy(T_target2base[:3, :3])
            target_pose = np.concatenate([pos, rpy])

            # 获取当前 TCP 姿态
            current_tcp = robot.get_current_tcp_pose()
            error = np.linalg.norm(current_tcp[:3] - pos)
            print(f"[追踪] 距离目标误差：{error:.4f} m")

            if error > MAX_ERROR_TO_MOVE:
                print(f"[移动] 正在追踪目标：pos={pos.round(4)}, rpy={rpy.round(3)}")
                robot.move_j_p(target_pose.tolist(), k_acc=0.3, k_vel=0.3)
            else:
                print("[暂停] 已对准目标，等待更新...")

            time.sleep(TRACKING_INTERVAL)

    except KeyboardInterrupt:
        print("\n[中止] 已停止PBVS追踪。")


if __name__ == '__main__':
    pbvs_control()