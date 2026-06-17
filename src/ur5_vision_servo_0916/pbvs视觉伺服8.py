"""
在第7版的基础上，将计算部分封装成一个独立的函数，并对主循环进行了相应的简化
尝试解决相机图像阻塞问题，创建了一个新类，在finally那边加了几行代码
但是还没有测试效果

"""
import numpy as np
import cv2
import json
import time
from UR_Robot import UR_Robot
from scipy.spatial.transform import Rotation as R


def load_hand_eye_result(path='calibration_result1_4.json'):
    """加载相机内参、畸变系数和手眼标定矩阵"""
    try:
        with open(path, 'r') as f:
            calib = json.load(f)
        camera_matrix = np.array(calib['camera_matrix'])
        dist_coeffs = np.array(calib['dist_coeffs'])
        T_cam2tool = np.array(calib['hand_eye_matrix'])
        print(f"[成功] 已从 {path} 加载相机参数和手眼矩阵。")
        return camera_matrix, dist_coeffs, T_cam2tool
    except FileNotFoundError:
        print(f"[错误] 找不到标定文件: {path}！请确保文件存在。")
        exit()


def solve_pnp_pose(color_img, camera_matrix, dist_coeffs, pattern_size, square_size):
    """在图像中寻找棋盘格，计算其位姿(rvec, tvec)和中心点在物体坐标系下的坐标(P_center_in_obj)"""
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None, None, None
    #继续优化像素，让其更精细
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    #生成理想的棋盘格坐标系，用于和实际的对应
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not success:
        return None, None, None

    #计算棋盘格中点
    center_x = (pattern_size[0] - 1) / 2.0 * square_size
    center_y = (pattern_size[1] - 1) / 2.0 * square_size
    P_center_in_obj = np.array([center_x, center_y, 0])

    return rvec, tvec, P_center_in_obj


def draw_camera_center(image, camera_matrix, size=15, color=(255, 0, 0), thickness=2):
    """在图像上绘制相机中心（主点），蓝色十字准星"""
    cx = int(camera_matrix[0, 2])
    cy = int(camera_matrix[1, 2])
    cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)
    return image


def draw_target_point(image, P_center_in_cam, camera_matrix, dist_coeffs):
    """将目标点（标定板中心）投影到图像上并绘制出来"""
    target_point_2d, _ = cv2.projectPoints(P_center_in_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    pt = tuple(target_point_2d[0][0].astype(int))
    cv2.circle(image, pt, 8, (0, 165, 255), -1)  # 橙色的圆点
    cv2.circle(image, pt, 9, (255, 255, 255), 1)  # 白色轮廓
    cv2.putText(image, "Target Center", (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return image


def compute_servoing_target(tcp_pose, rvec, tvec, P_center_in_obj, T_cam2tool,
                            R_correction_current, desired_distance, key_pressed=None):
    """
    根据机器人当前位姿和视觉检测结果，计算伺服的目标位姿。

    :param tcp_pose: 机器人当前TCP位姿 [x, y, z, rx, ry, rz] (rotvec)
    :param rvec: 视觉检测到的旋转向量
    :param tvec: 视觉检测到的平移向量
    :param P_center_in_obj: 目标中心点在物体坐标系下的坐标
    :param T_cam2tool: 手眼标定矩阵 (相机->工具)
    :param R_correction_current: 当前的姿态修正矩阵。如果为None或按键按下，则计算新的。
    :param desired_distance: 期望相机与目标的距离
    :param key_pressed: 外部传入的按键值，用于触发重置

    :return: (原始目标位姿[pos, rpy], 更新后的姿态修正矩阵, 用于可视化的相机系下中心点)
    """
    # --- 1. 构造各种变换矩阵 ---
    T_obj2cam = np.eye(4)
    R_obj2cam, _ = cv2.Rodrigues(rvec)
    T_obj2cam[:3, :3] = R_obj2cam
    T_obj2cam[:3, 3] = tvec.flatten()
    T_tool2base = np.eye(4)
    R_tool2base_current = R.from_rotvec(tcp_pose[3:]).as_matrix()
    T_tool2base[:3, :3] = R_tool2base_current
    T_tool2base[:3, 3] = tcp_pose[:3]
    T_cam2base = T_tool2base @ T_cam2tool
    T_obj2base = T_cam2base @ T_obj2cam
    R_obj2base = T_obj2base[:3, :3]

    # --- 2. 计算或更新姿态修正矩阵 ---
    R_correction_updated = R_correction_current
    if R_correction_updated is None or (key_pressed is not None and key_pressed == ord('s')):
        R_correction_updated = R_obj2base.T @ R_tool2base_current
        rpy_corr = R.from_matrix(R_correction_updated).as_euler('xyz', degrees=False)
        print(f"\n[成功] 姿态基准已锁定/更新！修正RPY: {np.round(rpy_corr, 3)}\n")

    # --- 3. 计算最终的目标位姿 ---
    R_tool_desired = R_obj2base @ R_correction_updated
    rpy_desired = R.from_matrix(R_tool_desired).as_euler('xyz', degrees=False)

    R_cam_desired_in_base = R_tool_desired @ T_cam2tool[:3, :3]
    z_axis_cam_desired_in_base = R_cam_desired_in_base[:, 2]

    P_center_in_base = (T_obj2base @ np.append(P_center_in_obj, 1))[:3]

    P_cam_desired_in_base = P_center_in_base + z_axis_cam_desired_in_base * desired_distance

    t_cam2tool_in_tool = T_cam2tool[:3, 3]
    pos_desired = P_cam_desired_in_base - R_tool_desired @ t_cam2tool_in_tool

    raw_target_pose_xyzrpy = np.concatenate([pos_desired, rpy_desired])

    # --- 4. 计算用于可视化的数据 ---
    P_center_in_cam = (T_obj2cam @ np.append(P_center_in_obj, 1))[:3]

    return raw_target_pose_xyzrpy, R_correction_updated, P_center_in_cam

# =============================================================================
#  2. 主功能函数
# =============================================================================

def unlocked_pose_visual_servoing_final():
    """
    V8 最终版：重构后的高性能PD伺服
    - 核心计算逻辑被封装到 compute_servoing_target 函数中
    - 主循环更清晰，易于阅读和维护
    """
    # --- 参数配置 ---
    ROBOT_IP = "192.168.0.1"
    CALIB_FILE = 'calibration_result1_4.json'
    PATTERN_SIZE = (8, 5)#标定板形状
    SQUARE_SIZE = 0.027#标定板单个正方形变成
    DESIRED_DISTANCE = -0.45#相机与标定板的距离
    Z_SAFE_LIMIT = 0.05#z轴过低保护
    VISUALIZATION_ENABLED = True#是否启用图形化界面
    FINE_TUNING_THRESHOLD = 0.01#小于这个值进入精调模式
    MAX_SPEED, MIN_SPEED, MOVE_ACCEL = 0.5, 0.01, 0.4
    KP, KD = 2.5, 2.0#pd控制参数
    POSITION_SMOOTHING_ALPHA = 0.7#位置平滑滤波器参数
    LOOP_INTERVAL = 0.1#循环频率（10hz）

    # --- 初始化 ---
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result(CALIB_FILE)
    try:
        robot = UR_Robot(ROBOT_IP)
    except Exception as e:
        print(f"[错误] 初始化机器人失败: {e}")
        return

    print("\n[启动] V8 重构版伺服已开始。")
    if VISUALIZATION_ENABLED:
        print("[提示] 按 's' 锁定/重置姿态基准, 按 'q' 退出。")
    else:
        print("[提示] 可视化已禁用。按 Ctrl+C 退出程序。")

    R_correction, smoothed_target_pos, last_error = None, None, 0.0

    try:
        while True:
            start_time = time.time()
            color_img, _ = robot.get_camera_data()
            if color_img is None: time.sleep(0.1); continue

            rvec, tvec, P_center_in_obj = solve_pnp_pose(color_img, camera_matrix, dist_coeffs, PATTERN_SIZE,
                                                         SQUARE_SIZE)

            key = -1
            if VISUALIZATION_ENABLED:
                display_img = color_img.copy()
                key = cv2.waitKey(1) & 0xFF

            if rvec is None:
                if VISUALIZATION_ENABLED:
                    cv2.putText(display_img, "Chessboard Not Found", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                else:
                    print("[警告] 未检测到棋盘格...", end='\r')
                smoothed_target_pos, last_error = None, 0.0
            else:
                # === 1. 调用核心计算函数 ===
                tcp_pose = robot.get_current_tcp_pose()
                raw_target_pose, R_correction, P_center_in_cam = compute_servoing_target(
                    tcp_pose, rvec, tvec, P_center_in_obj, T_cam2tool,
                    R_correction, DESIRED_DISTANCE, key
                )
                raw_target_pos = raw_target_pose[:3]
                rpy_desired = raw_target_pose[3:]

                # === 2. 位置平滑与PD控制 ===
                if smoothed_target_pos is None:
                    smoothed_target_pos = raw_target_pos
                else:
                    #位置平滑滤波器
                    smoothed_target_pos = POSITION_SMOOTHING_ALPHA * raw_target_pos + (
                                1 - POSITION_SMOOTHING_ALPHA) * smoothed_target_pos

                pos = smoothed_target_pos
                target_pose_xyzrpy = np.concatenate([pos, rpy_desired])
                error = np.linalg.norm(tcp_pose[:3] - pos)
                error_derivative = error - last_error
                last_error = error

                if error > FINE_TUNING_THRESHOLD:
                    if pos[2] < Z_SAFE_LIMIT:
                        print(f"[警告] 目标Z轴({pos[2]:.3f})过低，已跳过！")
                        continue
                    else:
                        control_signal = KP * error + KD * error_derivative
                        speed = np.clip(control_signal, MIN_SPEED, MAX_SPEED)
                        print(f"[追踪-PD] E:{error:.3f}, dE:{error_derivative:.3f}, Speed:{speed:.2f}", end='\r')
                        print(f"[移动] 目标 Pose (xyzrpy): pos={pos.round(4)}, rpy={rpy_desired.round(3)}")
                        #input("Press Enter to continue...") # 需要单步调试时取消此行注释
                        try:
                            robot.move_j_p_1(target_pose_xyzrpy.tolist(), k_acc=MOVE_ACCEL, k_vel=speed)
                        except Exception as e:
                            print(f"[错误] 机器人运动失败: {e}")
                else:
                    print(f"[已对准] E:{error:.3f} <= {FINE_TUNING_THRESHOLD}m. 保持静止。          ", end='\r')
                    smoothed_target_pos, last_error = tcp_pose[:3], 0.0

                # === 3. 可视化绘制 ===
                if VISUALIZATION_ENABLED:
                    display_img = draw_target_point(display_img, P_center_in_cam, camera_matrix, dist_coeffs)
                    display_img = draw_camera_center(display_img, camera_matrix)
                    cv2.putText(display_img, f"Error: {error:.4f} m", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2)

            if VISUALIZATION_ENABLED:
                cv2.imshow("Visual Servoing V8 - Refactored", display_img)
                if key == ord('q') or key == 27: break

            elapsed_time = time.time() - start_time
            sleep_time = LOOP_INTERVAL - elapsed_time
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[中止] 程序已手动终止。")
    finally:
        # ✅ 【核心修改】调用 shutdown 来安全关闭所有资源
        if 'robot' in locals():  # 确保robot对象已成功创建
            robot.shutdown()

        if VISUALIZATION_ENABLED: cv2.destroyAllWindows()
        print("\n程序结束。")


if __name__ == '__main__':
    unlocked_pose_visual_servoing_final()


