"""
尝试使用pd控制
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


def draw_axes(image, rvec, tvec, camera_matrix, dist_coeffs, axis_length=0.05):
    """在图像中画出三维坐标轴（代表棋盘格坐标系）"""
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int).reshape(-1, 2)
    origin = tuple(imgpts[0])
    cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X: Red
    cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y: Green
    cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z: Blue
    cv2.putText(image, "Obj Origin", (origin[0] - 30, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return image


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


# =============================================================================
#  2. 主功能函数
# =============================================================================

def unlocked_pose_visual_servoing_final():
    """
    V6 最终版：带PD控制的混合模式视觉伺服
    - 结合了连续追踪与稳定精定位
    - 使用了位置滤波、PD控制器来计算速度
    """
    # --- 参数配置 ---
    ROBOT_IP = "192.168.0.1"
    CALIB_FILE = '../calibration_result1_4.json'
    PATTERN_SIZE = (8, 5)
    SQUARE_SIZE = 0.027
    DESIRED_DISTANCE = -0.45
    Z_SAFE_LIMIT = 0.05

    # === 新的、更精细的控制参数 ===
    # 1. 阈值 (米)
    FINE_TUNING_THRESHOLD = 0.01  # 1cm: 进入精调模式的阈值

    # 2. 运动参数
    MAX_SPEED = 0.5  # 最大速度
    MIN_SPEED = 0.01  # 最低速度
    MOVE_ACCEL = 0.4  # 统一的加速度

    # 3. === PD 控制器增益 ===
    # Kp: 比例增益，决定了对当前误差的反应速度
    # Kd: 微分增益，决定了对误差变化趋势的“阻尼”或“刹车”力度
    # 这两个值是调试的核心，需要根据实际效果调整
    KP = 2.5  # 比例增益
    KD = 2  # 微分增益 (关键！)

    # 4. 位置平滑滤波器参数
    POSITION_SMOOTHING_ALPHA = 0.7  # 可以适当调高，让响应更快

    # 5. 循环频率
    LOOP_INTERVAL = 0.1  # 10Hz

    # --- 初始化 ---
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result(CALIB_FILE)
    try:
        robot = UR_Robot(ROBOT_IP)
    except Exception as e:
        print(f"[错误] 初始化机器人失败: {e}")
        return

    print("\n[启动] V6 PD混合模式伺服已开始。")
    print("[提示] 按 's' 锁定姿态基准, 按 'q' 退出。")

    R_correction = None
    smoothed_target_pos = None

    # === 为PD控制器增加初始化变量 ===
    last_error = 0.0  # 用于存储上一次的误差

    try:
        while True:
            # ... (循环开始和获取图像部分不变) ...
            start_time = time.time()
            color_img, _ = robot.get_camera_data()
            if color_img is None:
                time.sleep(0.1)
                continue

            rvec, tvec, P_center_in_obj = solve_pnp_pose(color_img, camera_matrix, dist_coeffs, PATTERN_SIZE,
                                                         SQUARE_SIZE)
            display_img = color_img.copy()
            display_img = draw_camera_center(display_img, camera_matrix)
            key = cv2.waitKey(1) & 0xFF

            if rvec is None:
                cv2.putText(display_img, "Chessboard Not Found", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            2)
                smoothed_target_pos = None
                last_error = 0.0  # 找不到目标时重置误差历史
            else:
                # --- 坐标变换和目标计算部分不变 ---
                # ... (此处代码与你原版完全相同) ...
                T_obj2cam = np.eye(4)
                R_obj2cam, _ = cv2.Rodrigues(rvec)
                T_obj2cam[:3, :3] = R_obj2cam
                T_obj2cam[:3, 3] = tvec.flatten()
                P_center_in_cam = (T_obj2cam @ np.append(P_center_in_obj, 1))[:3]
                display_img = draw_target_point(display_img, P_center_in_cam, camera_matrix, dist_coeffs)
                tcp_pose = robot.get_current_tcp_pose()
                T_tool2base = np.eye(4)
                R_tool2base_current = R.from_rotvec(tcp_pose[3:]).as_matrix()
                T_tool2base[:3, :3] = R_tool2base_current
                T_tool2base[:3, 3] = tcp_pose[:3]
                T_cam2base = T_tool2base @ T_cam2tool
                T_obj2base = T_cam2base @ T_obj2cam
                R_obj2base = T_obj2base[:3, :3]
                if key == ord('s') or R_correction is None:
                    R_correction = R_obj2base.T @ R_tool2base_current
                    rpy_corr = R.from_matrix(R_correction).as_euler('xyz', degrees=False)
                    print(f"\n[成功] 姿态基准已锁定！修正RPY: {np.round(rpy_corr, 3)}\n")
                    smoothed_target_pos = None
                    last_error = 0.0  # 锁定新基准时重置误差历史
                R_tool_desired = R_obj2base @ R_correction
                rpy_desired = R.from_matrix(R_tool_desired).as_euler('xyz', degrees=False)
                R_cam_desired_in_base = R_tool_desired @ T_cam2tool[:3, :3]
                z_axis_cam_desired_in_base = R_cam_desired_in_base[:, 2]
                P_center_in_base = (T_cam2base @ np.append(P_center_in_cam, 1))[:3]
                P_cam_desired_in_base = P_center_in_base + z_axis_cam_desired_in_base * DESIRED_DISTANCE
                t_cam2tool_in_tool = T_cam2tool[:3, 3]
                raw_target_pos = P_cam_desired_in_base - R_tool_desired @ t_cam2tool_in_tool

                # --- 1. 位置平滑滤波 (不变) ---
                if smoothed_target_pos is None:
                    smoothed_target_pos = raw_target_pos
                else:
                    smoothed_target_pos = POSITION_SMOOTHING_ALPHA * raw_target_pos + \
                                          (1 - POSITION_SMOOTHING_ALPHA) * smoothed_target_pos
                pos = smoothed_target_pos
                target_pose_xyzrpy = np.concatenate([pos, rpy_desired])

                # --- 2. 混合模式与PD控制 ---
                error = np.linalg.norm(tcp_pose[:3] - pos)
                cv2.putText(display_img, f"Error: {error:.4f} m", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)

                # === PD控制器的核心计算 ===
                # D项: 计算误差的变化率 (当前误差 - 上一次误差)
                error_derivative = error - last_error
                # 更新 last_error 以备下一次循环使用
                last_error = error

                if error > FINE_TUNING_THRESHOLD:
                    # 模式A：追踪模式 (误差 > 1cm)
                    if pos[2] < Z_SAFE_LIMIT:
                        print(f"[警告] 目标Z轴({pos[2]:.3f})过低，已跳过！")
                        continue
                    else:
                        # P项：与当前误差成正比
                        p_term = KP * error
                        # D项：与误差变化率成正比
                        d_term = KD * error_derivative

                        # 最终的控制信号 (用于决定速度)
                        control_signal = p_term + d_term

                        # 使用 np.clip 将计算出的速度限制在安全和有效的范围内
                        speed = np.clip(control_signal, MIN_SPEED, MAX_SPEED)

                        print(f"[追踪-PD] E:{error:.3f}, dE:{error_derivative:.3f}, Speed:{speed:.2f}")
                        try:
                            # 使用非阻塞的move_j_p
                            robot.move_j_p_1(target_pose_xyzrpy.tolist(), k_acc=MOVE_ACCEL, k_vel=speed)
                        except Exception as move_error:
                            print(f"[错误] 机器人运动失败: {move_error}")
                else:
                    # 模式B：精定位完成 (误差 <= 10mm)
                    print(f"[已对准] E:{error:.3f} <= {FINE_TUNING_THRESHOLD}m. 保持静止。")
                    smoothed_target_pos = tcp_pose[:3]
                    # 重置last_error，避免在下次启动时，dE值过大
                    last_error = 0.0

            # ... (imshow 和循环等待部分不变) ...
            cv2.imshow("Visual Servoing V6 - PD Hybrid", display_img)
            if key == ord('q') or key == 27: break
            elapsed_time = time.time() - start_time
            sleep_time = LOOP_INTERVAL - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[中止] 程序已手动终止。")
    finally:
        cv2.destroyAllWindows()



if __name__ == '__main__':
    unlocked_pose_visual_servoing_final()