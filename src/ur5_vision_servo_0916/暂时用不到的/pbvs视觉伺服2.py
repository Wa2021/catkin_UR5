import numpy as np
import cv2
import json
import time
from UR_Robot import UR_Robot
from scipy.spatial.transform import Rotation as R


# --- 核心变换逻辑 ---
# 目标: 计算出TCP应该移动到的目标位姿 T_tool_target2base
#
# 为了达到目标，我们需要以下变换链:
#
# 1. T_obj_center2base = T_tool_current2base @ T_cam2tool @ T_obj_center2cam
#    (计算出物体中心在基座坐标系下的当前位姿)
#    其中, T_obj_center2cam = T_obj_corner2cam @ np.linalg.inv(T_obj_center2obj_corner)
#
# 2. T_cam_desired2base = T_obj_center2base @ T_cam_desired2obj_center
#    (根据物体中心的位姿，计算出相机理想观测位姿在基座坐标系下的表示)
#
# 3. T_tool_target2base = T_cam_desired2base @ np.linalg.inv(T_cam2tool)
#    (根据相机的理想位姿，反推出TCP应该移动到的目标位姿)
#    其中, np.linalg.inv(T_cam2tool) 就是 T_tool2cam


def load_hand_eye_result(path='calibration_result1_4.json'):
    """加载手眼标定结果"""
    with open(path, 'r') as f:
        calib = json.load(f)
    camera_matrix = np.array(calib['camera_matrix'])
    dist_coeffs = np.array(calib['dist_coeffs'])
    # T_cam2tool: 从相机坐标系到工具坐标系的变换矩阵 (正确命名)
    T_cam2tool = np.array(calib['hand_eye_matrix'])
    return camera_matrix, dist_coeffs, T_cam2tool


def solve_pnp_pose(color_img, camera_matrix, dist_coeffs, pattern_size=(8, 5), square_size=0.027):
    """
    通过棋盘格角点解算PnP，获取物体角点到相机的位姿变换矩阵 T_obj_corner2cam
    """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not success:
        return None

    # 将旋转向量和平移向量转换为4x4的变换矩阵 T_obj_corner2cam (正确命名)
    R_obj_corner2cam, _ = cv2.Rodrigues(rvec)
    T_obj_corner2cam = np.eye(4)
    T_obj_corner2cam[:3, :3] = R_obj_corner2cam
    T_obj_corner2cam[:3, 3] = tvec.flatten()

    return T_obj_corner2cam


def draw_camera_center_crosshair(image):
    """在图像中心绘制一个十字准星，以表示相机光心"""
    h, w, _ = image.shape
    center_x, center_y = w // 2, h // 2
    crosshair_size = 15
    color = (0, 255, 0)
    thickness = 2
    cv2.line(image, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), color, thickness)
    cv2.line(image, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), color, thickness)
    cv2.circle(image, (center_x, center_y), 5, color, -1)
    return image


def pbvs_control():
    # --- 初始化 ---
    robot = UR_Robot("192.168.0.1")
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result()

    # --- 参数设置 ---
    DESIRED_DISTANCE = -0.45
    THRESHOLD = 0.005
    TRACKING_INTERVAL = 0.2
    PATTERN_SIZE = (8, 5)
    SQUARE_SIZE = 0.027

    print("[启动] 视觉伺服追踪已开始，按 'q' 或 'Esc' 停止...")

    # --- 锁定初始姿态 ---
    try:
        # 获取当前位姿（包含旋转向量）
        tcp_pose = robot.get_current_tcp_pose()
        # 将旋转向量转换为欧拉角 (RPY)
        # 注意：这里假设你的 UR_Robot 类有 rotvec_to_rpy 方法
        _, _, _, roll, pitch, yaw = robot.rotvec_to_rpy(*tcp_pose)
        # 锁定这组欧拉角
        locked_rpy = np.array([roll, pitch, yaw])
        print(f"[成功] 姿态锁定为 RPY (弧度): {np.round(locked_rpy, 3)}，将持续使用此姿态。")
    except Exception as e:
        print(f"[错误] 获取并锁定初始姿态失败: {e}")
        return

    # --- 计算从角点到中心的变换 ---
    center_x = (PATTERN_SIZE[0] - 1) * SQUARE_SIZE / 2.0
    center_y = (PATTERN_SIZE[1] - 1) * SQUARE_SIZE / 2.0

    # T_obj_center2obj_corner: 从物体中心坐标系到物体角点坐标系的变换 (正确命名)
    T_obj_center2obj_corner = np.eye(4)
    T_obj_center2obj_corner[0, 3] = center_x
    T_obj_center2obj_corner[1, 3] = center_y
    print(f"[信息] 棋盘格模型: X轴沿8个角点边, Y轴沿5个角点边。")
    print(f"[信息] 棋盘格中心相对于角点的偏移: x={center_x:.4f}, y={center_y:.4f}")

    first_move_check = True

    try:
        while True:
            # --- 1. 获取图像和位姿 ---
            color_img, _ = robot.get_camera_data()
            if color_img is None: time.sleep(0.5); continue

            display_img = color_img.copy()
            display_img = draw_camera_center_crosshair(display_img)

            # T_obj_corner2cam: 从物体角点坐标系到相机坐标系的变换 (正确命名)
            T_obj_corner2cam = solve_pnp_pose(color_img, camera_matrix, dist_coeffs,
                                              pattern_size=PATTERN_SIZE, square_size=SQUARE_SIZE)

            if T_obj_corner2cam is None:
                print("[警告] 未检测到棋盘格，等待中...")
                cv2.imshow("Camera View", display_img)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'): break
                time.sleep(0.5)
                continue

            # --- 2. 核心逻辑: 坐标变换链 ---

            # T_obj_center2cam: 从物体中心坐标系到相机坐标系的变换 (正确命名)
            T_obj_center2cam = T_obj_corner2cam @ T_obj_center2obj_corner

            # T_tool_current2base: 工具当前位姿在基座坐标系下的表示 (修改命名)
            current_tool_pose_list = robot.get_current_tcp_pose()
            T_tool_current2base = np.eye(4)
            T_tool_current2base[:3, :3] = R.from_rotvec(current_tool_pose_list[3:]).as_matrix()
            T_tool_current2base[:3, 3] = current_tool_pose_list[:3]

            # T_cam_desired2obj_center: 理想相机位姿在物体中心坐标系下的表示 (正确命名)
            T_cam_desired2obj_center = np.eye(4)
            T_cam_desired2obj_center[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()
            T_cam_desired2obj_center[2, 3] = DESIRED_DISTANCE

            # (变换链1) T_obj_center2base: 物体中心在基座坐标系下的表示 (正确命名)
            T_obj_center2base = T_tool_current2base @ T_cam2tool @ T_obj_center2cam

            # (变换链2) T_cam_desired2base: 理想相机位姿在基座坐标系下的表示 (正确命名)
            T_cam_desired2base = T_obj_center2base @ T_cam_desired2obj_center

            # (变换链3) T_tool_target2base: 工具目标位姿在基座坐标系下的表示 (正确命名)
            T_tool_target2base = T_cam_desired2base @ np.linalg.inv(T_cam2tool)

            # --- 3. 提取目标位姿并下达指令 ---
            target_pos = T_tool_target2base[:3, 3]
            target_pose_for_robot = np.concatenate([target_pos, locked_rpy])

            current_tcp_pos = T_tool_current2base[:3, 3]  # (修改命名)
            error = np.linalg.norm(current_tcp_pos - target_pos)
            print(f"[追踪] 距离目标误差：{error:.4f} m")

            if error > THRESHOLD:
                if target_pos[2] < 0.05:
                    print(f"[警告] 目标Z轴({target_pos[2]:.3f})过低，可能发生碰撞，跳过本次移动！")
                    continue
                else:
                    print(f"[移动] 目标TCP位置: pos={target_pos.round(4)}")
                    if first_move_check:
                        input("--- 移动前检查 --- 请确认目标位置合理。按回车键继续...")
                        first_move_check = False
                    try:
                        robot.move_j_p(target_pose_for_robot.tolist(), k_acc=0.1, k_vel=0.1)
                    except Exception as move_error:
                        print(f"[错误] 机器人运动失败：{move_error}")
            else:
                print("[完成] 已对准目标，伺服暂停。")
                first_move_check = True

            cv2.imshow("Camera View", display_img)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'): break
            time.sleep(TRACKING_INTERVAL)

    except KeyboardInterrupt:
        print("\n[中止] 视觉伺服已手动终止。")
    finally:
        cv2.destroyAllWindows()
        print("[关闭] 程序已退出。")


if __name__ == '__main__':
    pbvs_control()