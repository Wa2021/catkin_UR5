#基于第一个视觉伺服代码改的，成功了，不要再改了
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
        return None, None, None, None  # 返回None时也返回4个值

    # 亚像素优化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 构造目标点
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not success:
        return None, None, None, None

    # 计算标定板中心点在物体坐标系下的坐标
    center_x = (pattern_size[0] - 1) / 2 * square_size
    center_y = (pattern_size[1] - 1) / 2 * square_size
    P_center_in_obj = np.array([center_x, center_y, 0])

    return rvec, tvec, corners, P_center_in_obj


# === 绘制标定板中心点 ===
def draw_target_point(image, P_center_in_cam, camera_matrix, dist_coeffs):
    """将目标点（标定板中心，在相机坐标系下）投影到图像上并绘制出来"""
    # 目标点的3D坐标
    target_point_3d = P_center_in_cam.reshape(1, 1, 3)

    # 使用projectPoints将3D点投影到2D图像平面
    target_point_2d, _ = cv2.projectPoints(target_point_3d,
                                           np.zeros(3), np.zeros(3),
                                           camera_matrix, dist_coeffs)

    # 绘制点和文字
    pt = tuple(target_point_2d[0][0].astype(int))
    cv2.circle(image, pt, 8, (0, 165, 255), -1)  # 橙色的圆点
    cv2.circle(image, pt, 9, (255, 255, 255), 1)  # 白色轮廓
    cv2.putText(image, "Target (Board Center)", (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return image


def draw_camera_center(image, camera_matrix, size=15, color=(255, 0, 0), thickness=2):
    """在图像上绘制相机中心（主点），十字准星"""
    # 从相机内参矩阵获取主点坐标 (cx, cy)
    cx = int(camera_matrix[0, 2])
    cy = int(camera_matrix[1, 2])

    # 绘制水平线
    cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
    # 绘制垂直线
    cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)

    cv2.putText(image, "Cam Center", (cx + 5, cy - size - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image

def draw_axes(image, rvec, tvec, camera_matrix, dist_coeffs, axis_length=0.05):
    """在图像中画出三维坐标轴"""
    z_axis_length = axis_length * 2
    axis_points = np.float32([
        [0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, z_axis_length]
    ]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int).reshape(-1, 2)
    origin = tuple(imgpts[0])
    cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 3)
    cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)
    cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)
    cv2.circle(image, origin, 5, (0, 255, 255), -1)
    return image


def pbvs_control():
    # === 明确定义标定板参数，方便复用 ===
    PATTERN_SIZE = (8, 5)
    SQUARE_SIZE = 0.027

    robot = UR_Robot("192.168.0.1")
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result()

    DESIRED_DISTANCE = 0.45
    THRESHOLD = 0.005
    TRACKING_INTERVAL = 0.2

    print("[启动] PBVS 追踪已开始，按 Ctrl+C 停止...")

    try:
        # 锁定初始姿态，这在很多PBVS任务中是常见做法，可以简化控制
        tcp_pose = robot.get_current_tcp_pose()
        initial_rotvec = tcp_pose[3:]
        R_tool_desired_in_base = R.from_rotvec(initial_rotvec).as_matrix()
        # 将旋转矢量转为RPY，只是为了打印和作为move_j_p的参数
        _, _, _, r, p, y = robot.rotvec_to_rpy(*tcp_pose)
        desired_rpy = np.array([r, p, y])

        print(f"[成功] 姿态锁定为: RPY={np.round(desired_rpy, 3)}，将持续使用此姿态。")
    except Exception as e:
        print(f"[错误] 姿态锁定失败: {e}")
        return

    try:
        while True:
            color_img, _ = robot.get_camera_data()
            # === 从solve_pnp_pose接收4个返回值 ===
            rvec, tvec, corners, P_center_in_obj = solve_pnp_pose(color_img, camera_matrix, dist_coeffs, PATTERN_SIZE,
                                                                  SQUARE_SIZE)
            display_img = color_img.copy()

            if rvec is None:
                print("[警告] 未检测到棋盘格，等待中...")
                cv2.putText(display_img, "Chessboard Not Found", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Camera View", display_img)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'): break
                time.sleep(0.5)
                continue

            # 在图像上绘制棋盘格原点的坐标轴
            display_img = draw_axes(display_img, rvec, tvec, camera_matrix, dist_coeffs)
            display_img = draw_camera_center(display_img, camera_matrix)
            # --- 坐标变换与机器人控制 (这部分是核心修改) ---

            # ===计算目标位姿 ===

            # 1. 构建从物体(标定板)到相机的变换矩阵
            R_obj2cam, _ = cv2.Rodrigues(rvec)
            T_obj2cam = np.eye(4)
            T_obj2cam[:3, :3] = R_obj2cam
            T_obj2cam[:3, 3] = tvec.flatten()

            # 2. 计算标定板中心点在相机坐标系下的位置
            P_center_in_cam = (T_obj2cam @ np.append(P_center_in_obj, 1))[:3]

            # 在图像上绘制标定板中心点（新的目标点）
            display_img = draw_target_point(display_img, P_center_in_cam, camera_matrix, dist_coeffs)

            # 3. 获取当前的工具位姿，并计算出相机姿态，以便得到相机Z轴在基座标系下的方向
            tcp_pose = robot.get_current_tcp_pose()
            T_tool2base = np.eye(4)
            T_tool2base[:3, :3] = R.from_rotvec(tcp_pose[3:]).as_matrix()
            T_tool2base[:3, 3] = tcp_pose[:3]

            # 计算当前相机位姿
            T_cam2base = T_tool2base @ T_cam2tool

            # 4. 计算标定板中心点在基座标系下的位置
            P_center_in_base = (T_cam2base @ np.append(P_center_in_cam, 1))[:3]

            # 5. 计算期望的相机位置
            # 期望的相机Z轴（在基座标系下）应该与当前相机Z轴平行（因为我们锁定了TCP姿态）
            # R_tool_desired_in_base 是我们锁定的姿态
            R_cam_desired_in_base = R_tool_desired_in_base @ T_cam2tool[:3, :3]
            z_axis_cam_in_base = R_cam_desired_in_base[:, 2]  # 相机Z轴向量

            # 期望的相机位置 = 标定板中心位置 - 沿相机Z轴方向后退DESIRED_DISTANCE
            P_cam_desired_in_base = P_center_in_base - DESIRED_DISTANCE * z_axis_cam_in_base

            # 6. 根据期望的相机位置，反推期望的工具TCP位置
            # P_tool = P_cam - R_tool * t_cam2tool
            # 其中 t_cam2tool 是从相机到工具的平移，在工具坐标系下表示
            t_cam2tool_in_tool = T_cam2tool[:3, 3]
            P_tool_desired_in_base = P_cam_desired_in_base - R_tool_desired_in_base @ t_cam2tool_in_tool

            # 7. 组合成最终的目标位姿 [x, y, z, r, p, y]
            pos = P_tool_desired_in_base
            rpy = desired_rpy  # 使用我们锁定的姿态
            target_pose = np.concatenate([pos, rpy])

            # === 修改结束 ===

            current_tcp_pos = robot.get_current_tcp_pose()[:3]
            error = np.linalg.norm(current_tcp_pos - pos)
            print(f"[追踪] 距离目标误差：{error:.4f} m")

            if error > THRESHOLD:
                if pos[2] < 0.05:
                    print(f"[警告] 目标Z轴({pos[2]:.3f})过低，可能发生碰撞，跳过本次移动！")
                else:
                    print(f"[移动] 到 pos={pos.round(4)}, rpy={rpy.round(3)}")
                    input("请按enter键继续") # 调试时可以取消注释
                    try:
                        robot.move_j_p(target_pose.tolist(), k_acc=0.1, k_vel=0.1)
                    except Exception as move_error:
                        print(f"[错误] 运动失败：{move_error}")
            else:
                print("[暂停] 已对准目标，等待下一帧...")

            cv2.imshow("Camera View", display_img)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

            time.sleep(TRACKING_INTERVAL)

    except KeyboardInterrupt:
        print("\n[中止] PBVS追踪已手动终止。")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pbvs_control()