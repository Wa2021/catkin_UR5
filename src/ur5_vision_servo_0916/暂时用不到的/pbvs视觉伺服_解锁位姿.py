"""
pbvs视觉伺服4是沿着工具的z轴向后，但是工具的z轴和相机的z轴终究是不平行的，所以用手眼矩阵变换一下
"""
import numpy as np
import cv2
import json
import time
from UR_Robot import UR_Robot
from scipy.spatial.transform import Rotation as R


# =============================================================================
#  1. 辅助函数
# =============================================================================

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
    """V4 最终版解锁位姿视觉伺服"""
    # --- 参数配置 ---
    ROBOT_IP = "192.168.0.1"
    CALIB_FILE = '../calibration_result1_4.json'
    PATTERN_SIZE = (8, 5)  #棋盘格形状
    SQUARE_SIZE = 0.027 #棋盘格大小
    DESIRED_DISTANCE = -0.45# 注意：此距离为负值，表示期望工具Z轴正方向远离物体
    THRESHOLD = 0.005 #误差阈值
    Z_SAFE_LIMIT = 0.05

    # --- 初始化 ---
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result(CALIB_FILE)
    try:
        robot = UR_Robot(ROBOT_IP)
    except Exception as e:
        print(f"[错误] 初始化机器人失败: {e}")
        return

    print("\n[启动] 伺服已开始。")
    print("[提示] 请先将机器人移动到期望的俯视姿态，然后按's'键锁定该姿态为基准。")

    R_correction = None  # 用于存储从物体姿态到工具姿态的修正旋转

    try:
        while True:
            color_img, _ = robot.get_camera_data()
            if color_img is None:
                time.sleep(0.5)
                continue

            rvec, tvec, P_center_in_obj = solve_pnp_pose(color_img, camera_matrix, dist_coeffs, PATTERN_SIZE,
                                                         SQUARE_SIZE)

            display_img = color_img.copy()
            # 始终绘制相机中心
            display_img = draw_camera_center(display_img, camera_matrix)
            key = cv2.waitKey(1) & 0xFF

            if rvec is None:
                cv2.putText(display_img, "Chessboard Not Found", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            2)
            else:
                # --- 可视化 ---
                display_img = draw_axes(display_img, rvec, tvec, camera_matrix, dist_coeffs)

                #得到标定板到相机的变化矩阵
                T_obj2cam = np.eye(4)
                R_obj2cam, _ = cv2.Rodrigues(rvec)
                T_obj2cam[:3, :3] = R_obj2cam
                T_obj2cam[:3, 3] = tvec.flatten()

                # 将目标中心点转换到相机坐标系下，用于绘制
                P_center_in_cam = (T_obj2cam @ np.append(P_center_in_obj, 1))[:3]
                display_img = draw_target_point(display_img, P_center_in_cam, camera_matrix, dist_coeffs)

                #根据当前位姿，得到工具到基坐标系的变换矩阵
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
                    # 这行代码的数学意义是: R_correction = R_base2obj @ R_tool2base_current，“工具坐标系”相对于“物体坐标系”的旋转关系。
                    # 为了方便观察，将这个修正矩阵转成RPY角并打印出来
                    rpy_corr = R.from_matrix(R_correction).as_euler('xyz', degrees=False)
                    print(f"\n[成功] 姿态基准已锁定！修正RPY: {np.round(rpy_corr, 3)}\n")

                R_tool_desired = R_obj2base @ R_correction#工具坐标系相对于基坐标系的旋转位姿
                #转换成rpy格式
                rpy_desired = R.from_matrix(R_tool_desired).as_euler('xyz', degrees=False)
                # 根据期望的工具姿态，计算出期望的相机姿态
                R_cam_desired_in_base = R_tool_desired @ T_cam2tool[:3, :3]

                #从期望的相机姿态中，提取出相机Z轴在基坐标系下的方向（指明了z轴方向）
                z_axis_cam_desired_in_base = R_cam_desired_in_base[:, 2]

                #标定板中心在基坐标系下的位置
                P_center_in_base = (T_cam2base @ np.append(P_center_in_cam, 1))[:3]

                #相机的目标位姿，是标定板在基坐标系的位姿加上在z轴方向上的偏移量
                P_cam_desired_in_base = P_center_in_base + z_axis_cam_desired_in_base * DESIRED_DISTANCE

                #从相机期望位置转换成工具期望位置
                t_cam2tool_in_tool = T_cam2tool[:3, 3]#相机到工具的平移向量（在工具坐标系下）也即手眼标定的固定偏移量
                P_tool_desired_in_base = P_cam_desired_in_base - R_tool_desired @ t_cam2tool_in_tool
                #手眼标定的偏移量乘想要的工具姿态，相当于将偏移量旋转到基坐标系下
                #直观理解: 相机相对于法兰盘，永远是在Z轴负方向145mm。
                # 把法兰盘倾斜45度时，这个-145mm的偏移量在世界空间里也跟着倾斜了45度。`R_tool_desired` 就是用来做这个“旋转”的。
                #（相机的位置 = 工具的位置 + 在工具姿态下，从工具到相机的偏移向量）

                # =================================================================

                pos = P_tool_desired_in_base
                target_pose_xyzrpy = np.concatenate([pos, rpy_desired])

                error = np.linalg.norm(tcp_pose[:3] - pos)
                cv2.putText(display_img, f"Error: {error:.4f} m", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)

                if error > THRESHOLD:
                    if pos[2] < Z_SAFE_LIMIT:
                        print(f"[警告] 目标Z轴({pos[2]:.3f})过低，已跳过！")
                        continue
                    else:
                        print(f"[移动] 目标 Pose (xyzrpy): pos={pos.round(4)}, rpy={rpy_desired.round(3)}")
                        input("Press Enter to continue...") # 需要单步调试时取消此行注释
                        try:
                            robot.move_j_p(target_pose_xyzrpy.tolist(), k_acc=0.1, k_vel=0.1)
                        except Exception as move_error:
                            print(f"[错误] 机器人运动失败: {move_error}")
                else:
                    pass

            cv2.imshow("Visual Servoing V4 - Final", display_img)
            if key == ord('q') or key == 27: break

    except KeyboardInterrupt:
        print("\n[中止] 程序已手动终止。")
    finally:
        cv2.destroyAllWindows()


# =============================================================================
#  3. 程序入口
# =============================================================================
if __name__ == '__main__':
    unlocked_pose_visual_servoing_final()