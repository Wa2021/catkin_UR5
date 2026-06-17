# 描述: 这是一个独立的工具，用于可视化相机视角下的棋盘格三维坐标轴。
#      它不依赖于机器人，只需要一个连接到电脑的摄像头即可。
#      主要目的是帮助理解 solvePnP 函数是如何建立坐标系的。

import numpy as np
import cv2
import json


def load_camera_parameters(path=r'/home/xsh/catkin_UR5/src/ur5_vision_servo_0916/calibration_result1_4.json'):
    """
    加载相机标定结果。
    这个函数只加载相机内参和畸变系数，因为我们不需要手眼关系。
    """
    try:
        with open(path, 'r') as f:
            calib = json.load(f)
        camera_matrix = np.array(calib['camera_matrix'])
        dist_coeffs = np.array(calib['dist_coeffs'])
        print(f"[成功] 已从 {path} 加载相机参数。")
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print(f"[错误] 找不到相机参数文件: {path}")
        print("[提示] 请确保 'calibration_result1_4.json' 文件与此脚本在同一目录下。")
        return None, None


def find_chessboard_pose(image, camera_matrix, dist_coeffs, pattern_size, square_size):
    """
    在图像中寻找棋盘格，并返回其旋转向量(rvec)和平移向量(tvec)。
    这是位姿解算的核心步骤。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None, None

    # 亚像素优化，让角点定位更精确
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 构造棋盘格的三维世界坐标点
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    # 使用 solvePnP 计算位姿
    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not success:
        return None, None

    return rvec, tvec


def draw_axes(image, rvec, tvec, camera_matrix, dist_coeffs, axis_length=0.05):
    """
    在图像中根据计算出的位姿(rvec, tvec)画出三维坐标轴。
    """
    # 定义三维坐标轴的端点
    axis_points = np.float32([
        [0, 0, 0],  # 原点
        [axis_length, 0, 0],  # X轴端点
        [0, axis_length, 0],  # Y轴端点
        [0, 0, axis_length]  # Z轴端点
    ]).reshape(-1, 3)

    # 将三维点投影到二维图像平面上
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int).reshape(-1, 2)

    origin = tuple(imgpts[0])
    # 绘制坐标轴线段
    cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X轴 - 红色
    cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y轴 - 绿色
    cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z轴 - 蓝色
    # 在原点处画一个黄色的圆点
    cv2.circle(image, origin, 5, (0, 255, 255), -1)

    return image


def main():
    """主函数，负责循环处理视频流"""

    # --- 参数定义 ---
    PATTERN_SIZE = (8, 5)
    SQUARE_SIZE = 0.027  # 单位：米

    # 加载相机参数
    camera_matrix, dist_coeffs = load_camera_parameters()
    if camera_matrix is None:
        return

    # 从相机内参中获取相机中心点(cx, cy)
    camera_center_x = int(camera_matrix[0, 2])
    camera_center_y = int(camera_matrix[1, 2])

    # --- 初始化摄像头 ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头。")
        return

    print("\n[启动] 坐标轴可视化程序已开始。")
    print("[说明] 将摄像头对准棋盘格，即可看到三维坐标轴。")
    print("[说明] 屏幕中央的青色十字准星标记了相机的光学中心。")
    print("[操作] 按 'q' 键或 'Esc' 键退出程序。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[警告] 无法读取到摄像头图像帧。")
            break

        # 寻找棋盘格并计算位姿
        rvec, tvec = find_chessboard_pose(frame, camera_matrix, dist_coeffs, PATTERN_SIZE, SQUARE_SIZE)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- 绘图步骤 1: 绘制棋盘格坐标轴 (如果找到的话) ---
        if rvec is not None and tvec is not None:
            # 如果找到了，就绘制坐标轴
            frame = draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs)
            cv2.putText(frame, 'X: Red, Y: Green, Z: Blue', (10, 30), font, 0.7, (0, 255, 255), 2)
            distance = tvec[2][0]
            cv2.putText(frame, f'Distance: {distance:.3f} m', (10, 60), font, 0.7, (0, 255, 255), 2)
        else:
            # 如果没找到，就显示提示信息
            cv2.putText(frame, 'Chessboard Not Found...', (10, 30), font, 0.8, (0, 0, 255), 2)

        # --- 绘图步骤 2: 绘制相机中心十字准星 (每帧都绘制) ---
        crosshair_color = (255, 255, 0)  # 青色 (Cyan)
        crosshair_size = 15
        # 绘制水平线
        cv2.line(frame, (camera_center_x - crosshair_size, camera_center_y),
                 (camera_center_x + crosshair_size, camera_center_y), crosshair_color, 2)
        # 绘制垂直线
        cv2.line(frame, (camera_center_x, camera_center_y - crosshair_size),
                 (camera_center_x, camera_center_y + crosshair_size), crosshair_color, 2)

        # 显示最终的图像
        cv2.imshow("相机坐标轴可视化 (按q退出)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # --- 释放资源 ---
    cap.release()
    cv2.destroyAllWindows()
    print("[关闭] 程序已退出。")


if __name__ == '__main__':
    main()