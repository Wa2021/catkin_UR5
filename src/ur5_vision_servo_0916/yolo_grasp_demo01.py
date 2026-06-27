#简单的yolo抓取实验
import numpy as np
import cv2
import json
import time
import os
from UR_Robot import UR_Robot
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO


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

def get_stable_depth(depth_image, u, v, roi_size=5):
    """
    获取一个点附近ROI区域的稳定深度值，避免噪点。
    深度相机（特别是结构光和ToF相机）生成的深度图并不是完美的。在物体的边缘、反光或透明表面，很容易出现噪点或空洞（深度值为0）。
    如果只取 depth_image[v, u] 这一个点的深度，很可能恰好取到一个噪点或空洞，导致后续所有计算全盘崩溃。
    我们不取一个点，而是取目标点 (u,v) 周围的一个小区域（比如 11x11 的方块，因为 roi_size=5）。
    然后，我们把这个小方块里所有有效的深度值（大于0的）都收集起来。取中值
    中值是把所有数值排序后，取最中间的那个数。它对极端值（噪点）不敏感。
    即使有几个噪点，只要大部分深度值是准确的，中值就能非常稳定地代表这个区域的真实深度。
    因此，在处理可能有噪点的传感器数据时，中值是一种更鲁棒（robust）的统计方法。
    """
    u, v = int(u), int(v)
    
    # 处理可能的3通道深度图像（转换为单通道）
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]  # 取第一个通道
    
    h, w = depth_image.shape
    u_min, u_max = max(0, u - roi_size), min(w - 1, u + roi_size)
    v_min, v_max = max(0, v - roi_size), min(h - 1, v + roi_size)
    # 注意切片右端应 +1
    roi = depth_image[v_min:v_max + 1, u_min:u_max + 1]
    valid_depths = roi[roi > 0]
    if valid_depths.size == 0:
        return 0.0
    return float(np.median(valid_depths)) # 使用中值对异常值更鲁棒



# =============================================================================
#  2. 主功能函数
# =============================================================================

def yolo_grasping_with_debug_steps():
    """
    结合YOLO进行视觉抓取，每一步移动前都需要用户确认。
    """
    # --- 1. 参数配置 ---
    ROBOT_IP = "192.168.0.1"
    CALIB_FILE = 'calibration_result1_4.json'
    YOLO_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/yolov8n.pt'))  # YOLOv8 Nano 模型
    TARGET_CLASS_NAME = 'bottle'  # 你想要抓取的目标类别，例如：'bottle', 'cup', 'cell phone'
    GRASP_CONFIDENCE_THRESHOLD = 0.5  # YOLO检测的置信度阈值

    # --- 2. 初始化 ---
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result(CALIB_FILE)

    print("[YOLO] 正在加载模型...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"[错误] 加载YOLO模型失败: {e}")
        return

    try:
        robot = UR_Robot(ROBOT_IP)
    except Exception as e:
        print(f"[错误] 初始化机器人失败: {e}")
        return

    # --- 3. 移动到初始观察位置 ---
    # 定义一个安全的、视野开阔的观察姿态，格式为 UR 原生 [x, y, z, rx, ry, rz]
    observe_pose_rotvec = np.array([-0.478, -0.0678, 0.336, 2.222, -2.22, -0.140])
    print("\n[步骤 0] 移动到初始观察位置...")
    print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(observe_pose_rotvec, 4)}")
    input("  请按 Enter 键继续...")
    robot.move_j_pose_rotvec(observe_pose_rotvec, k_acc=0.5, k_vel=0.5)
    print("  ✅ 已到达观察位置。")

    try:
        print("\n" + "=" * 50)
        print("开始自动检测目标并执行抓取...")
        print("程序将自动检测目标，检测到后移动到目标上方5cm")
        print("按 Ctrl+C 可随时退出程序")
        print("=" * 50)
        
        detection_count = 0
        max_detections = 10  # 连续检测10帧确保目标稳定
        target_info = None  # 初始化target_info

        while True:
            # --- 4. 自动检测目标 ---
            color_image, depth_image = robot.get_camera_data()
            if color_image is None: 
                continue

            # YOLO 推理
            results = model(color_image, verbose=False)
            display_image = results[0].plot()  # 获取绘制了所有框的图像

            # 寻找置信度最高的目标
            best_box = None
            highest_conf = 0
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if model.names[int(box.cls[0])] == TARGET_CLASS_NAME and conf > GRASP_CONFIDENCE_THRESHOLD:
                    if conf > highest_conf:
                        highest_conf = conf
                        best_box = box.xyxy[0].cpu().numpy()

            if best_box is not None:
                # 计算中心点
                u = (best_box[0] + best_box[2]) / 2
                v = (best_box[1] + best_box[3]) / 2
                # 绘制一个醒目的目标中心点
                cv2.circle(display_image, (int(u), int(v)), 8, (0, 0, 255), -1)
                cv2.circle(display_image, (int(u), int(v)), 10, (255, 255, 255), 2)
                
                # 检测深度
                depth = get_stable_depth(depth_image, u, v)
                if depth > 0:
                    detection_count += 1
                    print(f"[检测中] '{TARGET_CLASS_NAME}' (置信度: {highest_conf:.2f}), 中心:({int(u)},{int(v)}), 深度:{depth:.3f}m [{detection_count}/{max_detections}]")
                    
                    if detection_count >= max_detections:
                        target_info = {'u': u, 'v': v, 'depth': depth}
                        print(f"\n[目标锁定] '{TARGET_CLASS_NAME}' 检测稳定，开始执行抓取流程...")
                        cv2.destroyAllWindows()
                        break
                else:
                    detection_count = 0  # 深度无效，重置计数
                    print("[警告] 目标中心深度无效，重新检测...")
            else:
                detection_count = 0  # 未检测到目标，重置计数
                print("未检测到目标，继续搜索...", end='\r')

            # 显示检测画面
            cv2.imshow("YOLO Grasping - Auto Detection", display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 按q或ESC退出
                raise KeyboardInterrupt

        # --- 5. 计算目标世界坐标 ---
        # a. 像素坐标 -> 相机坐标 (反投影)
        u, v, depth = target_info['u'], target_info['v'], target_info['depth']
        fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        P_cam = np.array([x_cam, y_cam, z_cam, 1.0]).reshape(4, 1)

        # b. 相机坐标 -> 机器人基坐标
        tcp_pose = robot.get_current_tcp_pose()
        T_tool2base = np.eye(4)
        R_tool2base = R.from_rotvec(tcp_pose[3:]).as_matrix()
        T_tool2base[:3, :3] = R_tool2base
        T_tool2base[:3, 3] = tcp_pose[:3]
        T_cam2base = T_tool2base @ T_cam2tool
        P_base = (T_cam2base @ P_cam).flatten()[:3]

        target_xyz = P_base
        print(f"[计算完成] 目标在机器人基坐标系下的位置: {np.round(target_xyz, 4)}")

        # --- 6. 执行完整抓取流程 ---
        # 保持初始的俯视姿态进行抓取
        grasp_rotvec = observe_pose_rotvec[3:]

        # a. 移动到目标点正上方 5cm (预抓取位置)
        approach_xyz = target_xyz.copy()
        approach_xyz[2] += 0.05
        approach_pose = np.concatenate([approach_xyz, grasp_rotvec])

        print("\n[步骤 1] 移动到目标点上方5cm...")
        print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(approach_pose, 4)}")
        input("  请按 Enter 键继续...")
        robot.move_j_pose_rotvec(approach_pose.tolist(), k_acc=0.5, k_vel=0.5)
        print("  ✅ 已到达预抓取位置。")

        # b. 垂直下降到目标点 (抓取位置)
        grasp_pose = np.concatenate([target_xyz, grasp_rotvec])
        print("\n[步骤 2] 垂直下降到抓取位置...")
        print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(grasp_pose, 4)}")
        input("  请按 Enter 键继续...")
        robot.move_j_pose_rotvec(grasp_pose.tolist(), k_acc=0.2, k_vel=0.2)
        print("  ✅ 已到达抓取位置。")

        # c. 执行抓取
        print("\n[步骤 3] 关闭夹爪...")
        input("  请按 Enter 键继续...")
        robot.close_gripper()
        print("  ✅ 已关闭夹爪。")
        time.sleep(1)

        # d. 垂直上升回到预抓取位置
        print("\n[步骤 4] 垂直上升...")
        print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(approach_pose, 4)}")
        input("  请按 Enter 键继续...")
        robot.move_j_pose_rotvec(approach_pose.tolist(), k_acc=0.5, k_vel=0.5)
        print("  ✅ 已垂直上升。")

        # e. 松开夹爪
        input("[步骤 5] 即将松开夹爪，按 Enter 继续...")
        robot.open_gripper()
        print("✅ 已松开夹爪")

        # f. 回到初始观察位置
        print("\n[步骤 6] 回到初始观察位置...")
        print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(observe_pose_rotvec, 4)}")
        input("  请按 Enter 键继续...")
        robot.move_j_pose_rotvec(observe_pose_rotvec, k_acc=0.5, k_vel=0.5)
        print("  ✅ 已回到初始位置。抓取流程结束。")

    except KeyboardInterrupt:
        print("\n[中止] 程序已手动终止。")
    finally:
        cv2.destroyAllWindows()
        print("\n程序结束。")


if __name__ == '__main__':
    yolo_grasping_with_debug_steps()


