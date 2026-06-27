# 简单的 yolo 抓取实验
# （带 bbox 内方向估计：Canny+Hough 或 PCA 回退）
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
    """
    u, v = int(u), int(v)
    
    # 处理可能的3通道深度图像（转换为单通道）
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]  # 取第一个通道
    
    h, w = depth_image.shape
    u_min, u_max = max(0, u - roi_size), min(w - 1, u + roi_size)
    v_min, v_max = max(0, v - roi_size), min(h - 1, v + roi_size)
    roi = depth_image[v_min:v_max + 1, u_min:u_max + 1]
    valid_depths = roi[roi > 0]
    if valid_depths.size == 0:
        return 0.0
    return float(np.median(valid_depths))  # 使用中值对异常值更鲁棒


def estimate_angle_from_bbox(color_image, bbox, debug_img=None):
    """
    在给定的 bbox 区域内估计目标主轴方向（弧度）。
    先尝试：Canny 边缘 -> 最大轮廓 -> minAreaRect -> 长轴方向
    若失败：回退到 PCA（对前景像素/非零像素做主成分分析）
    返回值：object_angle_rad（弧度），范围约在 [0, pi)
    参数:
      - color_image: 原始 BGR 图像
      - bbox: [x1, y1, x2, y2]（像素坐标）
      - debug_img: 可选的绘图图像，函数会在上面绘制结果用于调试（如果传 None 则不绘制）
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # 边界保护
    h, w = color_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = color_image[y1:y2 + 1, x1:x2 + 1].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # --- 方法1：Canny 边缘 + 找最大轮廓 + minAreaRect ---
    # 先平滑再做 Canny，减少噪声
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        # 选面积最大的轮廓
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) > 20:  # 面积阈值，避免噪点
            rect = cv2.minAreaRect(main_contour)
            # rect = ((cx, cy), (w, h), angle) 
            # OpenCV的angle定义：第一条边（从width对应的边）与x轴的夹角，范围[-90, 0)
            angle = rect[2]
            width, height = rect[1]
            
            print(f"[调试] OpenCV rect: width={width:.1f}, height={height:.1f}, angle={angle:.1f}°")
            print(f"[调试] 长轴判断: {'width' if width >= height else 'height'} 是长轴")
            
            # OpenCV minAreaRect 角度定义详解：
            # - angle 范围: [-90°, 0°)  
            # - 当 width >= height 时：angle 是长边（width对应边）与 x 轴的夹角
            # - 当 width < height 时：angle 是短边（width对应边）与 x 轴的夹角
            # - 我们始终需要得到长轴的角度
            
            if width >= height:
                # width 是长轴，angle 就是长轴角度
                main_axis_angle_deg = angle
            else:
                # height 是长轴，长轴角度 = angle + 90°
                main_axis_angle_deg = angle + 90
            
            # 将角度规范化到 [-90°, 90°) 范围
            # 这样可以避免不必要的大角度旋转
            if main_axis_angle_deg >= 90:
                main_axis_angle_deg -= 180
            elif main_axis_angle_deg < -90:
                main_axis_angle_deg += 180
                
            grasp_angle_rad = np.deg2rad(main_axis_angle_deg)
            
            print(f"[调试] 计算的主轴角度: {main_axis_angle_deg:.1f}°")

            # 可视化：绘制检测到的主轴方向（红色箭头）
            if debug_img is not None:
                box_pts = cv2.boxPoints(rect).astype(np.int32)
                # 将 box_pts 平移回原图坐标
                box_pts[:, 0] += x1
                box_pts[:, 1] += y1
                cv2.drawContours(debug_img, [box_pts], 0, (0, 255, 0), 2)
                center = (int(rect[0][0] + x1), int(rect[0][1] + y1))
                
                # 绘制主轴方向（红色）- 应该沿着长轴
                length = int(max(width, height) / 2)
                dx_main = int(length * np.cos(grasp_angle_rad))
                dy_main = int(length * np.sin(grasp_angle_rad))
                endp_main = (center[0] + dx_main, center[1] + dy_main)
                cv2.arrowedLine(debug_img, center, endp_main, (0, 0, 255), 3, tipLength=0.3)
                
                # 绘制垂直方向（蓝色）- 这是实际的抓取方向
                perp_angle_rad = grasp_angle_rad + np.pi/2
                dx_perp = int(length * np.cos(perp_angle_rad))
                dy_perp = int(length * np.sin(perp_angle_rad))
                endp_perp = (center[0] + dx_perp, center[1] + dy_perp)
                cv2.arrowedLine(debug_img, center, endp_perp, (255, 0, 0), 3, tipLength=0.3)
            return grasp_angle_rad

    # --- 方法2：PCA 回退（对前景像素或强度非零的像素） ---
    # 简单阈值分割（Otsu）得到前景 mask
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(th > 0)
    if xs.size >= 5:
        # PCA: 计算协方差并做特征分解
        pts = np.vstack((xs, ys)).astype(np.float64).T  # (N,2)，注意 ROI 坐标系
        # 中心化
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        # 特征分解
        eigvals, eigvecs = np.linalg.eigh(cov)  # 升序
        principal_vec = eigvecs[:, np.argmax(eigvals)]  # 最大特征值对应的特征向量
        # principal_vec 是 ROI 坐标系中的向量 (vx, vy)
        angle = np.arctan2(principal_vec[1], principal_vec[0])  # 弧度
        
        # 将角度规范化到[-π/2, π/2)范围（即[-90°, 90°)）
        if angle >= np.pi/2:
            angle -= np.pi
        elif angle < -np.pi/2:
            angle += np.pi
            
        grasp_angle_rad = angle

        # 可视化到 debug_img（若提供）
        if debug_img is not None:
            center = (int(mean[0] + x1), int(mean[1] + y1))
            length = int(max(x2 - x1, y2 - y1) / 3)
            
            # 绘制主轴方向（红色）
            dx_main = int(length * np.cos(grasp_angle_rad))
            dy_main = int(length * np.sin(grasp_angle_rad))
            endp_main = (center[0] + dx_main, center[1] + dy_main)
            cv2.arrowedLine(debug_img, center, endp_main, (0, 0, 255), 2, tipLength=0.3)
            
            # 绘制垂直方向（蓝色）- 这是实际的抓取方向
            perp_angle_rad = grasp_angle_rad + np.pi/2
            dx_perp = int(length * np.cos(perp_angle_rad))
            dy_perp = int(length * np.sin(perp_angle_rad))
            endp_perp = (center[0] + dx_perp, center[1] + dy_perp)
            cv2.arrowedLine(debug_img, center, endp_perp, (255, 0, 0), 2, tipLength=0.3)

        return grasp_angle_rad

    # 都失败则返回 0（默认不旋转）
    return 0.0


# =============================================================================
#  2. 主功能函数（主流程基本保留，插入方向估计）
# =============================================================================

def yolo_grasping_with_debug_steps():
    """
    结合YOLO进行视觉抓取，每一步移动前都需要用户确认。
    在 bbox 内估计物体主轴方向并调整末端 yaw。
    """
    # --- 1. 参数配置 ---
    ROBOT_IP = "192.168.0.1"
    CALIB_FILE = 'calibration_result1_4.json'
    YOLO_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/yolov8n.pt'))  # 使用检测模型（轻量）
    TARGET_CLASS_NAME = 'bottle'
    GRASP_CONFIDENCE_THRESHOLD = 0.3  # 降低置信度门槛，便于检测

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
    observe_pose_rotvec = np.array([-0.478, -0.0678, 0.336, 2.222, -2.22, -0.140])
    print("\n[步骤 0] 移动到初始观察位置...")
    print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(observe_pose_rotvec, 4)}")
    input("  请按 Enter 键继续...")
    robot.move_j_pose_rotvec(observe_pose_rotvec, k_acc=0.5, k_vel=0.5)
    print("  ✅ 已到达观察位置。")

    try:
        print("\n" + "=" * 50)
        print("开始自动检测目标并执行抓取...")
        print("程序将自动检测目标，检测到后执行完整抓取流程")
        print("按 Ctrl+C 可随时退出程序")
        print("=" * 50)
        
        detection_count = 0
        max_detections = 10  # 连续检测10帧确保目标稳定
        target_info = None  # 初始化target_info
        locked = False       # 是否已锁定目标（不立即跳出，保持窗口刷新）
        announced_locked = False  # 锁定提示只打印一次
        step1_prompted = False    # 是否已提示“步骤1按Enter”

        while True:
            # --- 4. 自动检测目标 ---
            color_image, depth_image = robot.get_camera_data()
            if color_image is None:
                continue

            # YOLO 推理
            results = model(color_image, verbose=False)
            
            # 先使用YOLO自带的绘制功能显示所有检测框
            display_image = results[0].plot()  # 这会显示所有检测到的物体
            
            # 然后找出我们的目标类别
            best_box = None
            highest_conf = 0
            object_angle_rad = 0.0
            
            # 显示所有检测到的物体信息（调试用）
            detected_objects = []
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                class_name = model.names[cls_idx]
                detected_objects.append(f"{class_name}:{conf:.2f}")
                
                # 检查是否是我们要的目标类别
                if class_name == TARGET_CLASS_NAME and conf > GRASP_CONFIDENCE_THRESHOLD:
                    if conf > highest_conf:
                        highest_conf = conf
                        best_box = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
            
            # 在图像上显示检测到的所有物体列表（用于调试）
            if detected_objects:
                objects_text = "Detected: " + ", ".join(detected_objects)
                cv2.putText(display_image, objects_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if best_box is not None:
                # 计算中心点
                u = (best_box[0] + best_box[2]) / 2
                v = (best_box[1] + best_box[3]) / 2
                
                # 在目标类别上额外绘制特殊标记
                x1, y1, x2, y2 = [int(v) for v in best_box]
                # 绘制加厚的目标框
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 4)  # 绿色粗框
                cv2.putText(display_image, f'TARGET: {TARGET_CLASS_NAME} {highest_conf:.2f}', 
                           (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                
                # 绘制中心点
                cv2.circle(display_image, (int(u), int(v)), 8, (0, 0, 255), -1)  # 红色中心点
                cv2.circle(display_image, (int(u), int(v)), 10, (255, 255, 255), 2)  # 白色轮廓

                # --- 在 bbox 内估计主轴方向 ---
                object_angle_rad = estimate_angle_from_bbox(color_image, best_box, debug_img=display_image)
                object_angle_deg = np.rad2deg(object_angle_rad)
                
                # 显示主轴角度信息
                cv2.putText(display_image, f"Main Axis: {object_angle_deg:.1f} deg (RED)",
                           (int(best_box[0]), int(max(0, best_box[1] - 50))),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 计算并显示垂直抓取角度（垂直于主轴的方向）
                # 将弧度转换为角度，然后加90°计算垂直角度
                main_axis_angle_deg = np.rad2deg(object_angle_rad)
                perp_angle_deg = main_axis_angle_deg + 90
                
                # 将垂直角度规范化到 [-90°, 90°) 范围
                if perp_angle_deg >= 90:
                    perp_angle_deg -= 180
                elif perp_angle_deg < -90:
                    perp_angle_deg += 180
                    
                cv2.putText(display_image, f"Grasp Dir: {perp_angle_deg:.1f} deg (BLUE)",
                           (int(best_box[0]), int(max(0, best_box[1] - 30))),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 检测深度
                depth = get_stable_depth(depth_image, u, v)
                if depth > 0:
                    # 未锁定时做稳定计数
                    # 持续更新target_info，避免定格使用旧帧
                    target_info = {'u': u, 'v': v, 'depth': depth, 'angle_rad': object_angle_rad}
                    
                    if not locked:
                        detection_count += 1
                        if detection_count >= max_detections:
                            locked = True
                        print(f"[检测中] '{TARGET_CLASS_NAME}' (置信度: {highest_conf:.2f}), 中心:({int(u)},{int(v)}), 深度:{depth:.3f}m, 主轴角度:{object_angle_deg:.1f}°, 抓取角度:{perp_angle_deg:.1f}° [{detection_count}/{max_detections}]")
                else:
                    detection_count = 0  # 深度无效，重置计数
                    print(f"[警告] 目标中心深度无效(深度:{depth:.3f}m)，重新检测...")
            else:
                detection_count = 0  # 未检测到目标，重置计数
                # 显示当前检测到的所有物体（调试信息）

                if detected_objects:
                    detected_str = ", ".join(detected_objects)
                    print(f"检测到其他物体: {detected_str}，但未找到'{TARGET_CLASS_NAME}'(置信度>{GRASP_CONFIDENCE_THRESHOLD})", end='\r')
                else:
                    print("未检测到任何物体...", end='\r')

            # 若已锁定，叠加引导提示；持续刷新窗口，不定格
            if locked:
                if not announced_locked:
                    print(f"\n[目标锁定] '{TARGET_CLASS_NAME}' 检测稳定。")
                    announced_locked = True
                if not step1_prompted:
                    # 在控制台提示步骤1，并在窗口叠加引导文本
                    print("\n[步骤 1] 移动到目标点上方5cm(预抓取位)...")
                    print("  请在图像窗口按 Enter 确认，或按 ESC 取消。")
                    step1_prompted = True
                cv2.putText(display_image, 'STEP 1: Press Enter to PRE-GRASP (+5cm)', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示检测画面，并读取按键
            cv2.imshow("YOLO Grasping with Direction - Auto Detection", display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 按q或ESC退出
                raise KeyboardInterrupt
            # 在窗口按Enter后关闭可视化并跳出检测循环
            if locked and (key == 13 or key == 10):
                cv2.destroyAllWindows()
                break

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

        # --- 6. 计算最终的抓取位姿（垂直于主轴抓取） ---
        # 基础俯视姿态，使用 UR 原生旋转向量并在旋转矩阵中叠加抓取角
        base_rotvec = observe_pose_rotvec[3:].copy()
        R_base_grasp = R.from_rotvec(base_rotvec).as_matrix()
        
        # 获取物体主轴角度（已规范化到[-90°, 90°]范围内）
        main_axis_angle_rad = target_info['angle_rad']
        main_axis_angle_deg = np.rad2deg(main_axis_angle_rad)
        
        # 计算垂直于主轴的抓取角度（+90°）
        perpendicular_angle_rad = main_axis_angle_rad + np.pi/2
        perpendicular_angle_deg = np.rad2deg(perpendicular_angle_rad)
        
        # 将垂直角度规范化到[-90°, 90°]范围，避免180°大旋转
        if perpendicular_angle_deg >= 90:
            perpendicular_angle_deg -= 180
        elif perpendicular_angle_deg < -90:
            perpendicular_angle_deg += 180
        
        # 坐标系转换：图像坐标系到机器人坐标系的角度转换
        # 图像坐标系：x向右，y向下
        # 机器人坐标系：需要反向角度来匹配实际旋转方向
        perpendicular_angle_deg = -perpendicular_angle_deg
        
        print(f"[坐标系转换] 原始图像角度: {-perpendicular_angle_deg:.1f}°")
        print(f"[坐标系转换] 转换后机器人角度: {perpendicular_angle_deg:.1f}°")
        
        # 应用垂直抓取角度
        object_yaw_offset = np.deg2rad(perpendicular_angle_deg)
        R_yaw_correction = R.from_euler('z', object_yaw_offset).as_matrix()
        final_grasp_rotvec = R.from_matrix(R_base_grasp @ R_yaw_correction).as_rotvec()
        
        print(f"\n[姿态计算] === 详细调试信息 ===")
        print(f"[计算完成] 目标位置 (XYZ): {np.round(target_xyz, 4)}")
        print(f"[计算完成] 保存的主轴角度: {main_axis_angle_deg:.1f}° (来自target_info)")
        print(f"[计算完成] 垂直抓取角度: {perpendicular_angle_deg:.1f}° (主轴+90°)")
        print(f"[计算完成] 应用偏移: {np.rad2deg(object_yaw_offset):.1f}°")
        print(f"[计算完成] 基础姿态 (rxryrz): {np.round(base_rotvec, 4)}")
        print(f"[计算完成] 最终姿态 (rxryrz): {np.round(final_grasp_rotvec, 4)}")
        print(f"[姿态计算] === 调试信息结束 ===")

        # --- 7. 执行完整抓取流程 ---

        approach_xyz = target_xyz.copy()
        approach_xyz[2] += 0.05  # 直接到目标上方5cm
        approach_pose = np.concatenate([approach_xyz, final_grasp_rotvec])
        print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(approach_pose, 4)}")
        # 图像窗口的Enter已确认并关闭窗口，这里直接执行移动
        robot.move_j_pose_rotvec(approach_pose.tolist(), k_acc=0.5, k_vel=0.5)
        print("  ✅ 已到达预抓取位置。")

        # b. 垂直下降到抓取位置（以最终姿态抓取）
        grasp_pose = np.concatenate([target_xyz, final_grasp_rotvec])
        print("\n[步骤 3] 垂直下降到抓取位置...")
        print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(grasp_pose, 4)}")
        input("  请按 Enter 键继续...")
        robot.move_j_pose_rotvec(grasp_pose.tolist(), k_acc=0.2, k_vel=0.2)
        print("  ✅ 已到达抓取位置。")

        # c. 执行抓取
        print("\n[步骤 4] 关闭夹爪...")
        input("  请按 Enter 键继续...")
        robot.close_gripper()
        print("  ✅ 已关闭夹爪。")
        time.sleep(1)

        # d. 垂直上升回到预抓取位置
        print("\n[步骤 5] 垂直上升...")
        print(f"  ==> 目标位姿 (xyzrxryrz): {np.round(approach_pose, 4)}")
        input("  请按 Enter 键继续...")
        robot.move_j_pose_rotvec(approach_pose.tolist(), k_acc=0.5, k_vel=0.5)
        print("  ✅ 已垂直上升。")

        # e. 松开夹爪
        input("[步骤 6] 即将松开夹爪，按 Enter 继续...")
        robot.open_gripper()
        print("✅ 已松开夹爪")

        # f. 回到初始观察位置
        print("\n[步骤 7] 回到初始观察位置...")
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
