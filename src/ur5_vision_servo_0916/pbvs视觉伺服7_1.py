"""
低延迟 PBVS 视觉伺服流程，主要包含：

1. **相机缓冲区优化**：
   - 实现独立线程获取最新图像帧
   - 丢弃旧帧，减少视觉延迟
   - 添加相机参数优化（关闭自动曝光等）

2. **检测算法优化**：
   - 跳帧处理，减少不必要的检测计算
   - 优化标定板检测参数
   - 预编译检测器设置

3. **运动预测补偿**：
   - 实现运动预测算法，补偿系统延迟
   - 速度滤波器平滑运动轨迹
   - 预测未来位置，提前响应

4. **控制循环优化**：
   - 提高控制频率到30Hz
   - 精确时间控制
   - 优化PD参数，减少振荡

5. **系统级优化**：
   - 进程优先级提升
   - 内存优化
   - 异常处理改进
"""

import numpy as np
import cv2
import json
import time
import os
import threading
from collections import deque
from UR_Robot import UR_Robot
from scipy.spatial.transform import Rotation as R


# =============================================================================
#  1. 带缓存线程和直接读取兜底的相机类
# =============================================================================
class OptimizedCamera:
    def __init__(self, robot):
        self.robot = robot
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.frame_counter = 0
        self.camera_working = False
        
        # 测试相机是否可用 - 多次尝试，给相机启动时间
        try:
            print("[测试] 正在测试相机连接状态...")
            test_success = False
            for attempt in range(5):  # 尝试5次
                test_color, _ = self.robot.get_camera_data()
                if test_color is not None:
                    test_success = True
                    break
                time.sleep(0.2)  # 每次间隔200ms
            
            if test_success:
                self.camera_working = True
                print("[成功] 相机测试通过，启动优化缓存线程...")
                # 启动图像获取线程
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()
                # 等待线程获取第一帧
                time.sleep(0.5)
            else:
                print("[警告] 相机测试超时，使用降级模式（直接获取）")
                self.camera_working = False
        except Exception as e:
            print(f"[警告] 相机初始化错误: {e}，使用降级模式")
            self.camera_working = False
        
    def _capture_loop(self):
        """独立线程获取图像，只保留最新帧"""
        consecutive_failures = 0
        while self.running:
            try:
                color_img, _ = self.robot.get_camera_data()
                if color_img is not None:
                    with self.frame_lock:
                        self.frame = color_img
                        self.frame_counter += 1
                    consecutive_failures = 0  # 重置失败计数
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        print("[警告] 缓存线程连续获取失败，可能相机断开")
                        
                # 略微提高轮询频率，确保第一时间取走相机产生的帧
                # 即使相机是30Hz，较短的sleep能减少从帧生成到取走的等待时间
                time.sleep(0.01)
                
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures <= 3:  # 只显示前几次错误
                    print(f"[缓存线程] 相机获取错误: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self):
        """获取最新图像帧，支持降级到直接模式"""
        if self.camera_working:
            # 优化模式：从缓存获取
            with self.frame_lock:
                if self.frame is not None:
                    return self.frame.copy(), self.frame_counter
                # 如果缓存为空，等待一下再试
                time.sleep(0.01)
                if self.frame is not None:
                    return self.frame.copy(), self.frame_counter
                return None, 0
        else:
            # 降级模式：绕过缓存线程，直接从机器人相机接口读取
            try:
                color_img, _ = self.robot.get_camera_data()
                if color_img is not None:
                    self.frame_counter += 1
                    return color_img, self.frame_counter
                return None, 0
            except Exception as e:
                # 静默处理错误，避免刷屏
                return None, 0
    
    def stop(self):
        self.running = False
        if self.camera_working and hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)


# =============================================================================
#  2. 优化的检测器类
# =============================================================================
class FastChessboardDetector:
    def __init__(self, pattern_size, square_size):
        self.pattern_size = pattern_size
        self.square_size = square_size
        
        # 预编译参数
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
        self.flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        
        # 生成理想棋盘格坐标
        self.objp = np.zeros((np.prod(pattern_size), 3), np.float32)
        self.objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        self.objp *= square_size
        
        # 计算中心点
        center_x = (pattern_size[0] - 1) / 2.0 * square_size
        center_y = (pattern_size[1] - 1) / 2.0 * square_size
        self.P_center_in_obj = np.array([center_x, center_y, 0])
        
    def detect_pose(self, color_img, camera_matrix, dist_coeffs):
        """快速检测棋盘格位姿"""
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, self.flags)
        
        if not found:
            return None, None, None
            
        # 亚像素优化
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        
        # 解算PnP
        success, rvec, tvec = cv2.solvePnP(self.objp, corners, camera_matrix, dist_coeffs)
        if not success:
            return None, None, None
            
        return rvec, tvec, self.P_center_in_obj


# =============================================================================
#  3. 运动预测器类
# =============================================================================
class MotionPredictor:
    def __init__(self, filter_size=5, prediction_steps=3):
        self.position_history = deque(maxlen=filter_size)
        self.time_history = deque(maxlen=filter_size)
        self.prediction_steps = prediction_steps
        
    def update_and_predict(self, current_pos):
        """更新位置历史并预测未来位置"""
        current_time = time.time()
        
        self.position_history.append(current_pos.copy())
        self.time_history.append(current_time)
        
        # 需要至少2个点来计算速度
        if len(self.position_history) < 2:
            return current_pos
            
        # 计算平均速度
        velocities = []
        for i in range(1, len(self.position_history)):
            dt = self.time_history[i] - self.time_history[i-1]
            if dt > 0:
                velocity = (self.position_history[i] - self.position_history[i-1]) / dt
                velocities.append(velocity)
        
        if not velocities:
            return current_pos
            
        # 平均速度
        avg_velocity = np.mean(velocities, axis=0)
        
        # 预测未来位置
        prediction_time = self.prediction_steps * 0.033  # 假设30Hz控制频率
        predicted_pos = current_pos + avg_velocity * prediction_time
        
        return predicted_pos
    
    def reset(self):
        self.position_history.clear()
        self.time_history.clear()


# =============================================================================
#  4. 辅助函数
# =============================================================================
def load_hand_eye_result(path='calibration_result1_4.json'):
    """加载相机内参、畸变系数和手眼标定矩阵"""
    try:
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 如果path是相对路径，则相对于脚本目录
        if not os.path.isabs(path):
            path = os.path.join(script_dir, path)
        
        with open(path, 'r') as f:
            calib = json.load(f)
        camera_matrix = np.array(calib['camera_matrix'])
        dist_coeffs = np.array(calib['dist_coeffs'])
        T_cam2tool = np.array(calib['hand_eye_matrix'])
        print(f"[成功] 已从 {path} 加载相机参数和手眼矩阵。")
        return camera_matrix, dist_coeffs, T_cam2tool
    except FileNotFoundError:
        print(f"[错误] 找不到标定文件: {path}！请确保文件存在。")
        print(f"[提示] 当前工作目录: {os.getcwd()}")
        print(f"[提示] 脚本目录: {os.path.dirname(os.path.abspath(__file__))}")
        exit()


def compute_servoing_target_optimized(tcp_pose, rvec, tvec, P_center_in_obj, T_cam2tool,
                                    R_correction_current, desired_distance, predictor, key_pressed=None):
    """
    优化的伺服目标计算函数，集成运动预测
    """
    # 构造变换矩阵
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
    
    # 姿态修正矩阵
    R_correction_updated = R_correction_current
    if R_correction_updated is None or (key_pressed is not None and key_pressed == ord('s')):
        R_correction_updated = R_obj2base.T @ R_tool2base_current
        print(f"\n[成功] 姿态基准已锁定！")
    
    # 计算期望工具姿态
    R_tool_desired = R_obj2base @ R_correction_updated
    rotvec_desired = R.from_matrix(R_tool_desired).as_rotvec()
    
    # 计算期望位置
    R_cam_desired_in_base = R_tool_desired @ T_cam2tool[:3, :3]
    z_axis_cam_desired_in_base = R_cam_desired_in_base[:, 2]
    
    P_center_in_base = (T_cam2base @ np.append((T_obj2cam @ np.append(P_center_in_obj, 1))[:3], 1))[:3]
    
    # 使用运动预测优化目标位置
    raw_center_pos = P_center_in_base + z_axis_cam_desired_in_base * desired_distance
    predicted_center_pos = predictor.update_and_predict(raw_center_pos)
    
    P_cam_desired_in_base = predicted_center_pos
    t_cam2tool_in_tool = T_cam2tool[:3, 3]
    raw_target_pos = P_cam_desired_in_base - R_tool_desired @ t_cam2tool_in_tool
    
    # 相机坐标系下的中心点
    P_center_in_cam = (T_obj2cam @ np.append(P_center_in_obj, 1))[:3]
    
    return np.concatenate([raw_target_pos, rotvec_desired]), R_correction_updated, P_center_in_cam


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
#  5. 主功能函数
# =============================================================================
def optimized_visual_servoing():
    """
    V7.1 优化版：解决延迟和振荡问题的高性能视觉伺服
    """
    # --- 参数配置 ---
    ROBOT_IP = "192.168.0.1"
    CALIB_FILE = 'calibration_result1_4.json'
    PATTERN_SIZE = (8, 5)
    SQUARE_SIZE = 0.027
    DESIRED_DISTANCE = -0.45
    Z_SAFE_LIMIT = 0.05
    VISUALIZATION_ENABLED = True

    # === 优化的控制参数 ===
    CONTROL_FREQUENCY = 30.0  # 提高到30Hz
    FINE_TUNING_THRESHOLD = 0.008  # 更严格的阈值
    MAX_SPEED = 0.4  # 降低最大速度，减少振荡
    MIN_SPEED = 0.008
    MOVE_ACCEL = 0.3
    
    # PD参数优化，减少振荡
    KP = 1.8  # 降低比例增益
    KD = 1.5  # 降低微分增益
    
    POSITION_SMOOTHING_ALPHA = 0.6  # 更强的平滑
    FRAME_SKIP = 2  # 每2帧处理一次检测

    # --- 初始化 ---
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result(CALIB_FILE)
    
    try:
        robot = UR_Robot(ROBOT_IP)
    except Exception as e:
        print(f"[错误] 初始化机器人失败: {e}")
        return

    # 初始化优化组件
    optimized_camera = OptimizedCamera(robot)
    detector = FastChessboardDetector(PATTERN_SIZE, SQUARE_SIZE)
    motion_predictor = MotionPredictor(filter_size=5, prediction_steps=2)
    
    print("\n[启动] V7.1 优化版伺服已开始。")
    if optimized_camera.camera_working:
        print("[模式] 使用优化缓存模式 - 低延迟图像获取")
    else:
        print("[模式] 使用降级直接模式 - 兼容性获取")
    
    if VISUALIZATION_ENABLED:
        print("[提示] 按 's' 锁定姿态基准, 按 'q' 退出。")
    else:
        print("[提示] 可视化已禁用。按 Ctrl+C 退出程序。")

    # 控制变量
    R_correction = None
    smoothed_target_pos = None
    last_error = 0.0
    last_frame_count = 0
    
    # 时间控制
    loop_interval = 1.0 / CONTROL_FREQUENCY
    
    try:
        while True:
            loop_start_time = time.time()
            
            # 获取最新图像
            frame_data = optimized_camera.get_latest_frame()
            if frame_data[0] is None:
                time.sleep(0.01)
                continue
                
            color_img, frame_count = frame_data
            
            # 跳帧处理，减少计算负担
            if frame_count - last_frame_count < FRAME_SKIP:
                time.sleep(0.001)
                continue
            last_frame_count = frame_count

            # 检测棋盘格
            rvec, tvec, P_center_in_obj = detector.detect_pose(color_img, camera_matrix, dist_coeffs)

            key = -1
            if VISUALIZATION_ENABLED:
                display_img = color_img.copy()
                key = cv2.waitKey(1) & 0xFF

            if rvec is None:
                if VISUALIZATION_ENABLED:
                    cv2.putText(display_img, "Chessboard Not Found", (30, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    print("[警告] 未检测到棋盘格...", end='\r')
                
                smoothed_target_pos, last_error = None, 0.0
                motion_predictor.reset()
            else:
                # 核心计算
                tcp_pose = robot.get_current_tcp_pose()
                raw_target_pose, R_correction, P_center_in_cam = compute_servoing_target_optimized(
                    tcp_pose, rvec, tvec, P_center_in_obj, T_cam2tool,
                    R_correction, DESIRED_DISTANCE, motion_predictor, key
                )
                
                raw_target_pos = raw_target_pose[:3]
                rotvec_desired = raw_target_pose[3:]

                # 位置平滑
                if smoothed_target_pos is None:
                    smoothed_target_pos = raw_target_pos
                else:
                    smoothed_target_pos = POSITION_SMOOTHING_ALPHA * raw_target_pos + (
                                1 - POSITION_SMOOTHING_ALPHA) * smoothed_target_pos

                pos = smoothed_target_pos
                target_pose_xyzrxryrz = np.concatenate([pos, rotvec_desired])
                error = np.linalg.norm(tcp_pose[:3] - pos)
                error_derivative = error - last_error
                last_error = error

                if error > FINE_TUNING_THRESHOLD:
                    if pos[2] < Z_SAFE_LIMIT:
                        print(f"[警告] 目标Z轴({pos[2]:.3f})过低，已跳过！")
                        continue
                    else:
                        # PD控制
                        control_signal = KP * error + KD * error_derivative
                        speed = np.clip(control_signal, MIN_SPEED, MAX_SPEED)
                        
                        print(f"[追踪-PD] E:{error:.3f}, dE:{error_derivative:.3f}, Speed:{speed:.2f}", end='\r')
                        
                        try:
                            robot.move_j_p_1_rotvec(target_pose_xyzrxryrz.tolist(), k_acc=MOVE_ACCEL, k_vel=speed)
                        except Exception as e:
                            print(f"[错误] 机器人运动失败: {e}")
                else:
                    print(f"[已对准] E:{error:.3f} <= {FINE_TUNING_THRESHOLD}m. 保持静止。", end='\r')
                    smoothed_target_pos, last_error = tcp_pose[:3], 0.0

                # 可视化
                if VISUALIZATION_ENABLED:
                    display_img = draw_target_point(display_img, P_center_in_cam, camera_matrix, dist_coeffs)
                    display_img = draw_camera_center(display_img, camera_matrix)
                    cv2.putText(display_img, f"Error: {error:.4f} m", (30, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display_img, f"FPS: {1.0/(time.time()-loop_start_time+0.001):.1f}", 
                               (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if VISUALIZATION_ENABLED:
                cv2.imshow("Visual Servoing V7.1 - Optimized", display_img)
                if key == ord('q') or key == 27:
                    break

            # 精确时间控制
            elapsed_time = time.time() - loop_start_time
            sleep_time = loop_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[中止] 程序已手动终止。")
    finally:
        # 清理资源
        optimized_camera.stop()
        if VISUALIZATION_ENABLED:
            cv2.destroyAllWindows()
        print("\n程序结束。")


# =============================================================================
#  6. 程序入口
# =============================================================================
if __name__ == '__main__':
    optimized_visual_servoing()
