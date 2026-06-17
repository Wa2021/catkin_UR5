#demo01只能是固定位姿的抓取，瓶子横躺在桌面上，也不一定是正对着夹爪
# 夹爪有时候需要旋转一定角度才能正对着瓶子进行抓取,要让夹爪能够适应物体（比如横躺的瓶子）的朝向
# 所以使用了分割模型，计算物体的掩码，然后求出最小外接矩形,从而计算出物体的主方向
import numpy as np
import cv2
import json
import time
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
    h, w = depth_image.shape
    u_min, u_max = max(0, u - roi_size), min(w - 1, u + roi_size)
    v_min, v_max = max(0, v - roi_size), min(h - 1, v + roi_size)
    # 注意切片右端应 +1
    roi = depth_image[v_min:v_max + 1, u_min:u_max + 1]
    valid_depths = roi[roi > 0]
    if valid_depths.size == 0:
        return 0.0
    return float(np.median(valid_depths))  # 使用中值对异常值更鲁棒



#  用于姿态估计的辅助函数
def get_grasp_angle_from_mask(mask, display_image):
    """
    从物体的掩码中计算抓取角度。
    :param mask: 物体的二值化掩码图。
    :param display_image: 用于绘制的图像。
    :return: 抓取角度（弧度）。
    """
    if mask is None:
        return 0.0  # 输入为空，返回默认角度0

    # --- 1. 掩码预处理 ---
    # mask 可能是 0/1 或 0/255，这里统一转成 uint8 类型
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() == 1:
        # 如果是 0/1 掩码，就放大到 0/255
        mask_uint8 = mask_uint8 * 255
    # 找到掩码的轮廓

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL：只提取最外层轮廓。
    # cv2.CHAIN_APPROX_SIMPLE： 简化轮廓点（压缩数据量）。
    if not contours:
        return 0.0
    #cv2.findContours返回一个轮廓列表 contours。每个 contour 是一个由若干点组成的多边形轮廓

    # 找到最大的轮廓（通常就是我们的目标物体）
    main_contour = max(contours, key=cv2.contourArea)
    #max(contours, key=cv2.contourArea)作用：遍历列表 contours，把每个轮廓交给 cv2.contourArea() 求面积。选出面积最大的那个轮廓。
    #掩码图里，可能有很多小噪点（小白点轮廓），但通常目标物体是面积最大的那个。
    # cv2.contourArea(contour) 这个函数计算给定contour的轮廓面积，单位是像素个数，面积越大，轮廓越大。

    # 计算最大轮廓的最小外接矩形
    # rect 的格式是 ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(main_contour)

    # --- 可视化：绘制旋转的边界框 ---
    box_points = cv2.boxPoints(rect)#生成四个顶点坐标
    box_points = box_points.astype(np.int32)#将数组中的小数转成整数
    if display_image is not None:
        cv2.drawContours(display_image, [box_points], 0, (0, 255, 0), 2)  # 绿色边框
    #drawContours 支持画多个轮廓，单个轮廓也要加方括号 0：绘制第0个轮廓  2：线宽是两个像素

    # --- 角度计算 ---
    # cv2.minAreaRect 返回的角度范围是 [-90, 0)
    # width 总是比 height 小
    # 我们需要将它转换为更直观的0到180度范围
    angle = rect[2]
    width, height = rect[1]

    # 我们通常希望沿着物体的长边进行抓取
    if width > height:
        # 如果宽度大于高度，说明矩形是“横着”的，角度需要加90度
        grasp_angle_deg = angle + 90
    else:
        # 如果是“竖着”的，角度就是它本身
        grasp_angle_deg = angle

        # --- 7. 角度归一化 ---
        # 把角度限制到 [0, 180)，避免出现 ±90 跳变
    grasp_angle_deg = grasp_angle_deg % 180

    # --- 8. 转换为弧度 ---
    grasp_angle_rad = np.deg2rad(grasp_angle_deg)

    # --- 9. 可视化抓取方向 ---
    if display_image is not None:
        center = tuple(np.int32(rect[0]))  # 矩形中心点
        length = int(max(width, height) / 2)  # 箭头长度（长边的一半）
        dx = int(length * np.cos(grasp_angle_rad))
        dy = int(length * np.sin(grasp_angle_rad))
        end_point = (center[0] + dx, center[1] + dy)
        cv2.arrowedLine(display_image, center, end_point, (0, 0, 255), 2, tipLength=0.3)  # 红色箭头

    return grasp_angle_rad

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
    # ✅ 使用分割模型
    YOLO_MODEL_PATH = 'yolov8n-seg.pt'
    TARGET_CLASS_NAME = 'bottle'  # 你想要抓取的目标类别，例如：'bottle', 'cup'等
    GRASP_CONFIDENCE_THRESHOLD = 0.5  # YOLO检测的置信度阈值

    # --- 2. 初始化 ---
    camera_matrix, dist_coeffs, T_cam2tool = load_hand_eye_result(CALIB_FILE)

    print("[YOLO] 正在加载分割模型...")
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
    #observe_pose_rpy = [-0.478, -0.0678, 0.336, 2.222, -2.22, -0.140]
    observe_pose_rpy = np.array([-0.478, -0.0678, 0.45, 2.222, -2.22, 0.0])  # ✅ 将Yaw设为0，方便计算
    print("\n[步骤 0] 移动到初始观察位置...")
    print(f"  ==> 目标位姿 (RPY): {np.round(observe_pose_rpy, 4)}")
    input("  请按 Enter 键继续...")
    robot.move_j_p(observe_pose_rpy, k_acc=0.5, k_vel=0.5)
    print("  ✅ 已到达观察位置。")

    try:
        while True:
            print("\n" + "=" * 20)
            print("等待检测目标... 将鼠标聚焦在图像窗口上，按 'g' 键锁定目标并开始抓取，按 'q' 退出。")

            # --- 4. 循环检测，等待用户触发抓取 ---
            target_info = None
            while True:
                color_image, depth_image = robot.get_camera_data()
                if color_image is None:
                    continue

                # YOLO 推理
                # --- 注意颜色通道: 将 BGR -> RGB 给模型（若你的get_camera_data返回BGR）
                input_img_for_model = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                results = model(input_img_for_model, verbose=False)
                display_image = results[0].plot()  # 获取绘制了所有框的图像

                best_box, highest_conf, best_mask = None, 0, None
                for i, box in enumerate(results[0].boxes):  # enumerate 是 python 的内置函数，可以同时返回索引和当前迭代的对象
                    conf = float(box.conf[0])  # 将原本可能是张量类型的数据转换成 float 类型
                    if model.names[int(box.cls[0])] == TARGET_CLASS_NAME and conf > GRASP_CONFIDENCE_THRESHOLD:
                        # box.cls 是个 tensor 张量，比如 tensor([0.])，所以要进行类型转换
                        if conf > highest_conf:
                            highest_conf = conf
                            best_box = box.xyxy[0].cpu().numpy()
                            # box.xyxy 是 这个框的坐标，形状是 tensor([[x1, y1, x2, y2]]) 坐标格式是 (左上角x, 左上角y, 右下角x, 右下角y)。

                            # ✅ 获取对应的掩码（需要确保分割模型有 masks）
                            if results[0].masks is not None:
                                best_mask = results[0].masks.data[i].cpu().numpy().astype(np.uint8)
                                # results[0].masks.data 是 每个检测框对应的掩码（二值图像）集合。
                                # [i] 拿到第 i 个目标的掩码，形状类似 tensor(H, W)，H/W 是图片大小，值为 0 或 1。
                                # YOLO 的计算通常在 GPU 上进行，结果也保存在 GPU 显存里。
                                # .cpu(): 把数据从 GPU 显存转移到 CPU 内存。
                                # .numpy(): 把 PyTorch 的 Tensor 格式转换成我们熟悉的 NumPy 数组格式
                                # astype 强制类型转换，OpenCV 显示或后续处理二值图，通常要求是 uint8 类型。最终变成值是 0 或 1，数据类型是 uint8。

                if best_box is not None and best_mask is not None:
                    # 计算目标框的中心点
                    u, v = (best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2
                    cv2.circle(display_image, (int(u), int(v)), 8, (0, 0, 255), -1)

                    # ✅ 只有在掩码有效时才计算抓取角度
                    grasp_angle_rad = get_grasp_angle_from_mask(best_mask, display_image)
                    grasp_angle_deg = np.rad2deg(grasp_angle_rad)  # 弧度转换成角度
                    cv2.putText(display_image, f"Angle: {grasp_angle_deg:.1f} deg",
                                (int(best_box[0]), int(best_box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # 如果没有检测到目标或掩码为空，则跳过角度计算
                    grasp_angle_rad = 0.0

                cv2.imshow("YOLO Grasping with Pose", display_image)
                key = cv2.waitKey(1) & 0xFF  # 最多等 1 毫秒，超时了就继续执行下一行代码。waitKey(0)：无限等待，直到你按下键盘。
                # 有些操作系统（特别是 Windows）waitKey() 会返回一个 32bit 整数，但实际有效只有低 8 位。所以 用 & 0xFF 防止返回奇怪的大整数。

                if key == ord('q'):
                    raise KeyboardInterrupt  # 触发退出
                elif key == ord('g'):
                    if best_box is None:
                        print("[警告] 当前没有检测到目标，无法抓取！")
                        continue  # 重新下一轮，重新检测
                    depth = get_stable_depth(depth_image, u, v)
                    if depth > 0:
                        target_info = {'u': u, 'v': v, 'depth': depth, 'angle_rad': grasp_angle_rad}
                        print(f"\n[锁定目标] '{TARGET_CLASS_NAME}' (置信度: {highest_conf:.2f}), 中心:({int(u)},{int(v)}), 深度:{depth:.3f}m")
                        break  # ✅ 可以去执行后续动作了
                    else:
                        print("[警告] 目标中心深度无效，跳过该帧！")
                        best_box = None  # 清空，防止继续误用旧框
                        continue  # 重新下一轮检测

            # --- 5. 计算目标世界坐标 ---
            if target_info is None: continue

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


            # 步骤 6. 计算最终的抓取位姿
            #  目标姿态
            # 1. 定义基准俯视姿态
            # 这是我们希望机器人抓取时保持的基本姿态（除了绕Z轴的旋转）
            # 我们直接使用初始观察位姿的姿态部分作为基准
            base_grasp_rpy = observe_pose_rpy[3:]
            R_base_grasp = R.from_euler('xyz', base_grasp_rpy).as_matrix()#把欧拉角转换成旋转矩阵

            # 2. 获取我们从视觉中计算出的物体旋转角度
            # grasp_angle_rad 是物体长轴与图像水平线(X轴)的夹角
            object_angle_rad = target_info['angle_rad']

            # 3. 确定工具坐标系中哪个轴是我们的“夹持轴”
            # 夹爪是平行于y轴的，收紧时抓住x轴
            # 应该让工具的 X 轴去对齐物体的长轴。
            # 在标准的工具坐标系中，X轴是红色的轴。

            # 4. 计算最终的旋转姿态
            # 我们的目标是，让最终姿态下的工具X轴，在世界坐标系中的投影，
            # 与物体的 `object_angle_rad` 对齐。

            # 思考一下坐标系的对齐关系：
            # - 初始时，工具坐标系 T 有一个由 R_base_grasp 定义的姿态。
            # - 我们希望旋转工具坐标系 T，得到新的坐标系 T'。
            # - 使得 T' 的 X 轴，在俯视时（在XY平面上）的方向，与物体的 `object_angle_rad` 一致。

            # 我们来分析 R_base_grasp，它将基坐标系旋转到工具坐标系。
            # 在这个姿态下，工具的X轴在基坐标系中的表示是 R_base_grasp 的第一列。
            # 它的Yaw角是多少呢？我们可以从这个向量的 arctan2(y, x) 得到。
            tool_x_axis_in_base = R_base_grasp[:, 0]#取所有行的第0列，也就是x轴在基坐标系下的朝向
            current_tool_yaw_rad = np.arctan2(tool_x_axis_in_base[1], tool_x_axis_in_base[0])
            #因为是平面抓取，所以只在xy平面上，要求的就是 X 轴在 XY 平面上的偏转角（Yaw）。
            #这个current_tool_yaw_rad = np.arctan2(y, x)求的就是 X 轴在 XY 平面上的偏转角（Yaw）。
            #也就是从 X 轴正方向顺时针转了多少角度能对齐这个向量。这个角度用于补偿夹爪朝向。



            # 我们期望的工具X轴的Yaw角是 object_angle_rad。
            # 所以，我们需要额外旋转的角度是：
            yaw_correction_rad = object_angle_rad - current_tool_yaw_rad

            # 创建一个只包含这个修正量的、绕世界Z轴的旋转
            R_yaw_correction = R.from_euler('z', yaw_correction_rad).as_matrix()
            #这个函数的核心意思是：从一个欧拉角，创建出对应的旋转矩阵。这里 'z' 表示：绕 Z 轴旋转，yaw_correction_rad 是你需要绕 Z 轴转多少弧度。

            # 先应用基础俯视姿态（R_base_grasp），再在这个基础上绕Z轴补偏转（R_yaw_correction）。
            # 所以顺序是右乘：R_final = R_base_grasp @ R_yaw_correction
            R_final_grasp = R_base_grasp @ R_yaw_correction

            # 将最终的旋转矩阵转换回RPY格式，用于发送给机器人
            final_grasp_rpy = R.from_matrix(R_final_grasp).as_euler('xyz')

            print(f"\n[姿态计算]")
            print(f"  - 物体角度 (视觉): {np.rad2deg(object_angle_rad):.2f} 度")
            print(f"  - 工具初始Yaw角: {np.rad2deg(current_tool_yaw_rad):.2f} 度")
            print(f"  - 需要修正的Yaw角: {np.rad2deg(yaw_correction_rad):.2f} 度")
            print(f"[计算完成] 目标位置 (XYZ): {np.round(target_xyz, 4)}")
            print(f"[计算完成] 最终姿态 (RPY): {np.round(final_grasp_rpy, 4)}")


            # --- 7. 分步执行抓取动作 ---
            # a. 移动到预抓取位置
            approach_xyz = target_xyz.copy()
            approach_xyz[2] += 0.05
            approach_pose = np.concatenate([approach_xyz, final_grasp_rpy])

            print("\n[步骤 1] 移动到目标点上方...")
            print(f"  ==> 目标位姿 (RPY): {np.round(approach_pose, 4)}")
            input("  请按 Enter 键继续...")
            robot.move_j_p(approach_pose.tolist(), k_acc=0.5, k_vel=0.5)

            # b. 垂直下降
            grasp_pose = np.concatenate([target_xyz, final_grasp_rpy])
            print("\n[步骤 2] 垂直下降到抓取位置...")
            print(f"  ==> 目标位姿 (RPY): {np.round(grasp_pose, 4)}")
            input("  请按 Enter 键继续...")
            robot.move_j_p(grasp_pose.tolist(), k_acc=0.2, k_vel=0.2)

            # c. 执行抓取 (示例，如果需要请取消注释)
            print("\n[步骤 3] 关闭夹爪...")
            input("  请按 Enter 键继续...")
            robot.close_gripper()
            print("  ✅ 已关闭夹爪。")
            time.sleep(1)

            # d. 垂直上升回到预抓取位置
            print("\n[步骤 4] 垂直上升...")
            print(f"  ==> 目标位姿 (RPY): {np.round(approach_pose, 4)}")
            input("  请按 Enter 键继续...")
            robot.move_j_p(approach_pose.tolist(), k_acc=0.5, k_vel=0.5)
            print("  ✅ 已垂直上升。")

            # e. 松开夹爪
            input("[步骤 5] 即将松开夹爪，按 Enter 继续...")
            robot.open_gripper()
            print("✅ 已松开夹爪")

            # f. 回到初始观察位置
            print("\n[步骤 6] 回到初始观察位置...")
            print(f"  ==> 目标位姿 (RPY): {np.round(observe_pose_rpy, 4)}")
            input("  请按 Enter 键继续...")
            robot.move_j_p(observe_pose_rpy, k_acc=0.5, k_vel=0.5)
            print("  ✅ 已回到初始位置。抓取流程结束。")

    except KeyboardInterrupt:
        print("\n[中止] 程序已手动终止。")
    finally:
        cv2.destroyAllWindows()
        print("\n程序结束。")


if __name__ == '__main__':
    yolo_grasping_with_debug_steps()


