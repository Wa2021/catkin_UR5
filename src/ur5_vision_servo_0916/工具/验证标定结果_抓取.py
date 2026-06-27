import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as R
from UR_Robot import UR_Robot

# 初始化机器人
robot = UR_Robot("192.168.0.1")
robot.open_gripper()

# 读取标定数据
with open(r"/calibration_result1_4.json", "r") as f:
    calib = json.load(f)
camera_matrix = np.array(calib["camera_matrix"])
T_cam2tool = np.array(calib["hand_eye_matrix"])  # 从相机坐标系 → 工具坐标系

# 初始位置
home_pose = [-0.483, -0.065, 0.297, 2.211, -2.224, -0.154]  # xyz + rotvec格式
robot.move_j_pose_rotvec(home_pose, k_acc=0.1, k_vel=0.1)


print("[INFO] 等待点击图像上的点进行验证...\n")


# 回调函数：点击图像触发
def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    print(f"\n[点击图像点] 像素坐标: ({x}, {y})")

    color_img, depth_img = robot.get_camera_data()
    depth_val = depth_img[y, x][0]

    if depth_val <= 0.01:
        print("❌ 深度无效或为0，跳过")
        return
    print(f"[深度信息] 像素点 ({x}, {y}) 的深度值为: {depth_val:.4f} 米")
    # 获取当前末端姿态（基坐标系下的位姿）
    tcp_pose = robot.get_current_tcp_pose()
    R_tool2base = R.from_rotvec(np.array(tcp_pose[3:])).as_matrix()
    t_tool2base = np.array(tcp_pose[:3]).reshape(3, 1)

    T_tool2base = np.eye(4)
    T_tool2base[:3, :3] = R_tool2base
    T_tool2base[:3, 3] = t_tool2base.flatten()

    print("\n[INFO] 当前末端基坐标位置：", tcp_pose[:3])

    # 像素坐标 → 相机坐标
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    x_cam = (x - cx) * depth_val / fx
    y_cam = (y - cy) * depth_val / fy
    z_cam = depth_val
    point_cam = np.array([[x_cam], [y_cam], [z_cam], [1]])

    # 从相机坐标系中的坐标变成工具坐标系下的坐标
    point_tool = T_cam2tool @ point_cam

    # 从工具坐标系中的坐标变成基座坐标系下的坐标
    point_base = T_tool2base @ point_tool
    target_xyz = point_base[:3].flatten()

    # 输出误差比较
    tcp_xyz = np.array(tcp_pose[:3])
    error = (target_xyz - tcp_xyz) * 1000  # mm

    print(f"[计算位置] 相机坐标系点: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}) m")
    print(f"[转换结果] 原始目标点: {target_xyz.round(4)}")
    print(f"[当前 TCP] 位姿: {tcp_xyz.round(4)}")
    print(f"[误差] (mm): {error.round(2)}")

    # 定义补偿向量 (x, y, z)，单位为米
    # y轴正方向偏1cm -> y + 0.01
    # z轴正方向加1cm -> z + 0.01
    compensation_vector = np.array([0.0, 0.01, 0.01])

    print(f"\n[补偿操作] 应用补偿向量: {compensation_vector}")

    # 将补偿应用到原始目标点上
    target_xyz_compensated = target_xyz + compensation_vector

    print(f"[补偿后] 最终目标点: {target_xyz_compensated.round(4)}\n")

    # 当前末端姿态（保持夹爪方向不变）
    rotvec = tcp_pose[3:]

    # === Step 1: 移动到目标点上方（提前 5cm）===
    # 使用补偿后的目标点来计算预抓取位置
    approach_xyz = target_xyz_compensated.copy()
    approach_xyz[2] += 0.05
    pose_above = [*approach_xyz, *rotvec]

    input("[Step 1] 即将移动到目标点上方，按 Enter 继续...")
    robot.move_j_pose_rotvec(pose_above, k_acc=0.1, k_vel=0.1)
    print("✅ 已移动到目标点上方")

    # === Step 2: 下降到实际抓取位置 ===
    # 使用补偿后的目标点作为抓取位置
    pose_grasp = [*target_xyz_compensated, *rotvec]

    input("[Step 2] 即将下降到目标位置，按 Enter 继续...")
    robot.move_j_pose_rotvec(pose_grasp, k_acc=0.1, k_vel=0.1)
    print("✅ 已到达抓取位置")

    # === Step 3: 执行夹爪抓取动作 ===
    input("[Step 3] 即将关闭夹爪，按 Enter 继续...")
    robot.close_gripper()
    print("✅ 已执行夹取")

    # === Step 4: 上升回到目标点上方 ===
    input("[Step 4] 即将上升到安全高度，按 Enter 继续...")
    robot.move_j_pose_rotvec(pose_above, k_acc=0.1, k_vel=0.1)
    print("✅ 已上升回预抓取位姿")

    # === Step 5: 松开夹爪 ===
    input("[Step 5] 即将松开夹爪，按 Enter 继续...")
    robot.open_gripper()
    print("✅ 已松开夹爪")

    # === Step 6: 回到初始位姿 ===
    input("[Step 6] 即将回到初始位姿，按 Enter 继续...")
    robot.move_j_pose_rotvec(home_pose, k_acc=0.1, k_vel=0.1)
    print("✅ 已回到初始位姿")


# 窗口设置
cv2.namedWindow("Click  to Grasp")
cv2.setMouseCallback("Click  to Grasp", on_click)

while True:
    img, _ = robot.get_camera_data()
    cv2.imshow("Click  to Grasp", img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
