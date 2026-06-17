import numpy as np
from UR_Robot import UR_Robot
from scipy.spatial.transform import Rotation as R
import json

def verify_hand_eye_target():
    # 1. 加载标定结果
    with open("../calibration_result1.2.1.json", "r") as f:
        data = json.load(f)
    hand_eye_matrix = np.array(data["hand_eye_matrix"])

    # 2. 初始化机器人
    robot = UR_Robot("192.168.0.1")

    # 3. 指定一个点（相机坐标系下，单位：米）
    point_in_camera = np.array([[0], [-0.05], [0.197], [1]])

    # 4. 转换到 gripper 坐标系下
    point_in_tool = hand_eye_matrix @ point_in_camera

    # 5. 获取当前末端位姿，获得的是末端在基坐标系下的位置（基坐标系下的 T_base_to_tool）
    tcp_pose = robot.get_current_tcp_pose()
    r = R.from_rotvec(np.array(tcp_pose[3:]))  # 当前旋转向量 → 旋转矩阵
    R_base_tool = r.as_matrix()
    t_base_tool = np.array(tcp_pose[:3]).reshape(3, 1)

    T_base_tool = np.eye(4)
    T_base_tool[:3, :3] = R_base_tool
    T_base_tool[:3, 3] = t_base_tool.flatten()

    # 6. 指定的点在基坐标系下的位置
    point_in_base = T_base_tool @ point_in_tool
    target_xyz = point_in_base[:3].flatten()
    print(f"[INFO] 相机点变换后（base系）目标位置: {target_xyz}")

    # 7. 保持当前姿态：旋转向量转为欧拉角
    _, _, _, roll, pitch, yaw = robot.rotvec_to_rpy(*tcp_pose)
    target_pose_rpy = [target_xyz[0], target_xyz[1], target_xyz[2], roll, pitch, yaw]

    # 8. 构造目标位姿 [x, y, z, rpy]
    input("即将移动到该点，按回车继续...")

    # 9. 移动
    robot.move_j_p(target_pose_rpy, k_acc=0.3, k_vel=0.3)

    # 10. 获取实际位置
    actual_pose = robot.get_current_tcp_pose()
    actual_pos = np.array(actual_pose[:3])
    error = (actual_pos - target_xyz) * 1000  # mm
    print("\n✅ 手眼验证完成")
    print(f"期望位置: {target_xyz}")
    print(f"实际位置: {actual_pos}")
    print(f"误差（单位mm）: {error}")

if __name__ == "__main__":
    verify_hand_eye_target()
