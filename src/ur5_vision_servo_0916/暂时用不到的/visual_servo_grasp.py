import numpy as np
import time
from UR_Robot import UR_Robot
from 暂时用不到的.visual_servo_control import VisualServoController

class VisualServoGrasping:
    def __init__(self, robot_ip="192.168.50.100", model_path="best.pt"):
        """
        初始化视觉伺服抓取系统
        :param robot_ip: UR机器人IP地址
        :param model_path: YOLOv8模型路径
        """
        # 初始化UR机器人
        self.robot = UR_Robot(tcp_host_ip=robot_ip)
        
        # 获取相机参数
        self.camera_matrix = self.robot.cam_intrinsics
        self.dist_coeffs = None  # 如果有畸变系数，在这里设置
        
        # 初始化视觉伺服控制器
        self.vs_controller = VisualServoController(
            model_path=model_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )
        
        # 设置控制参数
        self.max_iterations = 100  # 最大迭代次数
        self.distance_threshold = 0.01  # 位置误差阈值（米）
        self.angle_threshold = 0.05  # 角度误差阈值（弧度）
        
    def get_robot_camera_transform(self):
        """
        获取机器人末端执行器到相机的转换矩阵
        """
        # 这里需要根据实际的标定结果设置
        # 示例转换矩阵，需要根据实际标定结果修改
        R = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        t = np.array([0.05, 0, 0.05])  # 根据实际安装位置调整
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
        
    def visual_servo_to_target(self):
        """
        执行视觉伺服控制，将机器人移动到目标位置
        """
        iteration = 0
        while iteration < self.max_iterations:
            # 获取相机图像
            rgb_image, depth_image = self.robot.get_camera_data()
            
            # 处理图像并获取控制命令
            v_camera, bbox = self.vs_controller.process_frame(rgb_image, depth_image)
            
            if v_camera is None:
                print("No target detected")
                return False
                
            # 转换为机器人速度
            robot_camera_transform = self.get_robot_camera_transform()
            v_robot = self.vs_controller.camera_velocity_to_robot_velocity(
                v_camera, 
                robot_camera_transform
            )
            
            # 检查是否达到目标
            if np.linalg.norm(v_robot[:3]) < self.distance_threshold and \
               np.linalg.norm(v_robot[3:]) < self.angle_threshold:
                print("Target reached")
                return True
                
            # 执行运动
            current_pose = self.robot.get_current_tool_pos()
            new_pose = current_pose + v_robot * 0.1  # 0.1秒的控制周期
            
            try:
                self.robot.move_l(new_pose.tolist(), k_acc=0.3, k_vel=0.3)
            except Exception as e:
                print(f"Motion failed: {e}")
                return False
                
            iteration += 1
            time.sleep(0.1)
            
        print("Max iterations reached")
        return False
        
    def execute_grasp(self):
        """
        执行完整的视觉伺服抓取过程
        """
        try:
            # 1. 移动到初始观察位置
            self.robot.go_home()
            
            # 2. 执行视觉伺服控制
            if not self.visual_servo_to_target():
                print("Visual servoing failed")
                return False
                
            # 3. 打开夹爪
            self.robot.open_gripper()
            
            # 4. 向下移动进行抓取
            current_pose = self.robot.get_current_tool_pos()
            grasp_pose = current_pose.copy()
            grasp_pose[2] -= 0.05  # 向下移动5cm
            self.robot.move_l(grasp_pose.tolist(), k_acc=0.3, k_vel=0.3)
            
            # 5. 闭合夹爪
            self.robot.close_gripper()
            
            # 6. 检查抓取是否成功
            if not self.robot.check_grasp():
                print("Grasp failed")
                return False
                
            # 7. 提起物体
            lift_pose = grasp_pose.copy()
            lift_pose[2] += 0.1  # 向上提起10cm
            self.robot.move_l(lift_pose.tolist(), k_acc=0.3, k_vel=0.3)
            
            # 8. 返回到安全位置
            self.robot.go_home()
            
            return True
            
        except Exception as e:
            print(f"Grasp execution failed: {e}")
            return False
            
def main():
    # 创建视觉伺服抓取系统实例
    vs_grasping = VisualServoGrasping(
        robot_ip="192.168.50.100",
        model_path="best.pt"  # 使用训练好的YOLOv8模型
    )
    
    # 执行抓取
    success = vs_grasping.execute_grasp()
    
    if success:
        print("Grasp completed successfully")
    else:
        print("Grasp failed")

if __name__ == "__main__":
    main() 