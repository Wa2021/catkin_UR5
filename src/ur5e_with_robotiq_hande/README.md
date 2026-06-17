# ur5e_with_robotiq_hande

这是一个包含 UR5e 机械臂与 Robotiq Hand-E 夹爪组合模型的 ROS 功能包。

## 功能
该功能包主要用于提供机器人的 URDF/Xacro 描述文件，用于仿真或可视化。

## 主要文件
- `urdf/ur5e_with_robotiq_hande.xacro`: 组合了 UR5e 和 Hand-E 的主 Xacro 文件。
- `urdf/robotiq_hande_gripper.xacro`: Hand-E 夹爪的定义文件。

## 使用方法
通常可以在 launch 文件中调用此包中的 xacro 文件来加载机器人模型到参数服务器。
