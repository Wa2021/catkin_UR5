# 眼在手YOLO避障抓取系统 (Eye-in-Hand YOLO Grasp System)

这是一个集成了 **YOLOv8物体检测**、**真实点云过滤** 和 **MoveIt路径规划** 的机器人系统。专为UR5机器人的眼在手（Eye-in-Hand）配置设计。

系统通过实时检测目标物体（如键盘），将其从点云中剔除，从而让MoveIt能够生成忽略目标的障碍物地图，实现安全的避障抓取。

---

## 🚀 快速启动

由于YOLOv8需要较新的Python库，而ROS Noetic依赖旧版库，我们需要两个终端分别启动。

### 第一步：启动基础系统 (Terminal 1)
此步骤启动相机、MoveIt、机械臂驱动以及点云处理节点。
```bash
# 在ROS原生环境中
source /opt/ros/noetic/setup.bash
source ~/catkin_UR5/devel/setup.bash

roslaunch ur5_gripper_moveit_config eye_in_hand_yolo_grasp.launch
```
> ✅ **启动成功标志**：RViz打开，机械臂模型加载，且终端显示 `[INFO] Snapshot controller started`.

### 第二步：启动YOLO检测 (Terminal 2)
此步骤启动物体识别，开始过滤点云。
```bash
# 激活Conda环境
conda activate arm_grasp
source /opt/ros/noetic/setup.bash
source ~/catkin_UR5/devel/setup.bash

rosrun ur5_gripper_moveit_config yolo_moveit_detector.py
```
> ✅ **启动成功标志**：终端显示 `YOLO model loaded`，并且RViz中的Octomap障碍物开始更新（目标物体区域变空）。

---

## 🛠️ 调试模式与画面冻结 (Snapshot Mode)

本系统包含一个强大的调试功能，允许你**冻结**相机的点云数据。这在以下场景非常有用：
1.  **防止运动模糊**：机械臂移动时相机画面会因运动产生残影，导致障碍物地图混乱。
2.  **静态规划**：移动到观察位置后，冻结画面，确保规划路径时环境保持静止。
3.  **算法调试**：固定输入数据，调试过滤算法。

### 自动冻结机制
系统默认配置为 **8秒后自动冻结**。
*   启动后，你有8秒时间调整相机位置。
*   8秒后，终端提示 `Auto-freezing point cloud`，此时障碍物地图不再更新。

### 手动控制指令
你可以随时通过ROS Service控制冻结状态：

**1. 冻结画面 (停止更新)**
```bash
rosservice call /snapshot/freeze
```

**2. 解冻画面 (恢复实时更新)**
```bash
rosservice call /snapshot/unfreeze
```

**3. 清除所有障碍物 (用于重置)**
```bash
rosservice call /snapshot/clear
```
*这会发送空点云，MoveIt将清除当前Octomap中的所有障碍物。*

---

## 🌟 核心功能

*   **实时环境感知**：使用RealSense相机获取RGB-D数据。
*   **智能目标过滤**：通过YOLOv8识别目标物体（如键盘），并从点云中**剔除**其对应的点云，防止其被误识别为障碍物。
*   **自动障碍物生成**：MoveIt基于过滤后的点云自动构建Octomap障碍物环境。
*   **稳定的调试模式**：集成`snapshot_controller`，支持“冻结”当前视野，方便静态调试和路径规划。
*   **双环境架构**：分离ROS原生环境与YOLO Conda环境，解决依赖冲突。

---

## 🏗️ 系统架构

系统主要由两条数据流组成：

1.  **视觉流 (arm_grasp环境)**：
    *   `yolo_moveit_detector.py`: 运行YOLOv8，检测目标物体2D边界框，发布 `/yolo/detection_results`。

2.  **控制流 (ROS原生环境)**：
    *   `snapshot_controller`: 接收相机原始点云，支持“冻结”功能（保持最后一帧点云不变），发布 `/camera/depth/color/points_relay`。
    *   `pointcloud_target_filter`: 接收中继点云 + YOLO检测框。**移除**检测框区域内的点云数据，发布 `/camera/depth/color/points_filtered`。
    *   MoveIt: 接收过滤后的点云，生成环境障碍物。

### 数据处理流程
1.  **Camera** 采集原始点云。
2.  **Snapshot Controller** 处理点云流（实时转发或冻结）。
3.  **YOLO Detector** 检测图像中的目标（如键盘），输出边界框。
4.  **Filter** 结合YOLO边界框和点云，将目标区域挖空。
5.  **MoveIt** 接收处理后的点云，将剩余部分作为障碍物处理。

---

## ⚙️ 关键配置说明

### 1. 修改主要参数
编辑文件：`src/universal_robot/ur5_gripper_moveit_config/launch/eye_in_hand_yolo_grasp.launch`

*   **target_objects**: 定义哪些物体是“目标”（不作为障碍物）。
    ```xml
    <rosparam param="target_objects">['keyboard']</rosparam>
    ```
*   **obstacle_mode**: 
    *   `blacklist`: 除目标外，其余都是障碍物（默认）。
    *   `whitelist`: 仅指定列表内的物体是障碍物。

### 2. 修改YOLO模型
默认使用 `yolov8n.pt`。如需更强性能或自定义模型，请修改 `yolo_moveit_detector.py` 中的模型路径。

---

## ❓ 常见问题 (Troubleshooting)

**Q: 障碍物地图没有显示？**
*   检查RViz左侧Displays面板，确保 `PlanningScene` 组件已启用。
*   检查话题 `/planning_scene` 是否有数据：`rostopic hz /planning_scene`。

**Q: YOLO节点报错 "ModuleNotFoundError"?**
*   确保你在Terminal 2中正确激活了 `arm_grasp` 环境。

**Q: 机械臂移动时Octomap拖影严重？**
*   这是眼在手系统的常见问题。请使用 **冻结模式**：
    1. 移动机械臂到观察点。
    2. 等待画面稳定。
    3. `rosservice call /snapshot/freeze`。
    4. 再进行规划和执行。

**Q: 目标物体（键盘）仍然被显示为障碍物？**
*   检查YOLO是否检测到了键盘（查看 `/yolo/detection_image`）。
*   检查检测框是否准确覆盖了物体。
*   确保 `target_objects` 参数拼写正确（['keyboard']）。

---

## 📝 维护记录

*   **当前版本**: v2.1 (Merged Debug Guide)
*   **最后更新**: 2026-01-16
*   **维护者**: UESTC
